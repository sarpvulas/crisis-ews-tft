#!/usr/bin/env python
"""5-seed study: RA-TFT vs XGBoost, sequential training (no GPU contention).

Purpose: Determine true mean ROC-AUC for RA-TFT by removing parallel training
noise. XGBoost included as stable baseline reference.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.configs import TFTConfig
from models.ra_tft import RegimeAwareTFT
from models.tft import TemporalFusionTransformer
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.crisis_loss import CrisisAwareLoss
from training.dataset import PanelCrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_pr_auc, compute_roc_auc, compute_calibration_error,
    compute_brier_score, PostHocCalibrator,
)

PANEL_PATH = "data/processed/panel/panel_features.parquet"
ENC, DEC = 252, 63
TEST_START = "2015-01-01"
SEEDS = [42, 123, 456, 789, 1024]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_datasets(seed=42):
    """Load panel datasets with a given seed for val split."""
    ds_train, ds_val = PanelCrisisDataset.train_val_split(
        PANEL_PATH, train_end=TEST_START, val_ratio=0.15, seed=seed,
        encoder_steps=ENC, decoder_steps=DEC,
    )
    ds_test = PanelCrisisDataset(
        PANEL_PATH, split="test", encoder_steps=ENC, decoder_steps=DEC,
        train_end=TEST_START, val_end=TEST_START,
        feature_scaler=ds_train.feature_scaler,
        exclude_crisis_windows=False,
    )
    return ds_train, ds_val, ds_test


def evaluate_pytorch(model, dataset, device):
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=dataset.collate_fn, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, targets in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch_dev)
            probs = torch.sigmoid(outputs["predicted"].squeeze(-1))
            all_preds.append(probs.cpu().numpy().flatten())
            all_labels.append(targets["ews_label"].numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_scores, y_true):
    m = {}
    if len(np.unique(y_true)) >= 2:
        m["roc_auc"] = compute_roc_auc(y_true, y_scores)
        m["pr_auc"] = compute_pr_auc(y_true, y_scores)
        m["calibration_error"] = compute_calibration_error(y_true, y_scores)
        m["brier_score"] = compute_brier_score(y_true, y_scores)
    return m


def train_ra_tft(seed, ds_train, ds_val, ds_test, dims, n_markets, pos_weight):
    """Train RA-TFT with a given seed, sequential, full GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_hist = dims["num_historical_numeric"]
    n_future = dims["num_future_numeric"]

    cfg = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        num_static_categorical=1, static_categorical_cardinalities=[n_markets],
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.2,
        encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
        num_regime_states=3, use_regime_module=True, use_regime_attention=True,
    )
    model = RegimeAwareTFT(cfg)

    ckpt_dir = f"checkpoints/seed_study/ra_tft_seed{seed}"
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": 3e-4, "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": ckpt_dir,
                "early_stopping_patience": 15},
        train_dataset=ds_train, val_dataset=ds_val,
    )

    history = trainer.train(epochs=100)
    best_val = min(history["val_loss"])
    n_ep = len(history["train_loss"])
    print(f"  [RA-TFT seed={seed}] Done: best_val={best_val:.4f}, epochs={n_ep}")

    # Restore best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [RA-TFT seed={seed}] Restored best checkpoint (epoch {ckpt['epoch']+1})")

    # Evaluate
    y_val_s, y_val_t = evaluate_pytorch(model, ds_val, DEVICE)
    y_s, y_t = evaluate_pytorch(model, ds_test, DEVICE)
    metrics = compute_metrics(y_s, y_t)

    # Calibration
    cal = PostHocCalibrator(method="platt")
    cal.fit(y_val_s, y_val_t)
    y_cal = cal.transform(y_s)
    cal_metrics = compute_metrics(y_cal, y_t)
    metrics["calibrated_platt_ece"] = cal_metrics.get("calibration_error")
    metrics["calibrated_platt_brier"] = cal_metrics.get("brier_score")

    return metrics


def train_vanilla_tft(seed, ds_train, ds_val, ds_test, dims, n_markets, pos_weight):
    """Train Vanilla TFT with a given seed for comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_hist = dims["num_historical_numeric"]
    n_future = dims["num_future_numeric"]

    cfg = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        num_static_categorical=1, static_categorical_cardinalities=[n_markets],
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.2,
        encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
        use_regime_module=False, use_regime_attention=False,
    )
    model = TemporalFusionTransformer(cfg)

    ckpt_dir = f"checkpoints/seed_study/vanilla_tft_seed{seed}"
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": 3e-4, "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": ckpt_dir,
                "early_stopping_patience": 15},
        train_dataset=ds_train, val_dataset=ds_val,
    )

    history = trainer.train(epochs=100)
    best_val = min(history["val_loss"])
    n_ep = len(history["train_loss"])
    print(f"  [Vanilla TFT seed={seed}] Done: best_val={best_val:.4f}, epochs={n_ep}")

    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [Vanilla TFT seed={seed}] Restored best checkpoint (epoch {ckpt['epoch']+1})")

    y_s, y_t = evaluate_pytorch(model, ds_test, DEVICE)
    return compute_metrics(y_s, y_t)


def train_xgboost(seed, ds_train, ds_val, ds_test):
    """Train XGBoost with a given seed."""
    np.random.seed(seed)

    sk_loader = DataLoader(ds_train, batch_size=256, shuffle=False,
                           collate_fn=ds_train.collate_fn, num_workers=0)
    X_chunks, y_chunks = [], []
    for batch, targets in sk_loader:
        X_chunks.append(batch["historical_ts_numeric"].numpy())
        y_chunks.append(targets["ews_label"].numpy())
    X_hist = np.concatenate(X_chunks)
    y = np.concatenate(y_chunks)

    xgb = XGBoostBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC,
                           random_state=seed)

    xgb.fit(X_hist, y)

    sk_test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                                collate_fn=ds_test.collate_fn, num_workers=0)
    xgb_preds, xgb_labels = [], []
    for batch, targets in sk_test_loader:
        X = batch["historical_ts_numeric"].numpy()
        preds = xgb.predict_from_arrays(X)
        xgb_preds.append(preds.flatten())
        xgb_labels.append(targets["ews_label"].numpy().flatten())

    y_s = np.concatenate(xgb_preds)
    y_t = np.concatenate(xgb_labels)
    return compute_metrics(y_s, y_t)


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Models: RA-TFT, Vanilla TFT, XGBoost (sequential, no GPU contention)\n")

    # Load datasets once (same val split seed=42 for all, model seed varies)
    print("Loading datasets...")
    ds_train, ds_val, ds_test = load_datasets(seed=42)
    dims = ds_train.get_feature_dims()
    n_markets = ds_train.num_markets

    total_ews = sum(md["ews"].sum() for md in ds_train.market_data.values())
    total_samples = sum(len(md["ews"]) for md in ds_train.market_data.values())
    pos_rate = total_ews / max(total_samples, 1)
    pos_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 10.0)
    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
    print(f"  Pos rate: {pos_rate:.3f}, pos_weight: {pos_weight:.1f}\n")

    all_results = {"seeds": SEEDS, "ra_tft": [], "vanilla_tft": [], "xgboost": []}

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*70}")

        # RA-TFT (sequential, full GPU)
        print(f"\n  Training RA-TFT (seed={seed})...")
        ra_metrics = train_ra_tft(seed, ds_train, ds_val, ds_test, dims, n_markets, pos_weight)
        all_results["ra_tft"].append(ra_metrics)
        print(f"  RA-TFT: ROC={ra_metrics.get('roc_auc',0):.4f} ECE={ra_metrics.get('calibration_error',0):.4f}")

        # Vanilla TFT (sequential, full GPU)
        print(f"\n  Training Vanilla TFT (seed={seed})...")
        vtft_metrics = train_vanilla_tft(seed, ds_train, ds_val, ds_test, dims, n_markets, pos_weight)
        all_results["vanilla_tft"].append(vtft_metrics)
        print(f"  Vanilla TFT: ROC={vtft_metrics.get('roc_auc',0):.4f}")

        # XGBoost
        print(f"\n  Training XGBoost (seed={seed})...")
        xgb_metrics = train_xgboost(seed, ds_train, ds_val, ds_test)
        all_results["xgboost"].append(xgb_metrics)
        print(f"  XGBoost: ROC={xgb_metrics.get('roc_auc',0):.4f}")

    # === Summary ===
    print(f"\n\n{'='*90}")
    print("5-SEED STUDY RESULTS (Sequential Training)")
    print(f"{'='*90}")
    print(f"{'Model':<18} {'Seed':>6} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'Brier':>10}")
    print("-" * 90)

    for model_name, key in [("RA-TFT", "ra_tft"), ("Vanilla TFT", "vanilla_tft"), ("XGBoost", "xgboost")]:
        for i, seed in enumerate(SEEDS):
            m = all_results[key][i]
            roc = f"{m.get('roc_auc', 0):.4f}"
            pr = f"{m.get('pr_auc', 0):.4f}"
            ece = f"{m.get('calibration_error', 0):.4f}"
            brier = f"{m.get('brier_score', 0):.4f}"
            print(f"{model_name:<18} {seed:>6} {roc:>10} {pr:>10} {ece:>10} {brier:>10}")
        print()

    # Averages
    print("-" * 90)
    print(f"{'Model':<18} {'Mean ROC':>10} {'Std ROC':>10} {'Mean PR':>10} {'Mean ECE':>10}")
    print("-" * 90)
    for model_name, key in [("RA-TFT", "ra_tft"), ("Vanilla TFT", "vanilla_tft"), ("XGBoost", "xgboost")]:
        rocs = [m.get("roc_auc", 0) for m in all_results[key]]
        prs = [m.get("pr_auc", 0) for m in all_results[key]]
        eces = [m.get("calibration_error", 0) for m in all_results[key]]
        print(f"{model_name:<18} {np.mean(rocs):>10.4f} {np.std(rocs):>10.4f} "
              f"{np.mean(prs):>10.4f} {np.mean(eces):>10.4f}")

    # RA-TFT calibrated
    if all_results["ra_tft"][0].get("calibrated_platt_ece") is not None:
        cal_eces = [m["calibrated_platt_ece"] for m in all_results["ra_tft"]]
        cal_briers = [m["calibrated_platt_brier"] for m in all_results["ra_tft"]]
        print(f"\n{'RA-TFT (Platt)':<18} {'Mean cal ECE':>10}: {np.mean(cal_eces):.4f}  "
              f"{'Mean cal Brier':>14}: {np.mean(cal_briers):.4f}")

    print("=" * 90)

    # Save
    os.makedirs("output", exist_ok=True)
    with open("output/seed_study.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to output/seed_study.json")


if __name__ == "__main__":
    main()
