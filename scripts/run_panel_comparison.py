#!/usr/bin/env python
"""Train and evaluate all models on the multi-market panel EWS task.

Fixes:
  - Val split: random holdout from training windows (same crisis distribution)
  - Parallel GPU training: 4 models train simultaneously on one GPU
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.baselines.lstm_baseline import LSTMBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.crisis_loss import CrisisAwareLoss
from training.dataset import PanelCrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_pr_auc, compute_roc_auc, compute_calibration_error,
    compute_brier_score, compute_bootstrap_ci, PostHocCalibrator,
)

PANEL_PATH = "data/processed/panel/panel_features.parquet"
ENC, DEC = 252, 63
TEST_START = "2015-01-01"  # Everything after this is test

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_datasets():
    """Random holdout val from training period — both have crises."""
    ds_train, ds_val = PanelCrisisDataset.train_val_split(
        PANEL_PATH,
        train_end=TEST_START,
        val_ratio=0.15,
        seed=42,
        encoder_steps=ENC,
        decoder_steps=DEC,
    )
    ds_test = PanelCrisisDataset(
        PANEL_PATH, split="test", encoder_steps=ENC, decoder_steps=DEC,
        train_end=TEST_START, val_end=TEST_START,
        feature_scaler=ds_train.feature_scaler,
        exclude_crisis_windows=False,
    )
    return ds_train, ds_val, ds_test


def evaluate_pytorch(model, ds_test, device):
    model.eval()
    model.to(device)
    loader = DataLoader(ds_test, batch_size=128, shuffle=False,
                        collate_fn=ds_test.collate_fn, num_workers=0)
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
    m = {"n_samples": len(y_true), "positive_rate": float(y_true.mean())}
    if len(np.unique(y_true)) >= 2:
        m["roc_auc"] = compute_roc_auc(y_true, y_scores)
        m["pr_auc"] = compute_pr_auc(y_true, y_scores)
        m["calibration_error"] = compute_calibration_error(y_true, y_scores)
        m["brier_score"] = compute_brier_score(y_true, y_scores)
        pr_ci = compute_bootstrap_ci(compute_pr_auc, y_true, y_scores, n_bootstrap=500)
        m["pr_auc_ci"] = f"[{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]"
    return m


def train_and_eval_pytorch(name, model, ds_train, ds_val, ds_test,
                           device, pos_weight, result_dict):
    """Train a single PyTorch model and store results (raw + calibrated)."""
    # Set seed in subprocess for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"  [{name}] Starting on {device}...")
    ckpt_dir = f"checkpoints/panel_{name.lower().replace(' ','-')}"
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": 3e-4, "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": device,
                "checkpoint_dir": ckpt_dir,
                "early_stopping_patience": 15},
        train_dataset=ds_train, val_dataset=ds_val,
    )
    history = trainer.train(epochs=100)
    best_val = min(history["val_loss"])
    n_ep = len(history["train_loss"])
    print(f"  [{name}] Done: best_val={best_val:.4f}, epochs={n_ep}")

    # Restore best checkpoint before evaluation
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [{name}] Restored best checkpoint (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")

    # Collect val predictions for calibrator fitting
    y_val_scores, y_val_true = evaluate_pytorch(model, ds_val, device)

    # Collect test predictions
    y_s, y_t = evaluate_pytorch(model, ds_test, device)
    metrics = compute_metrics(y_s, y_t)

    # Fit calibrators on val, apply to test
    for cal_method in ("platt", "isotonic"):
        cal = PostHocCalibrator(method=cal_method)
        cal.fit(y_val_scores, y_val_true)
        y_cal = cal.transform(y_s)
        cal_metrics = compute_metrics(y_cal, y_t)
        metrics[f"calibrated_{cal_method}"] = {
            "calibration_error": cal_metrics.get("calibration_error"),
            "brier_score": cal_metrics.get("brier_score"),
            "roc_auc": cal_metrics.get("roc_auc"),
        }
        print(f"  [{name}] {cal_method}: ECE={cal_metrics.get('calibration_error',0):.4f} "
              f"Brier={cal_metrics.get('brier_score',0):.4f}")

    result_dict[name] = metrics
    print(f"  [{name}] ROC={metrics.get('roc_auc',0):.4f} PR={metrics.get('pr_auc',0):.4f} "
          f"ECE={metrics.get('calibration_error',0):.4f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {DEVICE}")
    print("Loading panel datasets (random holdout val)...")
    ds_train, ds_val, ds_test = load_datasets()
    dims = ds_train.get_feature_dims()
    n_markets = ds_train.num_markets

    print(f"  Train: {len(ds_train)} windows")
    print(f"  Val:   {len(ds_val)} windows (random holdout, same crisis distribution)")
    print(f"  Test:  {len(ds_test)} windows")
    print(f"  Markets: {n_markets}, Features: {dims['num_historical_numeric']}")

    # Compute pos_weight
    total_ews = sum(md["ews"].sum() for md in ds_train.market_data.values())
    total_samples = sum(len(md["ews"]) for md in ds_train.market_data.values())
    pos_rate = total_ews / max(total_samples, 1)
    pos_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 10.0)
    print(f"  Pos rate: {pos_rate:.3f}, pos_weight: {pos_weight:.1f}")

    # === Build all PyTorch models ===
    n_hist = dims["num_historical_numeric"]
    n_future = dims["num_future_numeric"]

    def make_ra_tft():
        torch.manual_seed(42)
        cfg = TFTConfig(
            num_historical_numeric=n_hist, num_future_numeric=n_future,
            num_static_categorical=1, static_categorical_cardinalities=[n_markets],
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.2,
            encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
            num_regime_states=3, use_regime_module=True, use_regime_attention=True,
        )
        return RegimeAwareTFT(cfg)

    def make_tft():
        torch.manual_seed(42)
        cfg = TFTConfig(
            num_historical_numeric=n_hist, num_future_numeric=n_future,
            num_static_categorical=1, static_categorical_cardinalities=[n_markets],
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.2,
            encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
            use_regime_module=False, use_regime_attention=False,
        )
        return TemporalFusionTransformer(cfg)

    def make_lstm():
        torch.manual_seed(42)
        return LSTMBaseline(
            input_dim=n_hist, hidden_dim=64, num_layers=2,
            task="A", encoder_steps=ENC, decoder_steps=DEC, num_quantiles=1,
        )

    # === Parallel GPU training ===
    if DEVICE == "cuda" and torch.cuda.is_available():
        print("\n=== Parallel GPU Training (3 models) ===")
        mp.set_start_method("spawn", force=True)

        manager = mp.Manager()
        result_dict = manager.dict()

        models_spec = [
            ("RA-TFT", make_ra_tft()),
            ("Vanilla TFT", make_tft()),
            ("LSTM", make_lstm()),
        ]

        # Share datasets across processes (they're read-only)
        processes = []
        for name, model in models_spec:
            model.share_memory()
            p = mp.Process(
                target=train_and_eval_pytorch,
                args=(name, model, ds_train, ds_val, ds_test,
                      DEVICE, pos_weight, result_dict),
            )
            processes.append(p)

        # Start all simultaneously
        for p in processes:
            p.start()

        # Wait for all to complete
        for p in processes:
            p.join()

        results = dict(result_dict)
    else:
        # Sequential fallback for MPS/CPU
        print("\n=== Sequential Training ===")
        results = {}

        for name, model in [("RA-TFT", make_ra_tft()), ("Vanilla TFT", make_tft()), ("LSTM", make_lstm())]:
            print(f"\nTraining {name}...")
            result_holder = {}
            train_and_eval_pytorch(name, model, ds_train, ds_val, ds_test,
                                   DEVICE, pos_weight, result_holder)
            results.update(result_holder)

    # === XGBoost (CPU, sequential) ===
    print("\nTraining XGBoost...")
    sk_loader = DataLoader(ds_train, batch_size=256, shuffle=False,
                           collate_fn=ds_train.collate_fn, num_workers=0)
    X_chunks, y_chunks = [], []
    for batch, targets in sk_loader:
        X_chunks.append(batch["historical_ts_numeric"].numpy())
        y_chunks.append(targets["ews_label"].numpy())
    X_hist = np.concatenate(X_chunks)
    y = np.concatenate(y_chunks)
    print(f"  sklearn data: {X_hist.shape[0]} samples, {X_hist.shape[1]*X_hist.shape[2]} flat features")

    xgb = XGBoostBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC)
    xgb.fit(X_hist, y)

    # Val predictions for calibration
    sk_val_loader = DataLoader(ds_val, batch_size=256, shuffle=False,
                               collate_fn=ds_val.collate_fn, num_workers=0)
    xgb_val_preds, xgb_val_labels = [], []
    for batch, targets in sk_val_loader:
        X = batch["historical_ts_numeric"].numpy()
        preds = xgb.predict_from_arrays(X)
        xgb_val_preds.append(preds.flatten())
        xgb_val_labels.append(targets["ews_label"].numpy().flatten())
    y_val_s = np.concatenate(xgb_val_preds)
    y_val_t = np.concatenate(xgb_val_labels)

    # Test predictions
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

    xgb_metrics = compute_metrics(y_s, y_t)
    for cal_method in ("platt", "isotonic"):
        cal = PostHocCalibrator(method=cal_method)
        cal.fit(y_val_s, y_val_t)
        y_cal = cal.transform(y_s)
        cal_metrics = compute_metrics(y_cal, y_t)
        xgb_metrics[f"calibrated_{cal_method}"] = {
            "calibration_error": cal_metrics.get("calibration_error"),
            "brier_score": cal_metrics.get("brier_score"),
            "roc_auc": cal_metrics.get("roc_auc"),
        }
        print(f"  [XGBoost] {cal_method}: ECE={cal_metrics.get('calibration_error',0):.4f} "
              f"Brier={cal_metrics.get('brier_score',0):.4f}")
    results["XGBoost"] = xgb_metrics
    print(f"  XGBoost ROC={xgb_metrics.get('roc_auc',0):.4f} PR={xgb_metrics.get('pr_auc',0):.4f} "
          f"ECE={xgb_metrics.get('calibration_error',0):.4f}")

    # === Print comparison ===
    print(f"\n{'='*105}")
    print(f"MULTI-MARKET PANEL EWS COMPARISON (Test: {TEST_START}+)")
    print(f"{'='*105}")
    print(f"{'Model':<18} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'Brier':>10} {'PR-AUC CI':>20}")
    print("-" * 105)

    for name in ["RA-TFT", "Vanilla TFT", "LSTM", "XGBoost"]:
        if name not in results:
            print(f"{name:<18} {'FAILED':>10}")
            continue
        m = results[name]
        roc = f"{m.get('roc_auc', 0):.4f}"
        pr = f"{m.get('pr_auc', 0):.4f}"
        ece = f"{m.get('calibration_error', 0):.4f}"
        brier = f"{m.get('brier_score', 0):.4f}"
        ci = m.get("pr_auc_ci", "N/A")
        print(f"{name:<18} {roc:>10} {pr:>10} {ece:>10} {brier:>10} {ci:>20}")

    # === Print calibration results ===
    print(f"\n{'='*105}")
    print("POST-HOC CALIBRATION RESULTS")
    print(f"{'='*105}")
    print(f"{'Model':<18} {'Method':>10} {'ECE (raw)':>12} {'ECE (cal)':>12} "
          f"{'Brier (raw)':>12} {'Brier (cal)':>12} {'ROC (cal)':>10}")
    print("-" * 105)

    for name in ["RA-TFT", "Vanilla TFT", "LSTM", "XGBoost"]:
        if name not in results:
            continue
        m = results[name]
        raw_ece = m.get("calibration_error", 0)
        raw_brier = m.get("brier_score", 0)
        for cal_method in ("platt", "isotonic"):
            cal_key = f"calibrated_{cal_method}"
            if cal_key not in m:
                continue
            cm = m[cal_key]
            print(f"{name:<18} {cal_method:>10} {raw_ece:>12.4f} {cm['calibration_error']:>12.4f} "
                  f"{raw_brier:>12.4f} {cm['brier_score']:>12.4f} {cm['roc_auc']:>10.4f}")

    print("=" * 105)

    os.makedirs("output", exist_ok=True)
    with open("output/panel_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to output/panel_comparison.json")


if __name__ == "__main__":
    main()
