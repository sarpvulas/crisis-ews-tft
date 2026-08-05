#!/usr/bin/env python
"""Walk-forward crisis-anchored evaluation of all EWS models.

Each fold trains on all data before a crisis, then tests on that crisis.
Final metrics are averaged across folds — the gold standard for EWS evaluation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.baselines.lstm_baseline import LSTMBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.baselines.logistic_baseline import LogisticBaseline
from models.crisis_loss import CrisisAwareLoss
from training.dataset import CrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_pr_auc, compute_roc_auc, compute_calibration_error,
)

ENC, DEC = 252, 63
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FOLD_DIR = Path("data/processed/folds")


def load_fold(fold_name):
    """Load a fold's train/val/test datasets."""
    ds_train = CrisisDataset(split="train", data_dir=None, encoder_steps=ENC, decoder_steps=DEC)
    ds_train._load_fold(FOLD_DIR, fold_name, "train")

    ds_val = CrisisDataset(split="val", data_dir=None, encoder_steps=ENC, decoder_steps=DEC)
    ds_val._load_fold(FOLD_DIR, fold_name, "val", feature_scaler=ds_train.feature_scaler)

    ds_test = CrisisDataset(split="test", data_dir=None, encoder_steps=ENC, decoder_steps=DEC,
                            exclude_crisis_windows=False)
    ds_test._load_fold(FOLD_DIR, fold_name, "test", feature_scaler=ds_train.feature_scaler)

    return ds_train, ds_val, ds_test


def evaluate_pytorch(model, ds_test):
    model.eval()
    model.to(DEVICE)
    loader = DataLoader(ds_test, batch_size=64, shuffle=False, collate_fn=ds_test.collate_fn)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, targets in loader:
            batch_dev = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch_dev)
            probs = torch.sigmoid(outputs["predicted"].squeeze(-1))
            all_preds.append(probs.cpu().numpy().flatten())
            all_labels.append(targets["ews_label"].numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_sklearn(model, ds_test):
    loader = DataLoader(ds_test, batch_size=256, shuffle=False, collate_fn=ds_test.collate_fn)
    all_preds, all_labels = [], []
    for batch, targets in loader:
        X = batch["historical_ts_numeric"].numpy()
        preds = model.predict_from_arrays(X)
        all_preds.append(preds.flatten())
        all_labels.append(targets["ews_label"].numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_scores, y_true):
    m = {"n_samples": len(y_true), "positive_rate": float(y_true.mean())}
    if len(np.unique(y_true)) >= 2:
        m["roc_auc"] = compute_roc_auc(y_true, y_scores)
        m["pr_auc"] = compute_pr_auc(y_true, y_scores)
        m["calibration_error"] = compute_calibration_error(y_true, y_scores)
    return m


def train_pytorch(name, model, ds_train, ds_val, lr=3e-4, pos_weight=5.0):
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": lr, "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": f"checkpoints/wf_{name}",
                "early_stopping_patience": 10},
        train_dataset=ds_train, val_dataset=ds_val,
    )
    history = trainer.train(epochs=100)
    return model


def build_tft(n_hist, n_future, regime=False):
    cfg = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.2,
        encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
        num_regime_states=3, use_regime_module=regime, use_regime_attention=regime,
    )
    return RegimeAwareTFT(cfg) if regime else TemporalFusionTransformer(cfg)


def main():
    folds = sorted([f.stem.replace("_train", "")
                    for f in FOLD_DIR.glob("*_train.parquet")])
    print(f"Found {len(folds)} folds: {folds}")

    model_names = ["RA-TFT", "Vanilla TFT", "LSTM", "XGBoost", "Logistic Reg."]
    all_results = {name: [] for name in model_names}

    for fold_name in folds:
        print(f"\n{'='*70}")
        print(f"FOLD: {fold_name}")
        print(f"{'='*70}")

        ds_train, ds_val, ds_test = load_fold(fold_name)
        n_hist = ds_train._num_historical
        n_future = ds_train._num_future

        print(f"  Train: {len(ds_train)} windows, Val: {len(ds_val)}, Test: {len(ds_test)}")
        pos_rate = ds_train.ews_labels[~ds_train.in_crisis].mean()
        pos_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 10.0)
        print(f"  Pos rate: {pos_rate:.3f}, pos_weight: {pos_weight:.1f}")

        # --- RA-TFT ---
        print(f"\n  Training RA-TFT...")
        torch.manual_seed(42); np.random.seed(42)
        ra_tft = build_tft(n_hist, n_future, regime=True)
        train_pytorch(f"{fold_name}_ra-tft", ra_tft, ds_train, ds_val, pos_weight=pos_weight)
        y_s, y_t = evaluate_pytorch(ra_tft, ds_test)
        all_results["RA-TFT"].append(compute_metrics(y_s, y_t))

        # --- Vanilla TFT ---
        print(f"\n  Training Vanilla TFT...")
        torch.manual_seed(42); np.random.seed(42)
        tft = build_tft(n_hist, n_future, regime=False)
        train_pytorch(f"{fold_name}_tft", tft, ds_train, ds_val, pos_weight=pos_weight)
        y_s, y_t = evaluate_pytorch(tft, ds_test)
        all_results["Vanilla TFT"].append(compute_metrics(y_s, y_t))

        # --- LSTM ---
        print(f"\n  Training LSTM...")
        torch.manual_seed(42); np.random.seed(42)
        lstm = LSTMBaseline(input_dim=n_hist, hidden_dim=64, num_layers=2,
                            task="A", encoder_steps=ENC, decoder_steps=DEC, num_quantiles=1)
        train_pytorch(f"{fold_name}_lstm", lstm, ds_train, ds_val, pos_weight=pos_weight)
        y_s, y_t = evaluate_pytorch(lstm, ds_test)
        all_results["LSTM"].append(compute_metrics(y_s, y_t))

        # --- sklearn baselines ---
        print(f"\n  Training sklearn baselines...")
        sk_loader = DataLoader(ds_train, batch_size=256, shuffle=False, collate_fn=ds_train.collate_fn)
        X_chunks, y_chunks = [], []
        for batch, targets in sk_loader:
            X_chunks.append(batch["historical_ts_numeric"].numpy())
            y_chunks.append(targets["ews_label"].numpy())
        X_hist = np.concatenate(X_chunks)
        y = np.concatenate(y_chunks)

        xgb = XGBoostBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC)
        xgb.fit(X_hist, y)
        y_s, y_t = evaluate_sklearn(xgb, ds_test)
        all_results["XGBoost"].append(compute_metrics(y_s, y_t))

        lr_model = LogisticBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC)
        lr_model.fit(X_hist, y)
        y_s, y_t = evaluate_sklearn(lr_model, ds_test)
        all_results["Logistic Reg."].append(compute_metrics(y_s, y_t))

        # Print fold results
        print(f"\n  Fold '{fold_name}' results:")
        for name in model_names:
            m = all_results[name][-1]
            print(f"    {name:<18} ROC={m.get('roc_auc', 0):.3f}  PR={m.get('pr_auc', 0):.3f}  ECE={m.get('calibration_error', 0):.3f}")

    # === Aggregate across folds ===
    print(f"\n\n{'='*85}")
    print(f"WALK-FORWARD RESULTS (averaged across {len(folds)} folds)")
    print(f"{'='*85}")
    print(f"{'Model':<18} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10}   Per-fold ROC-AUC")
    print("-" * 85)

    summary = {}
    for name in model_names:
        fold_results = all_results[name]
        rocs = [m.get("roc_auc", 0) for m in fold_results]
        prs = [m.get("pr_auc", 0) for m in fold_results]
        eces = [m.get("calibration_error", 0) for m in fold_results]

        mean_roc = np.mean(rocs)
        mean_pr = np.mean(prs)
        mean_ece = np.mean(eces)
        per_fold = ", ".join([f"{r:.3f}" for r in rocs])

        print(f"{name:<18} {mean_roc:>10.4f} {mean_pr:>10.4f} {mean_ece:>10.4f}   [{per_fold}]")

        summary[name] = {
            "mean_roc_auc": round(mean_roc, 4),
            "mean_pr_auc": round(mean_pr, 4),
            "mean_ece": round(mean_ece, 4),
            "per_fold_roc": rocs,
            "per_fold_pr": prs,
            "folds": [f["meta"] for f in [{"meta": {"fold": fn}} for fn in folds]],
        }

    print("=" * 85)

    os.makedirs("output", exist_ok=True)
    with open("output/ews_walkforward.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to output/ews_walkforward.json")


if __name__ == "__main__":
    main()
