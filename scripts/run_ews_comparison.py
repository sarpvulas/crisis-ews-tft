#!/usr/bin/env python
"""Train and evaluate all models on the EWS task, report comparison."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

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
    compute_early_warning_lead_time, compute_bootstrap_ci,
)

DATA_DIR = "data/processed"
ENC, DEC = 252, 63
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_datasets():
    ds_train = CrisisDataset(split="train", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC)
    ds_val = CrisisDataset(split="val", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC,
                           feature_scaler=ds_train.feature_scaler)
    ds_test = CrisisDataset(split="test", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC,
                            feature_scaler=ds_train.feature_scaler, exclude_crisis_windows=False)
    return ds_train, ds_val, ds_test


def evaluate_pytorch_model(model, ds_test):
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


def evaluate_sklearn_model(model, ds_test):
    loader = DataLoader(ds_test, batch_size=256, shuffle=False, collate_fn=ds_test.collate_fn)

    all_preds, all_labels = [], []
    for batch, targets in loader:
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        preds = model.predict_from_arrays(X_hist, X_static)
        all_preds.append(preds.flatten())
        all_labels.append(targets["ews_label"].numpy().flatten())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_scores, y_true):
    metrics = {"n_samples": len(y_true), "positive_rate": float(y_true.mean())}
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = compute_roc_auc(y_true, y_scores)
        metrics["pr_auc"] = compute_pr_auc(y_true, y_scores)
        metrics["calibration_error"] = compute_calibration_error(y_true, y_scores)
        pr_ci = compute_bootstrap_ci(compute_pr_auc, y_true, y_scores, n_bootstrap=500)
        metrics["pr_auc_ci"] = f"[{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]"
    return metrics


def train_pytorch(model_name, model, ds_train, ds_val):
    loss_fn = CrisisAwareLoss()
    trainer = Trainer(
        model=model,
        config={"lr": 1e-3, "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": f"checkpoints/ews_{model_name}"},
        train_dataset=ds_train, val_dataset=ds_val, loss_fn=loss_fn,
    )
    history = trainer.train(epochs=50)
    best_val = min(history["val_loss"])
    print(f"  {model_name}: best val_loss={best_val:.4f}, stopped at epoch {len(history['train_loss'])}")
    return model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading data...")
    ds_train, ds_val, ds_test = load_datasets()
    n_hist = ds_train._num_historical
    n_future = ds_train._num_future

    results = {}

    # === 1. RA-TFT ===
    print("\n[1/5] Training RA-TFT...")
    cfg = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
        encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
        num_regime_states=3, use_regime_module=True, use_regime_attention=True,
    )
    ra_tft = RegimeAwareTFT(cfg)
    train_pytorch("ra-tft", ra_tft, ds_train, ds_val)
    y_scores, y_true = evaluate_pytorch_model(ra_tft, ds_test)
    results["RA-TFT"] = compute_metrics(y_scores, y_true)

    # === 2. Vanilla TFT ===
    print("\n[2/5] Training Vanilla TFT...")
    torch.manual_seed(SEED)
    cfg_v = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
        encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
        use_regime_module=False, use_regime_attention=False,
    )
    tft = TemporalFusionTransformer(cfg_v)
    train_pytorch("tft", tft, ds_train, ds_val)
    y_scores, y_true = evaluate_pytorch_model(tft, ds_test)
    results["Vanilla TFT"] = compute_metrics(y_scores, y_true)

    # === 3. LSTM ===
    print("\n[3/5] Training LSTM...")
    torch.manual_seed(SEED)
    lstm = LSTMBaseline(
        input_dim=n_hist, hidden_dim=64, num_layers=2,
        task="A", encoder_steps=ENC, decoder_steps=DEC, num_quantiles=1,
    )
    train_pytorch("lstm", lstm, ds_train, ds_val)
    y_scores, y_true = evaluate_pytorch_model(lstm, ds_test)
    results["LSTM"] = compute_metrics(y_scores, y_true)

    # === 4 & 5: sklearn baselines — collect training data in batches ===
    print("\n[4/5] Collecting training data for sklearn baselines...")
    sk_loader = DataLoader(ds_train, batch_size=256, shuffle=False, collate_fn=ds_train.collate_fn)
    X_chunks, y_chunks = [], []
    for batch, targets in sk_loader:
        X_chunks.append(batch["historical_ts_numeric"].numpy())
        y_chunks.append(targets["ews_label"].numpy())
    X_hist = np.concatenate(X_chunks)
    y = np.concatenate(y_chunks)
    print(f"  Training data: {X_hist.shape[0]} samples, {X_hist.shape[1]*X_hist.shape[2]} features (flattened)")

    print("  Training XGBoost...")
    xgb = XGBoostBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC)
    xgb.fit(X_hist, y)
    y_scores, y_true = evaluate_sklearn_model(xgb, ds_test)
    results["XGBoost"] = compute_metrics(y_scores, y_true)
    print(f"  XGBoost done: ROC-AUC={results['XGBoost'].get('roc_auc', 0):.4f}")

    print("\n[5/5] Training Logistic Regression...")
    lr_model = LogisticBaseline(task="A", encoder_steps=ENC, decoder_steps=DEC)
    lr_model.fit(X_hist, y)
    y_scores, y_true = evaluate_sklearn_model(lr_model, ds_test)
    results["Logistic Reg."] = compute_metrics(y_scores, y_true)
    print(f"  Logistic Reg done: ROC-AUC={results['Logistic Reg.'].get('roc_auc', 0):.4f}")

    # === Print comparison table ===
    print("\n" + "=" * 80)
    print("EWS MODEL COMPARISON (Test Set)")
    print("=" * 80)
    print(f"{'Model':<18} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'PR-AUC CI':>20}")
    print("-" * 80)
    for name, m in results.items():
        roc = f"{m.get('roc_auc', 0):.4f}"
        pr = f"{m.get('pr_auc', 0):.4f}"
        ece = f"{m.get('calibration_error', 0):.4f}"
        ci = m.get("pr_auc_ci", "N/A")
        print(f"{name:<18} {roc:>10} {pr:>10} {ece:>10} {ci:>20}")
    print("=" * 80)

    # Save
    out_path = "output/ews_comparison.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
