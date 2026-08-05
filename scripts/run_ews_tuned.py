#!/usr/bin/env python
"""Train tuned RA-TFT and TFT for EWS, compare with previous results."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.crisis_loss import CrisisAwareLoss
from training.dataset import CrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_pr_auc, compute_roc_auc, compute_calibration_error,
    compute_bootstrap_ci,
)

DATA_DIR = "data/processed"
ENC, DEC = 252, 63
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_datasets():
    ds_train = CrisisDataset(split="train", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC)
    ds_val = CrisisDataset(split="val", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC,
                           feature_scaler=ds_train.feature_scaler)
    ds_test = CrisisDataset(split="test", data_dir=DATA_DIR, encoder_steps=ENC, decoder_steps=DEC,
                            feature_scaler=ds_train.feature_scaler, exclude_crisis_windows=False)
    return ds_train, ds_val, ds_test


def evaluate_model(model, ds_test):
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

    y_scores = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = {"n_samples": len(y_true), "positive_rate": float(y_true.mean())}
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = compute_roc_auc(y_true, y_scores)
        metrics["pr_auc"] = compute_pr_auc(y_true, y_scores)
        metrics["calibration_error"] = compute_calibration_error(y_true, y_scores)
        pr_ci = compute_bootstrap_ci(compute_pr_auc, y_true, y_scores, n_bootstrap=500)
        metrics["pr_auc_ci"] = f"[{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]"
    return metrics


def train_model(name, model, ds_train, ds_val, lr, pos_weight, patience, epochs):
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model,
        config={
            "lr": lr, "batch_size": 64, "grad_clip": 1.0, "task": "B",
            "device": DEVICE, "checkpoint_dir": f"checkpoints/ews_tuned_{name}",
            "early_stopping_patience": patience,
        },
        train_dataset=ds_train, val_dataset=ds_val, loss_fn=loss_fn,
    )
    history = trainer.train(epochs=epochs)
    best_val = min(history["val_loss"])
    n_epochs = len(history["train_loss"])
    print(f"  {name}: best val={best_val:.4f}, epochs={n_epochs}, final_train={history['train_loss'][-1]:.4f}")
    return model


def main():
    print("Loading data...")
    ds_train, ds_val, ds_test = load_datasets()
    n_hist = ds_train._num_historical
    n_future = ds_train._num_future

    # Compute actual class ratio for pos_weight
    pos_rate = ds_train.ews_labels[~ds_train.in_crisis].mean()
    ideal_pos_weight = (1 - pos_rate) / pos_rate
    print(f"Positive rate (train, non-crisis): {pos_rate:.3f}, ideal pos_weight: {ideal_pos_weight:.1f}")

    results = {}

    # Tuned hyperparams:
    # - lr: 3e-4 (3x lower)
    # - pos_weight: actual class ratio (~4) instead of 10
    # - patience: 10 instead of 5
    # - epochs: 100 (let it train longer)
    # - dropout: 0.2 (more regularization)

    configs = [
        ("RA-TFT (tuned)", True, True),
        ("Vanilla TFT (tuned)", False, False),
    ]

    for name, use_regime, use_attn in configs:
        print(f"\nTraining {name}...")
        torch.manual_seed(42)
        np.random.seed(42)

        cfg = TFTConfig(
            num_historical_numeric=n_hist, num_future_numeric=n_future,
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2,
            dropout=0.2,
            encoder_steps=ENC, decoder_steps=DEC, task_type="ews", num_outputs=1,
            num_regime_states=3, use_regime_module=use_regime, use_regime_attention=use_attn,
        )
        if use_regime:
            model = RegimeAwareTFT(cfg)
        else:
            model = TemporalFusionTransformer(cfg)

        model = train_model(
            name, model, ds_train, ds_val,
            lr=3e-4,
            pos_weight=ideal_pos_weight,
            patience=10,
            epochs=100,
        )
        metrics = evaluate_model(model, ds_test)
        results[name] = metrics

    # Load previous results for comparison
    prev_path = "output/ews_comparison.json"
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)
    else:
        prev = {}

    # Print comparison
    print("\n" + "=" * 85)
    print("EWS COMPARISON: TUNED vs BASELINE (Test Set)")
    print("=" * 85)
    print(f"{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'PR-AUC CI':>20}")
    print("-" * 85)

    # Print tuned results
    for name, m in results.items():
        roc = f"{m.get('roc_auc', 0):.4f}"
        pr = f"{m.get('pr_auc', 0):.4f}"
        ece = f"{m.get('calibration_error', 0):.4f}"
        ci = m.get("pr_auc_ci", "N/A")
        print(f"{name:<25} {roc:>10} {pr:>10} {ece:>10} {ci:>20}")

    print("-" * 85)
    # Print previous baselines for context
    for name in ["RA-TFT", "Vanilla TFT", "LSTM", "XGBoost", "Logistic Reg."]:
        if name in prev:
            m = prev[name]
            roc = f"{m.get('roc_auc', 0):.4f}"
            pr = f"{m.get('pr_auc', 0):.4f}"
            ece = f"{m.get('calibration_error', 0):.4f}"
            ci = m.get("pr_auc_ci", "N/A")
            print(f"{name + ' (prev)':<25} {roc:>10} {pr:>10} {ece:>10} {ci:>20}")

    print("=" * 85)

    # Save
    out_path = "output/ews_comparison_tuned.json"
    combined = {**prev, **results}
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
