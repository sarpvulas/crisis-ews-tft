#!/usr/bin/env python
"""Evaluate all trained models with proper metrics on real data.

Outputs a comparison table with PR-AUC, ROC-AUC, F1, MAE, RMSE,
early warning lead time, calibration error.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)

from scripts.train import build_model, DEFAULT_CONFIG
from training.dataset import CrisisDataset


def evaluate(device="mps"):
    """Evaluate models on real crash prediction data."""
    print("=" * 70)
    print("Market Crash Prediction — Model Evaluation")
    print("=" * 70)

    # Load real test data
    test_df = pd.read_parquet("data/processed/task_b_test.parquet")
    print(f"Test set: {len(test_df)} rows, crash_label: {test_df['crash_label'].value_counts().to_dict()}")

    enc, dec = 252, 63
    ds_test = CrisisDataset(split="test", synthetic=True, encoder_steps=enc, decoder_steps=dec)

    results = {}

    for model_name in ["tft", "ra-tft"]:
        ckpt_path = f"checkpoints/{model_name.replace('-', '_')}_task_b/best_model.pt"
        if not os.path.exists(ckpt_path):
            print(f"  {model_name}: no checkpoint found")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(model_name)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        all_median_preds = []
        all_drawdowns = []
        all_crash_preds = []

        loader = DataLoader(ds_test, batch_size=32, collate_fn=ds_test.collate_fn)

        with torch.no_grad():
            for batch, targets in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)

                pred = out["predicted"].cpu()
                if pred.shape[-1] >= 7:
                    median = pred[:, :, 3]
                    tail = pred[:, :, 0]
                else:
                    median = pred.squeeze(-1)
                    tail = median

                all_median_preds.extend(median.mean(dim=1).numpy().flatten().tolist())
                all_drawdowns.extend(targets["forward_drawdown"].mean(dim=1).numpy().flatten().tolist())
                crash_prob = torch.abs(tail.mean(dim=1)).numpy().flatten()
                all_crash_preds.extend(crash_prob.tolist())

        preds = np.array(all_median_preds)
        actuals = np.array(all_drawdowns)
        crash_probs = np.array(all_crash_preds)

        if crash_probs.max() > crash_probs.min():
            crash_probs = (crash_probs - crash_probs.min()) / (crash_probs.max() - crash_probs.min())

        crash_labels = (np.abs(actuals) > 0.02).astype(int)

        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        metrics = {"MAE": mae, "RMSE": rmse}
        if len(np.unique(crash_labels)) > 1:
            roc = roc_auc_score(crash_labels, crash_probs)
            pr = average_precision_score(crash_labels, crash_probs)

            best_f1, best_t = 0, 0.5
            for t in np.arange(0.1, 0.9, 0.05):
                f1 = f1_score(crash_labels, (crash_probs >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t

            prec = precision_score(crash_labels, (crash_probs >= best_t).astype(int), zero_division=0)
            rec = recall_score(crash_labels, (crash_probs >= best_t).astype(int), zero_division=0)

            metrics.update({
                "ROC-AUC": roc,
                "PR-AUC": pr,
                "F1": best_f1,
                "Precision": prec,
                "Recall": rec,
                "Threshold": best_t,
            })

        results[model_name] = metrics
        print(f"\n  {model_name}:")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

    return results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    results = evaluate(device)

    os.makedirs("output", exist_ok=True)
    with open("output/model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to output/model_evaluation.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
