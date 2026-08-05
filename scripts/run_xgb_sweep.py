#!/usr/bin/env python
"""XGBoost hyperparameter grid search on panel EWS data.

Fast — runs on CPU or GPU, each config takes ~1-2 min.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from itertools import product
from torch.utils.data import DataLoader

from models.baselines.xgboost_baseline import XGBoostBaseline
from training.dataset import PanelCrisisDataset
from training.evaluation import (
    compute_roc_auc, compute_pr_auc, compute_calibration_error,
    compute_brier_score, PostHocCalibrator,
)

PANEL_PATH = "data/processed/panel/panel_features.parquet"
ENC, DEC = 252, 63
TEST_START = "2015-01-01"

# Grid search space
GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1.0, 3.7],
}


def load_datasets():
    ds_train, ds_val = PanelCrisisDataset.train_val_split(
        PANEL_PATH, train_end=TEST_START, val_ratio=0.15, seed=42,
        encoder_steps=ENC, decoder_steps=DEC,
    )
    ds_test = PanelCrisisDataset(
        PANEL_PATH, split="test", encoder_steps=ENC, decoder_steps=DEC,
        train_end=TEST_START, val_end=TEST_START,
        feature_scaler=ds_train.feature_scaler,
        exclude_crisis_windows=False,
    )
    return ds_train, ds_val, ds_test


def extract_arrays(ds):
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        collate_fn=ds.collate_fn, num_workers=0)
    X_chunks, y_chunks = [], []
    for batch, targets in loader:
        X_chunks.append(batch["historical_ts_numeric"].numpy())
        y_chunks.append(targets["ews_label"].numpy())
    return np.concatenate(X_chunks), np.concatenate(y_chunks)


def main():
    print("Loading datasets...")
    ds_train, ds_val, ds_test = load_datasets()
    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    print("Extracting arrays...")
    X_train, y_train = extract_arrays(ds_train)
    X_val, y_val = extract_arrays(ds_val)
    X_test, y_test = extract_arrays(ds_test)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Generate all combinations
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    total = len(combos)
    print(f"\nGrid search: {total} combinations")

    results = []
    best_roc = 0
    best_config = None

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        t0 = time.time()

        try:
            xgb = XGBoostBaseline(
                task="A", encoder_steps=ENC, decoder_steps=DEC,
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                scale_pos_weight=config["scale_pos_weight"],
                random_state=42,
            )
            xgb.fit(X_train, y_train)

            # Val predictions for calibration
            val_preds = xgb.predict_from_arrays(X_val).flatten()
            val_labels = y_val.flatten()

            # Test predictions
            test_preds = xgb.predict_from_arrays(X_test).flatten()
            test_labels = y_test.flatten()

            roc = compute_roc_auc(test_labels, test_preds)
            pr = compute_pr_auc(test_labels, test_preds)
            ece = compute_calibration_error(test_labels, test_preds)
            brier = compute_brier_score(test_labels, test_preds)

            # Platt calibration
            cal = PostHocCalibrator(method="platt")
            cal.fit(val_preds, val_labels)
            cal_preds = cal.transform(test_preds)
            cal_ece = compute_calibration_error(test_labels, cal_preds)
            cal_brier = compute_brier_score(test_labels, cal_preds)

            elapsed = time.time() - t0

            result = {**config, "roc_auc": roc, "pr_auc": pr,
                      "calibration_error": ece, "brier_score": brier,
                      "cal_ece": cal_ece, "cal_brier": cal_brier,
                      "time_sec": elapsed}
            results.append(result)

            if roc > best_roc:
                best_roc = roc
                best_config = config

            if (i + 1) % 20 == 0 or roc > 0.63:
                print(f"  [{i+1}/{total}] ROC={roc:.4f} PR={pr:.4f} | "
                      f"n_est={config['n_estimators']} depth={config['max_depth']} "
                      f"lr={config['learning_rate']} sub={config['subsample']} "
                      f"col={config['colsample_bytree']} pw={config['scale_pos_weight']} "
                      f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1}/{total}] FAILED: {e}")
            continue

    # Sort by ROC
    results.sort(key=lambda x: x.get("roc_auc", 0), reverse=True)

    # Print top 20
    print(f"\n{'='*120}")
    print(f"TOP 20 XGBoost CONFIGURATIONS (out of {total})")
    print(f"{'='*120}")
    print(f"{'Rank':<6} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'CalECE':>10} "
          f"{'n_est':>8} {'depth':>6} {'lr':>8} {'sub':>6} {'col':>6} {'pw':>6} {'Time':>8}")
    print("-" * 120)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<6} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} "
              f"{r['calibration_error']:>10.4f} {r['cal_ece']:>10.4f} "
              f"{r['n_estimators']:>8} {r['max_depth']:>6} {r['learning_rate']:>8.2f} "
              f"{r['subsample']:>6.1f} {r['colsample_bytree']:>6.1f} "
              f"{r['scale_pos_weight']:>6.1f} {r['time_sec']:>7.1f}s")

    print("=" * 120)
    print(f"\nBaseline (default): ROC=0.6213")
    print(f"Best found:         ROC={results[0]['roc_auc']:.4f}")
    print(f"Improvement:        +{results[0]['roc_auc'] - 0.6213:.4f}")

    # Save all results
    os.makedirs("output", exist_ok=True)
    with open("output/xgb_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll {len(results)} results saved to output/xgb_sweep.json")


if __name__ == "__main__":
    main()
