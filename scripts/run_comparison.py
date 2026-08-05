#!/usr/bin/env python
"""Core model comparison experiment for RA-TFT market crash prediction.

Trains 5 models x 5 seeds and evaluates with:
- PR-AUC, ROC-AUC, F1/Precision/Recall (at optimal threshold)
- Expected Calibration Error (ECE)
- Early Warning Lead Time

Usage:
    python scripts/run_comparison.py
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.baselines.logistic_baseline import LogisticBaseline
from models.baselines.lstm_baseline import LSTMBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.configs import TFTConfig
from models.crisis_loss import CrisisAwareLoss
from models.ra_tft import RegimeAwareTFT
from models.tft import TemporalFusionTransformer
from training.dataset import CrisisDataset
from training.trainer import Trainer

# ==============================================================================
# Configuration
# ==============================================================================

SEEDS = list(range(5))
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
ENCODER_STEPS = 252
DECODER_STEPS = 63
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

MODEL_NAMES = [
    "logistic",
    "xgboost",
    "lstm",
    "tft",
    "ra-tft",
]


def get_device():
    """Auto-detect best device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Dataset helpers
# ==============================================================================

def load_datasets(seed):
    """Load train/val/test datasets. Scaler fitted on train, reused for val/test."""
    ds_train = CrisisDataset(
        split="train", data_dir=DATA_DIR,
        encoder_steps=ENCODER_STEPS, decoder_steps=DECODER_STEPS,
    )
    scaler = ds_train.feature_scaler
    ds_val = CrisisDataset(
        split="val", data_dir=DATA_DIR,
        encoder_steps=ENCODER_STEPS, decoder_steps=DECODER_STEPS,
        feature_scaler=scaler,
    )
    ds_test = CrisisDataset(
        split="test", data_dir=DATA_DIR,
        encoder_steps=ENCODER_STEPS, decoder_steps=DECODER_STEPS,
        feature_scaler=scaler,
    )
    return ds_train, ds_val, ds_test


def get_test_labels(ds_test):
    """Extract crash_label from test dataset windows."""
    # Each window's target covers decoder_steps ahead
    # We take the max crash_label across the decoder window (any crash = positive)
    labels = []
    for i in range(len(ds_test)):
        _, targets = ds_test[i]
        # crash_label is stored in crisis_mask for Task B (derived from crash_label column)
        # Actually, let's read it from the raw df
        start = i
        enc_end = start + ENCODER_STEPS
        dec_end = enc_end + DECODER_STEPS
        labels.append(ds_test.crisis_masks[enc_end:dec_end].max())
    return np.array(labels)


def get_test_crash_labels_from_df(ds_test):
    """Get crash_label from the underlying dataframe, respecting valid_indices."""
    if "crash_label" in ds_test.df.columns:
        crash_col = ds_test.df["crash_label"].values.astype(np.float32)
    else:
        crash_col = ds_test.crisis_masks

    labels = []
    for i in range(len(ds_test)):
        start = ds_test.valid_indices[i]
        enc_end = start + ENCODER_STEPS
        dec_end = enc_end + DECODER_STEPS
        labels.append(float(crash_col[enc_end:dec_end].max()))
    return np.array(labels)


# ==============================================================================
# Model building
# ==============================================================================

def build_pytorch_model(model_name, ds_train):
    """Build a PyTorch model with auto-detected feature dims."""
    config_dict = {
        "num_historical_numeric": ds_train._num_historical,
        "num_static_numeric": ds_train._num_static,
        "num_future_numeric": ds_train._num_future,
        "encoder_steps": ENCODER_STEPS,
        "decoder_steps": DECODER_STEPS,
        "task_type": "regression",
        "num_outputs": 7,
        "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        "state_size": 64,
        "hidden_size": 64,
        "attention_heads": 4,
        "lstm_layers": 2,
        "dropout": 0.1,
        "num_regime_states": 3,
    }

    if model_name == "lstm":
        return LSTMBaseline(
            input_dim=ds_train._num_historical,
            hidden_dim=64, num_layers=2, task="B",
            encoder_steps=ENCODER_STEPS, decoder_steps=DECODER_STEPS,
            num_quantiles=7,
        )

    is_regime = model_name == "ra-tft"

    cfg = TFTConfig(
        **{k: v for k, v in config_dict.items() if k in TFTConfig.__dataclass_fields__},
        use_regime_module=is_regime,
        use_regime_attention=is_regime,
    )

    if model_name == "ra-tft":
        return RegimeAwareTFT(cfg)
    else:
        return TemporalFusionTransformer(cfg)


# ==============================================================================
# Training
# ==============================================================================

def _collect_numpy_windows(ds):
    """Efficiently collect all windows as numpy arrays from a CrisisDataset.

    Returns X_hist (N, encoder_steps, features), y_drawdown (N, decoder_steps),
    and X_static (N, static_features) or None.
    """
    n = len(ds)
    X_hist = np.stack([
        ds.historical_numeric[ds.valid_indices[i]:ds.valid_indices[i] + ds.encoder_steps]
        for i in range(n)
    ])
    y_dd = np.stack([
        ds.forward_drawdowns[ds.valid_indices[i] + ds.encoder_steps:ds.valid_indices[i] + ds.encoder_steps + ds.decoder_steps]
        for i in range(n)
    ])
    X_static = None
    if ds._num_static > 0:
        X_static = np.stack([
            ds.static_numeric[ds.valid_indices[i]:ds.valid_indices[i] + ds.encoder_steps].mean(axis=0)
            for i in range(n)
        ])
    return X_hist, y_dd, X_static


def train_baseline(model_name, ds_train, ds_test, seed):
    """Train logistic or xgboost baseline, return crash scores on test set."""
    set_seed(seed)

    # Collect data directly from numpy arrays (much faster than DataLoader)
    X_hist_train, y_train, X_static_train = _collect_numpy_windows(ds_train)
    X_hist_test, _, X_static_test = _collect_numpy_windows(ds_test)

    # Flatten: (N, T, F) -> (N, T*F)
    N_train = X_hist_train.shape[0]
    N_test = X_hist_test.shape[0]
    X_flat_train = X_hist_train.reshape(N_train, -1)
    X_flat_test = X_hist_test.reshape(N_test, -1)
    if X_static_train is not None:
        X_flat_train = np.concatenate([X_flat_train, X_static_train], axis=1)
        X_flat_test = np.concatenate([X_flat_test, X_static_test], axis=1)

    # Handle NaN/Inf
    X_flat_train = np.nan_to_num(X_flat_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_flat_test = np.nan_to_num(X_flat_test, nan=0.0, posinf=1.0, neginf=-1.0)

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        # Get crash labels
        if "crash_label" in ds_train.df.columns:
            crash_col = ds_train.df["crash_label"].values.astype(np.float32)
            y_binary = np.stack([
                crash_col[ds_train.valid_indices[i] + ENCODER_STEPS:ds_train.valid_indices[i] + ENCODER_STEPS + DECODER_STEPS]
                for i in range(len(ds_train))
            ])
        else:
            y_binary = (np.abs(y_train) > 0.10).astype(np.float32)

        # Single logistic model: predict max crash prob across decoder window
        y_any_crash = y_binary.max(axis=1)

        lr_model = LogisticRegression(
            max_iter=300, solver="saga", C=0.1, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        lr_model.fit(X_flat_train, y_any_crash)
        crash_scores = lr_model.predict_proba(X_flat_test)[:, 1]

    else:  # xgboost
        from xgboost import XGBRegressor

        # Predict max absolute drawdown across decoder window
        y_max_dd = np.abs(y_train).max(axis=1)

        xgb_model = XGBRegressor(
            n_estimators=100, max_depth=5, random_state=seed,
            n_jobs=-1, tree_method="hist",
        )
        xgb_model.fit(X_flat_train, y_max_dd)
        raw_preds = xgb_model.predict(X_flat_test)
        crash_scores = np.clip(np.abs(raw_preds) / 0.3, 0, 1)

    return crash_scores


def train_pytorch_model(model_name, ds_train, ds_val, ds_test, seed, device):
    """Train a PyTorch model and return crash scores and feature importance on test set."""
    set_seed(seed)

    model = build_pytorch_model(model_name, ds_train)
    loss_fn = CrisisAwareLoss()

    trainer_config = {
        "lr": LR, "batch_size": BATCH_SIZE, "grad_clip": 1.0,
        "task": "B", "device": device,
        "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints", f"{model_name}_seed{seed}"),
    }
    trainer = Trainer(
        model=model, config=trainer_config,
        train_dataset=ds_train, val_dataset=ds_val,
        loss_fn=loss_fn,
    )
    trainer.train(epochs=EPOCHS)

    model.eval()
    model.to(device)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ds_test.collate_fn)

    all_preds = []
    all_hist_weights = []
    with torch.no_grad():
        for batch_dict, _ in test_loader:
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            outputs = model(batch_dict)
            preds = outputs["predicted"]
            all_preds.append(preds.cpu())
            if "historical_weights" in outputs:
                all_hist_weights.append(outputs["historical_weights"].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    median_pred = all_preds[:, :, 3]
    crash_scores = torch.abs(median_pred).max(dim=1).values.numpy()
    crash_scores = np.clip(crash_scores / 0.3, 0, 1)

    # Feature importance: mean VSN weights across all test samples and timesteps
    feature_importance = None
    if all_hist_weights:
        weights = torch.cat(all_hist_weights, dim=0)
        feature_importance = weights.mean(dim=(0, 1)).numpy()

    return crash_scores, feature_importance


# ==============================================================================
# Evaluation metrics
# ==============================================================================

def compute_ece(y_true, y_scores, n_bins=10):
    """Compute Expected Calibration Error."""
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=n_bins, strategy="uniform"
        )
        bin_counts = np.histogram(y_scores, bins=n_bins, range=(0, 1))[0]
        total = len(y_true)
        ece = 0.0
        for i in range(len(fraction_of_positives)):
            if i < len(bin_counts) and bin_counts[i] > 0:
                ece += (bin_counts[i] / total) * abs(fraction_of_positives[i] - mean_predicted_value[i])
        return float(ece)
    except Exception:
        return float("nan")


def compute_early_warning_lead_time(y_true, y_scores, threshold, min_sustained=3):
    """Compute average early warning lead time (in weeks).

    For each crash event in the test set (consecutive crash_label=1 periods),
    find the earliest sustained alarm (min_sustained+ consecutive weeks above threshold)
    before the event starts. Report the lead time in weeks.
    """
    n = len(y_true)
    alarms = (y_scores >= threshold).astype(int)

    # Find crash event start indices
    crash_events = []
    i = 0
    while i < n:
        if y_true[i] == 1:
            start = i
            while i < n and y_true[i] == 1:
                i += 1
            crash_events.append((start, i))
        else:
            i += 1

    if len(crash_events) == 0:
        return float("nan")

    lead_times = []
    for event_start, event_end in crash_events:
        # Look backwards from event_start for sustained alarm
        best_lead = 0
        # Check if there's a sustained alarm in the window before the event
        search_start = max(0, event_start - 252)  # Look up to 252 trading days back
        consecutive = 0
        first_alarm_week = None

        for j in range(search_start, event_start):
            if alarms[j]:
                consecutive += 1
                if consecutive >= min_sustained and first_alarm_week is None:
                    first_alarm_week = j - (min_sustained - 1)
            else:
                consecutive = 0

        if first_alarm_week is not None:
            best_lead = event_start - first_alarm_week
        lead_times.append(best_lead)

    return float(np.mean(lead_times)) if lead_times else float("nan")


def _compute_core_metrics(y_true, crash_scores):
    """Compute PR-AUC, ROC-AUC, F1, Precision, Recall, ECE for a label/score pair."""
    results = {}
    unique = np.unique(y_true)
    has_both_classes = len(unique) > 1

    if has_both_classes:
        try:
            results["pr_auc"] = float(average_precision_score(y_true, crash_scores))
        except Exception:
            results["pr_auc"] = float("nan")
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, crash_scores))
        except Exception:
            results["roc_auc"] = float("nan")
    else:
        results["pr_auc"] = float("nan")
        results["roc_auc"] = float("nan")

    best_f1 = 0
    best_threshold = 0.5
    for thr in np.arange(0.05, 0.96, 0.05):
        y_pred = (crash_scores >= thr).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr
        except Exception:
            continue

    y_pred_opt = (crash_scores >= best_threshold).astype(int)
    results["f1"] = float(f1_score(y_true, y_pred_opt, zero_division=0))
    results["precision"] = float(precision_score(y_true, y_pred_opt, zero_division=0))
    results["recall"] = float(recall_score(y_true, y_pred_opt, zero_division=0))
    results["optimal_threshold"] = float(best_threshold)
    results["ece"] = compute_ece(y_true, crash_scores)
    return results


def evaluate_crash_predictions(y_true, crash_scores):
    """Compute all metrics: standard, strided (de-duplicated), and onset-only."""
    # === 1. Standard metrics (all windows, stride=1) ===
    results = _compute_core_metrics(y_true, crash_scores)
    results["early_warning_days"] = compute_early_warning_lead_time(
        y_true, crash_scores, results["optimal_threshold"]
    )

    # === 2. Strided metrics (every 21 days = ~1 month, removes persistence inflation) ===
    stride = 21
    strided_idx = np.arange(0, len(y_true), stride)
    y_strided = y_true[strided_idx]
    scores_strided = crash_scores[strided_idx]
    strided = _compute_core_metrics(y_strided, scores_strided)
    for k, v in strided.items():
        results[f"strided_{k}"] = v

    # === 3. Onset-only metrics (only transition points: 0->1 and their calm neighbors) ===
    # Find onset indices: crash_label transitions from 0 to 1
    onset_indices = []
    for i in range(1, len(y_true)):
        if y_true[i] == 1 and y_true[i - 1] == 0:
            onset_indices.append(i)

    if len(onset_indices) >= 2:
        # Build evaluation set: onset windows + equal number of calm windows (spaced out)
        calm_indices = np.where(y_true == 0)[0]
        # Sample calm indices evenly spaced to avoid cluster bias
        calm_step = max(1, len(calm_indices) // (len(onset_indices) * 3))
        calm_sample = calm_indices[::calm_step][:len(onset_indices) * 3]

        eval_idx = np.concatenate([onset_indices, calm_sample])
        eval_idx = np.unique(eval_idx)
        eval_idx = eval_idx[eval_idx < len(y_true)]

        y_onset = y_true[eval_idx]
        scores_onset = crash_scores[eval_idx]
        onset = _compute_core_metrics(y_onset, scores_onset)
        for k, v in onset.items():
            results[f"onset_{k}"] = v
        results["onset_n_events"] = len(onset_indices)
        results["onset_n_eval"] = len(eval_idx)
    else:
        for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "ece"]:
            results[f"onset_{k}"] = float("nan")
        results["onset_n_events"] = len(onset_indices)

    return results


# ==============================================================================
# Main experiment
# ==============================================================================

def run_experiment():
    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Epochs (PyTorch): {EPOCHS}")
    print(f"Data dir: {DATA_DIR}")
    print("=" * 80)

    all_results = defaultdict(list)  # model_name -> list of metric dicts (per seed)

    for model_name in MODEL_NAMES:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            t0 = time.time()

            # Load fresh datasets
            ds_train, ds_val, ds_test = load_datasets(seed)

            # Get test labels
            y_true = get_test_crash_labels_from_df(ds_test)
            print(f"  Test set: {len(y_true)} windows, {int(y_true.sum())} crash events ({y_true.mean()*100:.1f}%)")

            try:
                feature_importance = None
                if model_name in ("logistic", "xgboost"):
                    crash_scores = train_baseline(model_name, ds_train, ds_test, seed)
                else:
                    crash_scores, feature_importance = train_pytorch_model(
                        model_name, ds_train, ds_val, ds_test, seed, device,
                    )

                # Ensure lengths match
                min_len = min(len(y_true), len(crash_scores))
                y_true = y_true[:min_len]
                crash_scores = crash_scores[:min_len]

                metrics = evaluate_crash_predictions(y_true, crash_scores)
                elapsed = time.time() - t0
                metrics["time_seconds"] = round(elapsed, 1)
                metrics["seed"] = seed

                if feature_importance is not None:
                    feat_names = ds_train.historical_numeric_cols
                    if len(feat_names) < len(feature_importance):
                        # Include extra auto-detected features
                        feat_names = feat_names + [f"extra_{i}" for i in range(len(feature_importance) - len(feat_names))]
                    metrics["feature_importance"] = {
                        name: round(float(w), 6)
                        for name, w in zip(feat_names[:len(feature_importance)], feature_importance)
                    }

                all_results[model_name].append(metrics)
                print(f"  PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | ECE: {metrics['ece']:.4f} | "
                      f"EW Lead: {metrics['early_warning_days']:.1f}d | "
                      f"Time: {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[model_name].append({
                    "seed": seed,
                    "error": str(e),
                    "time_seconds": round(elapsed, 1),
                })

    # ===========================================================================
    # Print comparison table
    # ===========================================================================
    # ===========================================================================
    # Print comparison tables for all 3 evaluation modes
    # ===========================================================================
    summary = {}

    eval_modes = [
        ("STANDARD (all windows, stride=1)", ["pr_auc", "roc_auc", "f1", "precision", "recall", "ece", "early_warning_days"]),
        ("STRIDED (every 21 days, no persistence inflation)", ["strided_pr_auc", "strided_roc_auc", "strided_f1", "strided_precision", "strided_recall", "strided_ece"]),
        ("ONSET-ONLY (transition points 0→1 + calm samples)", ["onset_pr_auc", "onset_roc_auc", "onset_f1", "onset_precision", "onset_recall"]),
    ]

    for mode_name, metric_keys in eval_modes:
        print(f"\n\n{'=' * 120}")
        print(f"COMPARISON: {mode_name}")
        print(f"{'=' * 120}")
        # Clean column names for display
        display_keys = [k.replace("strided_", "").replace("onset_", "") for k in metric_keys]
        header = f"{'Model':<20}" + "".join(f"{k:>16}" for k in display_keys)
        print(header)
        print("-" * 120)

        for model_name in MODEL_NAMES:
            runs = all_results[model_name]
            valid_runs = [r for r in runs if "error" not in r]
            if not valid_runs:
                print(f"{model_name:<20}" + "  (all runs failed)")
                continue

            row = f"{model_name:<20}"
            if model_name not in summary:
                summary[model_name] = {}
            for k in metric_keys:
                vals = [r[k] for r in valid_runs if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))]
                if vals:
                    mean_v = np.mean(vals)
                    std_v = np.std(vals)
                    row += f"{mean_v:>10.4f}+-{std_v:.3f}"
                    summary[model_name][k] = {"mean": round(float(mean_v), 4), "std": round(float(std_v), 4)}
                else:
                    row += f"{'N/A':>16}"
                    summary[model_name][k] = {"mean": None, "std": None}
            print(row)
            summary[model_name]["n_valid_seeds"] = len(valid_runs)

    print("=" * 120)

    # ===========================================================================
    # Feature importance summary (TFT and RA-TFT)
    # ===========================================================================
    feat_importance_summary = {}
    for model_name in ["tft", "ra-tft"]:
        runs = all_results.get(model_name, [])
        fi_runs = [r["feature_importance"] for r in runs if "feature_importance" in r]
        if fi_runs:
            # Average across seeds
            all_feats = list(fi_runs[0].keys())
            avg_importance = {}
            for feat in all_feats:
                vals = [r[feat] for r in fi_runs if feat in r]
                avg_importance[feat] = round(float(np.mean(vals)), 6)
            # Sort by importance descending
            sorted_fi = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
            feat_importance_summary[model_name] = sorted_fi

            print(f"\n{'='*80}")
            print(f"FEATURE IMPORTANCE: {model_name} (mean VSN weight across seeds)")
            print(f"{'='*80}")
            for i, (feat, weight) in enumerate(sorted_fi.items()):
                bar = "█" * int(weight * 200)
                print(f"  {i+1:2d}. {feat:<25s} {weight:.4f}  {bar}")

    # ===========================================================================
    # Save results
    # ===========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "config": {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "encoder_steps": ENCODER_STEPS,
            "decoder_steps": DECODER_STEPS,
            "device": device,
        },
        "summary": summary,
        "feature_importance": feat_importance_summary,
        "per_seed": {model: runs for model, runs in all_results.items()},
    }

    output_path = os.path.join(OUTPUT_DIR, "comparison_results_v2.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
