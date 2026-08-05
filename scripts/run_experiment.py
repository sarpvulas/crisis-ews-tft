#!/usr/bin/env python
"""Unified experiment driver for all crisis-prediction architectures.

This script is the single entry point used by:
  - local smoke tests:           python scripts/run_experiment.py --model tft --task B
  - sbatch single jobs:          scripts/job_train.sh --model ra-tft --task B --seed 7
  - sbatch array jobs (seeds):   sbatch --array=0-9 scripts/job_array.sh --model ra-tft --task B
  - wandb sweep agents:          sbatch --array=0-29 scripts/job_array.sh --sweep <sweep_id>

It does five things, in order:
  1. Build train / val / test datasets from a named split (training/splits.py).
  2. Build the model from the registry (models/registry.py).
  3. Train via the appropriate adapter (Torch trainer or sklearn fit).
  4. Evaluate on val and test, with bootstrap CIs.
  5. Write a single JSON results file to --results-dir.

Outputs are designed so downstream analysis can run on the JSON files alone —
no need to reload model checkpoints unless you want to inspect attention or
recompute calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

# Ensure project root is on path when invoked via sbatch.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing registry triggers all @register_model decorators.
import models.registry  # noqa: F401
import models.tdt_adapter  # noqa: F401 — registers 'tdt'
import models.tdt_features  # noqa: F401 — registers 'tdt-xgb' hybrid
from models.base import build_model, list_models
from training.dataset import CrisisDataset
from training.splits import (
    get_primary_split,
    get_walkforward_fold,
)


# ---------- Helpers ----------

def _device_for(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int) -> None:
    """Seed every RNG and force deterministic kernels for reproducibility.

    Previously only numpy + torch + cuda were seeded, which left python's
    `random`, the DataLoader shuffle order, and cuDNN/cuBLAS autotuning
    nondeterministic — the reason the same seed gave different results on
    cpu vs cuda and why ra-tft swung wildly across seeds.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS workspace must be fixed for use_deterministic_algorithms on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: some LSTM/attention CUDA kernels lack deterministic
    # implementations; warn instead of hard-failing so the run still completes.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _get_probs_labels(model, dataset) -> tuple:
    """Return aligned (probs, y_true) for a dataset.

    Split out from _evaluate so we can refit calibrators on val probs and
    apply them to test probs without re-running inference.
    """
    from torch.utils.data import DataLoader

    probs = model.predict_proba(dataset).astype(np.float64).reshape(-1)
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)
    _, targets = next(iter(loader))
    y_true = targets["ews_label"].numpy().reshape(-1).astype(np.float64)
    if probs.shape != y_true.shape:
        if y_true.size % probs.size == 0:
            probs = np.repeat(probs, y_true.size // probs.size)
        else:
            raise ValueError(f"prob shape {probs.shape} != label shape {y_true.shape}")
    return probs, y_true


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    from training.evaluation import (
        compute_pr_auc, compute_roc_auc,
        compute_calibration_error, compute_brier_score, compute_bootstrap_ci,
    )
    results: Dict[str, float] = {
        "n_samples": int(y_true.size),
        "positive_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) >= 2:
        results["pr_auc"] = compute_pr_auc(y_true, probs)
        results["roc_auc"] = compute_roc_auc(y_true, probs)
        results["calibration_error"] = compute_calibration_error(y_true, probs)
        results["brier_score"] = compute_brier_score(y_true, probs)
        pr_lo, pr_hi = compute_bootstrap_ci(compute_pr_auc, y_true, probs)
        roc_lo, roc_hi = compute_bootstrap_ci(compute_roc_auc, y_true, probs)
        results["pr_auc_ci"] = [pr_lo, pr_hi]
        results["roc_auc_ci"] = [roc_lo, roc_hi]
    return results


def _evaluate(model, dataset) -> Dict[str, float]:
    probs, y_true = _get_probs_labels(model, dataset)
    return _compute_metrics(y_true, probs)


def _build_datasets(args) -> tuple:
    """Build train/val/test CrisisDatasets per the requested split.

    For topology-augmented models (`topo-*`), the base CrisisDataset is wrapped
    with TopologyAugmentedDataset, which precomputes persistence features and
    appends them to the historical channels. The wrapper is transparent to the
    rest of the harness — `get_feature_dims()` already advertises the inflated
    channel count.
    """
    # --- Multi-market panel branch ---
    if getattr(args, "panel_path", None):
        from training.dataset import PanelCrisisDataset
        common = dict(panel_path=args.panel_path, encoder_steps=args.encoder_steps,
                      decoder_steps=args.decoder_steps,
                      train_end=args.panel_train_end, val_end=args.panel_val_end,
                      test_end=getattr(args, "panel_test_end", None),
                      per_market_norm=bool(args.panel_per_market_norm))
        ds_train = PanelCrisisDataset(split="train", **common)
        ds_val = PanelCrisisDataset(split="val", feature_scaler=ds_train.feature_scaler, **common)
        ds_test = PanelCrisisDataset(split="test", feature_scaler=ds_train.feature_scaler, **common)
        return ds_train, ds_val, ds_test, get_primary_split(args.task)

    if args.walkforward:
        split = get_walkforward_fold(args.walkforward)
    else:
        split = get_primary_split(args.task)

    ds_train = CrisisDataset(
        split="train", data_dir=args.data_dir,
        encoder_steps=args.encoder_steps, decoder_steps=args.decoder_steps,
        synthetic=(args.data_dir is None),
    )
    ds_val = CrisisDataset(
        split="val", data_dir=args.data_dir,
        encoder_steps=args.encoder_steps, decoder_steps=args.decoder_steps,
        synthetic=(args.data_dir is None),
        feature_scaler=ds_train.feature_scaler,
    )
    ds_test = CrisisDataset(
        split="test", data_dir=args.data_dir,
        encoder_steps=args.encoder_steps, decoder_steps=args.decoder_steps,
        synthetic=(args.data_dir is None),
        feature_scaler=ds_train.feature_scaler,
    )

    if args.model.startswith("topo-"):
        from features.topology import TopologyConfig
        from training.topology_dataset import TopologyAugmentedDataset

        topo_cfg = TopologyConfig(
            window=min(60, args.encoder_steps // 2),
            homology_dims=tuple(args.topo_homology_dims),
            embed_resolution=args.topo_embed_resolution,
            embed_sigma=args.topo_embed_sigma,
        )
        cache_dir = args.topo_cache_dir
        ds_train = TopologyAugmentedDataset(ds_train, topo_cfg, cache_dir=cache_dir)
        ds_val = TopologyAugmentedDataset(ds_val, topo_cfg, cache_dir=cache_dir)
        ds_test = TopologyAugmentedDataset(ds_test, topo_cfg, cache_dir=cache_dir)

    return ds_train, ds_val, ds_test, split


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Unified crisis-prediction experiment driver")
    parser.add_argument("--model", required=True, choices=list_models(),
                        help="Architecture name (must be registered).")
    parser.add_argument("--task", default="B", choices=["A", "B"],
                        help="A = systemic macro crisis, B = market crash.")
    parser.add_argument("--walkforward", default=None,
                        help="Walk-forward fold name. Overrides --task split if set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-steps", type=int, default=252)
    parser.add_argument("--decoder-steps", type=int, default=63)
    parser.add_argument("--data-dir", default=None,
                        help="If unset, uses synthetic data (smoke-test only).")
    parser.add_argument("--panel-path", default=None,
                        help="Path to a multi-market panel.parquet (PanelCrisisDataset). "
                             "Overrides --data-dir; uses market-identity static categorical embedding.")
    parser.add_argument("--panel-train-end", default="2015-12-31",
                        help="Panel temporal split: last train date.")
    parser.add_argument("--panel-val-end", default="2018-12-31",
                        help="Panel temporal split: last val date (test = after this).")
    parser.add_argument("--panel-test-end", default=None,
                        help="Panel: bound the test window end date (for walk-forward folds).")
    parser.add_argument("--panel-per-market-norm", type=int, default=0, choices=[0, 1],
                        help="Panel: fit a separate feature scaler per market (RevIN-style) vs one global.")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Where to save best-model checkpoints.")
    parser.add_argument("--results-dir", default="output/results",
                        help="Where to write the result JSON.")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the most recent checkpoint in --checkpoint-dir.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb (offline-by-default via WANDB_MODE).")
    parser.add_argument("--wandb-project", default="crisis-prediction")
    # HP overrides passed straight into the model factory's config dict.
    parser.add_argument("--n-estimators", type=int, default=None, help="Tree baselines: number of trees/iterations.")
    parser.add_argument("--max-depth", type=int, default=None, help="Tree baselines: max tree depth.")
    parser.add_argument("--xgb-lr", type=float, default=None, help="Tree baselines (xgb/histgb): learning rate.")
    parser.add_argument("--min-samples-leaf", type=int, default=None, help="RF baseline: min samples per leaf.")
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    # --- Regime factorial ablation (RA-TFT component toggles) ---
    parser.add_argument("--use-regime-module", type=int, default=None, choices=[0, 1],
                        help="RA-TFT: 1/0 to enable/disable the regime-detection module "
                             "(default: model's own default — on for ra-tft).")
    parser.add_argument("--use-regime-attention", type=int, default=None, choices=[0, 1],
                        help="RA-TFT: 1/0 to enable/disable regime-conditioned attention bias.")
    parser.add_argument("--num-regime-states", type=int, default=None,
                        help="RA-TFT: number of latent regime states (default 3).")
    # --- Loss/objective ablation ---
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"],
                        help="EWS loss term: weighted BCE (default) or focal loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focusing parameter for --loss focal (0 == weighted BCE).")
    parser.add_argument("--beta", type=float, default=None,
                        help="Weight on the EWS classification loss term. "
                             "0 removes the classification head from the objective "
                             "entirely (regression-only ablation; pair with "
                             "--use-aux 1 --predict-from aux).")
    parser.add_argument("--predict-from", default="clf", choices=["clf", "aux"],
                        help="Which head produces the crisis score. 'clf' (default) "
                             "= sigmoid of the classification head. 'aux' = the "
                             "auxiliary drawdown regression head, mapped to a "
                             "probability by a 1-D logistic fitted on the TRAIN "
                             "split (direction learned, never assumed).")
    # --- Novel TFT-on-top ablations ---
    parser.add_argument("--use-aux", type=int, default=None, choices=[0, 1],
                        help="Add the auxiliary forward-drawdown regression head (multi-task). "
                             "Weight set via the existing --lambda-aux (default 0.5).")
    parser.add_argument("--use-regime-vsn", type=int, default=None, choices=[0, 1],
                        help="Regime-conditioned variable selection (inject a learned regime context into VSN).")
    parser.add_argument("--mc-dropout", type=int, default=0,
                        help="MC-dropout inference: average N stochastic passes (0/1 = off).")
    parser.add_argument("--tag", default="default",
                        help="Free-form label written into the result JSON for grouping.")
    # Topology features (only used when --model starts with 'topo-')
    parser.add_argument("--topo-homology-dims", type=int, nargs="+", default=[0],
                        help="Homology dimensions for persistence (0, 1, ...).")
    parser.add_argument("--topo-embed-resolution", type=int, default=8,
                        help="Persistence image resolution per axis (embed dim = R^2 * len(dims)).")
    parser.add_argument("--topo-embed-sigma", type=float, default=0.1,
                        help="Gaussian smoothing for persistence images.")
    parser.add_argument("--topo-cache-dir", default=None,
                        help="Cache dir for topology embeddings. None disables caching.")
    # TDT-specific (Stage 1.J + ablation knobs)
    parser.add_argument("--lambda-clf", type=float, default=None,
                        help="Weight on the discriminative BCE term. 0 disables the class head loss.")
    parser.add_argument("--lambda-diff", type=float, default=None,
                        help="Weight on the diffusion loss term. 0 disables diffusion regularization.")
    parser.add_argument("--lambda-aux", type=float, default=None,
                        help="Weight on auxiliary regime classification loss.")
    parser.add_argument("--pos-weight-clf", type=float, default=None,
                        help="pos_weight for the BCE term (handles class imbalance).")
    parser.add_argument("--condition-mode", default=None, choices=["ews", "regime"],
                        help="What TDT conditions classifier-free guidance on.")
    parser.add_argument("--predict-via", default=None,
                        choices=["class_head", "likelihood_ratio"],
                        help="TDT inference path. class_head=discriminative, likelihood_ratio=generative.")
    parser.add_argument("--use-class-head", type=int, default=None,
                        help="1 to enable class head, 0 to disable.")
    parser.add_argument("--d-model", type=int, default=None,
                        help="TDT/iTransformer hidden dim. Overrides --state-size for those models.")
    parser.add_argument("--e-layers", type=int, default=None,
                        help="TDT/iTransformer encoder depth.")
    parser.add_argument("--balanced-sampler", type=int, default=None,
                        help="1 to use class-balanced WeightedRandomSampler for TDT, 0 for shuffled (ablation).")
    parser.add_argument("--cond-drop-prob", type=float, default=None,
                        help="TDT classifier-free guidance dropout probability (default 0.1).")
    parser.add_argument("--diffusion-steps", type=int, default=None,
                        help="TDT number of diffusion timesteps T_diff (default 1000).")
    parser.add_argument("--posthoc-calibrate", default=None,
                        choices=["platt", "isotonic"],
                        help="Fit PostHocCalibrator on val raw probs, apply to test. "
                             "Adds calibrated_test_metrics to the JSON. Raw metrics unchanged.")
    parser.add_argument("--dump-preds", action="store_true",
                        help="Also save per-window test predictions+labels to a .npz "
                             "next to the JSON (for reliability diagrams / usefulness metric).")

    args = parser.parse_args()
    device = _device_for(args.device)
    _seed_everything(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] run: model={args.model} task={args.task} "
          f"fold={args.walkforward or 'primary'} seed={args.seed} device={device}")

    # 1. Datasets
    ds_train, ds_val, ds_test, split = _build_datasets(args)
    feature_dims = ds_train.get_feature_dims()
    print(f"  features: hist={feature_dims['num_historical_numeric']} "
          f"static={feature_dims.get('num_static_numeric', 0)} "
          f"future={feature_dims['num_future_numeric']}; "
          f"train_windows={len(ds_train)} val_windows={len(ds_val)} test_windows={len(ds_test)}")

    # 2. Model
    model_cfg: Dict[str, Any] = {
        "encoder_steps": args.encoder_steps,
        "decoder_steps": args.decoder_steps,
        "state_size": args.state_size,
        "hidden_size": args.hidden_size,
        "attention_heads": args.attention_heads,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "gamma": args.gamma,
        "pos_weight": args.pos_weight,
        "loss_type": args.loss,
        "focal_gamma": args.focal_gamma,
        "lambda_aux": (args.lambda_aux if args.lambda_aux is not None else 0.5),
        "predict_from": args.predict_from,
    }
    if args.beta is not None:
        model_cfg["beta"] = args.beta
    # Regime toggles only override the model's defaults when explicitly passed,
    # so omitting them leaves ra-tft = full regime model, tft = vanilla.
    if args.use_regime_module is not None:
        model_cfg["use_regime_module"] = bool(args.use_regime_module)
    if args.use_regime_attention is not None:
        model_cfg["use_regime_attention"] = bool(args.use_regime_attention)
    if args.num_regime_states is not None:
        model_cfg["num_regime_states"] = args.num_regime_states
    if args.use_aux is not None:
        model_cfg["use_aux"] = bool(args.use_aux)
    if args.use_regime_vsn is not None:
        model_cfg["use_regime_vsn"] = bool(args.use_regime_vsn)
    # Optional TDT / Stage-1.J overrides — set only when the user explicitly
    # passes a value, so default behaviour is unaffected.
    if args.lambda_clf is not None:
        model_cfg["lambda_clf"] = args.lambda_clf
    if args.lambda_diff is not None:
        model_cfg["lambda_diff"] = args.lambda_diff
    if args.lambda_aux is not None:
        model_cfg["lambda_aux"] = args.lambda_aux
    if args.pos_weight_clf is not None:
        model_cfg["pos_weight"] = args.pos_weight_clf
    for _k, _v in [("n_estimators", args.n_estimators), ("max_depth", args.max_depth),
                   ("xgb_lr", args.xgb_lr), ("min_samples_leaf", args.min_samples_leaf)]:
        if _v is not None:
            model_cfg[_k] = _v
    if args.condition_mode is not None:
        model_cfg["condition_mode"] = args.condition_mode
    if args.predict_via is not None:
        model_cfg["predict_via"] = args.predict_via
    if args.use_class_head is not None:
        model_cfg["use_class_head"] = bool(args.use_class_head)
    if args.d_model is not None:
        model_cfg["d_model"] = args.d_model
    if args.e_layers is not None:
        model_cfg["e_layers"] = args.e_layers
    if args.balanced_sampler is not None:
        model_cfg["balanced_sampler"] = bool(args.balanced_sampler)
    if args.cond_drop_prob is not None:
        model_cfg["cond_drop_prob"] = args.cond_drop_prob
    if args.diffusion_steps is not None:
        model_cfg["diffusion_steps"] = args.diffusion_steps
    model = build_model(args.model, model_cfg, feature_dims)
    if getattr(args, "mc_dropout", 0) and hasattr(model, "mc_samples"):
        model.mc_samples = args.mc_dropout
    print(f"  built {args.model} ({model.n_params:,} params)")

    # 3. Train
    train_cfg: Dict[str, Any] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": 1.0,
        "task": args.task,
        "device": device,
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": args.epochs,
        "seed": args.seed,  # threaded into the trainer's DataLoader generator
        "use_wandb": args.wandb,
        "wandb_config": {"project": args.wandb_project} if args.wandb else None,
    }
    fit_result = model.fit(ds_train, ds_val, train_cfg)
    print(f"  fit done in {fit_result.wall_time_s:.1f}s "
          f"(best_val={fit_result.best_val_loss:.4f}, epochs={fit_result.final_epoch})")

    # 4. Evaluate
    val_probs, val_y = _get_probs_labels(model, ds_val)
    test_probs, test_y = _get_probs_labels(model, ds_test)
    val_metrics = _compute_metrics(val_y, val_probs)
    test_metrics = _compute_metrics(test_y, test_probs)
    print(f"  val:  {val_metrics}")
    print(f"  test: {test_metrics}")

    calibrated_test_metrics = None
    if args.posthoc_calibrate is not None and len(np.unique(val_y)) >= 2:
        from training.evaluation import PostHocCalibrator
        cal = PostHocCalibrator(method=args.posthoc_calibrate)
        cal.fit(val_probs, val_y)
        test_probs_cal = cal.transform(test_probs)
        calibrated_test_metrics = _compute_metrics(test_y, test_probs_cal)
        calibrated_test_metrics["method"] = args.posthoc_calibrate
        print(f"  test (calibrated, {args.posthoc_calibrate}): {calibrated_test_metrics}")

    # Save final checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, f"{args.model}_seed{args.seed}.pt")
    try:
        model.save(final_path)
    except Exception as e:
        print(f"  WARNING: save failed: {e}")
        final_path = ""

    # 5. Write results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    fold_tag = args.walkforward or "primary"
    out_path = os.path.join(
        args.results_dir,
        f"{args.model}__task{args.task}__{fold_tag}__seed{args.seed}__{args.tag}.json",
    )
    result_blob = {
        "model": args.model,
        "task": args.task,
        "split": split.name,
        "seed": args.seed,
        "device": device,
        "tag": args.tag,
        "n_params": model.n_params,
        "wall_time_s": fit_result.wall_time_s,
        "final_epoch": fit_result.final_epoch,
        "best_val_loss": fit_result.best_val_loss,
        "config": model_cfg,
        "train_config": {k: v for k, v in train_cfg.items() if k != "wandb_config"},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibrated_test_metrics": calibrated_test_metrics,
        "checkpoint_path": final_path,
        "history": fit_result.history,
    }
    with open(out_path, "w") as f:
        json.dump(result_blob, f, indent=2, default=str)
    print(f"  results -> {out_path}")

    # Optional: dump per-window predictions for reliability diagrams / usefulness.
    if args.dump_preds:
        npz_path = out_path[:-5] + "__preds.npz"
        dump = {
            "test_y": test_y.astype(np.float32),
            "test_probs": test_probs.astype(np.float32),
            "val_y": val_y.astype(np.float32),
            "val_probs": val_probs.astype(np.float32),
        }
        if calibrated_test_metrics is not None:
            dump["test_probs_cal"] = test_probs_cal.astype(np.float32)
        np.savez_compressed(npz_path, **dump)
        print(f"  preds -> {npz_path}")


if __name__ == "__main__":
    main()
