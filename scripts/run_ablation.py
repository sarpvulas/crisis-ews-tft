#!/usr/bin/env python
"""CLI for running ablation studies.

Iterates all ablation configurations from the design spec across multiple seeds.

Usage:
    python scripts/run_ablation.py --seeds 5 --epochs 50
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.configs import TFTConfig
from models.ra_tft import RegimeAwareTFT
from models.tft import TemporalFusionTransformer
from models.crisis_loss import CrisisAwareLoss
from training.dataset import CrisisDataset
from training.trainer import Trainer


# Ablation configurations from the design spec
ABLATION_CONFIGS = {
    "full_ra_tft": {
        "use_regime_module": True,
        "use_regime_attention": True,
        "gamma": 0.5,
        "delta": 0.3,
    },
    "no_regime_module": {
        "use_regime_module": False,
        "use_regime_attention": False,
        "gamma": 0.0,
        "delta": 0.3,
    },
    "no_regime_attention": {
        "use_regime_module": True,
        "use_regime_attention": False,
        "gamma": 0.5,
        "delta": 0.3,
    },
    "no_early_warning_loss": {
        "use_regime_module": True,
        "use_regime_attention": True,
        "gamma": 0.5,
        "delta": 0.0,
    },
    "no_regime_supervision": {
        "use_regime_module": True,
        "use_regime_attention": True,
        "gamma": 0.0,
        "delta": 0.3,
    },
    "vanilla_tft": {
        "use_regime_module": False,
        "use_regime_attention": False,
        "gamma": 0.0,
        "delta": 0.0,
    },
}

DEFAULT_CONFIG = dict(
    num_historical_numeric=10, num_static_numeric=0, num_future_numeric=2,
    encoder_steps=252, decoder_steps=63, task_type="regression", num_outputs=7,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
)


def run_single(
    ablation_name: str,
    ablation_cfg: dict,
    seed: int,
    epochs: int,
    data_dir: str,
    device: str,
    output_dir: str,
    lr: float,
    batch_size: int,
) -> dict:
    """Run a single ablation experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_cfg = DEFAULT_CONFIG.copy()
    enc = task_cfg.pop("encoder_steps")
    dec = task_cfg.pop("decoder_steps")

    model_config = TFTConfig(
        **task_cfg,
        encoder_steps=enc,
        decoder_steps=dec,
        state_size=64,
        hidden_size=64,
        attention_heads=4,
        lstm_layers=2,
        dropout=0.1,
        num_regime_states=3,
        use_regime_module=ablation_cfg["use_regime_module"],
        use_regime_attention=ablation_cfg["use_regime_attention"],
    )

    if ablation_cfg["use_regime_module"]:
        model = RegimeAwareTFT(model_config)
    else:
        model = TemporalFusionTransformer(model_config)

    loss_fn = CrisisAwareLoss(
        gamma=ablation_cfg["gamma"],
    )

    ds_train = CrisisDataset(
        split="train", data_dir=data_dir,
        encoder_steps=enc, decoder_steps=dec,
        synthetic=(data_dir is None),
    )
    ds_val = CrisisDataset(
        split="val", data_dir=data_dir,
        encoder_steps=enc, decoder_steps=dec,
        synthetic=(data_dir is None),
    )

    checkpoint_dir = os.path.join(output_dir, f"{ablation_name}_seed{seed}")
    trainer = Trainer(
        model=model,
        config={
            "lr": lr, "batch_size": batch_size, "grad_clip": 1.0,
            "task": "B", "device": device, "checkpoint_dir": checkpoint_dir,
        },
        train_dataset=ds_train,
        val_dataset=ds_val,
        loss_fn=loss_fn,
    )

    history = trainer.train(epochs=epochs)

    return {
        "ablation": ablation_name,
        "seed": seed,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="ablation_results")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for ablation_name, ablation_cfg in ABLATION_CONFIGS.items():
        for seed in range(args.seeds):
            print(f"\n{'='*60}")
            print(f"Ablation: {ablation_name} | Seed: {seed}")
            print(f"{'='*60}")

            result = run_single(
                ablation_name=ablation_name,
                ablation_cfg=ablation_cfg,
                seed=seed,
                epochs=args.epochs,
                data_dir=args.data_dir,
                device=args.device,
                output_dir=args.output_dir,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            all_results.append(result)
            print(f"  Final val loss: {result['final_val_loss']:.4f}")

    # Save summary
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary (mean +/- std of val loss across seeds):")
    print(f"{'='*60}")
    for ablation_name in ABLATION_CONFIGS:
        runs = [r for r in all_results if r["ablation"] == ablation_name]
        losses = [r["best_val_loss"] for r in runs]
        print(f"  {ablation_name:30s}: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")


if __name__ == "__main__":
    main()
