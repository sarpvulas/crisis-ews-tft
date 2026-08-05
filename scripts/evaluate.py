#!/usr/bin/env python
"""CLI for evaluating trained market crash prediction models.

Usage:
    python scripts/evaluate.py --model-path checkpoints/best_model.pt --split test
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.configs import TFTConfig
from models.ra_tft import RegimeAwareTFT
from models.tft import TemporalFusionTransformer
from training.dataset import CrisisDataset
from training.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate market crash prediction models")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (None for synthetic)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")

    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model_cfg = checkpoint.get("model_config", checkpoint.get("config", {}))

    enc = model_cfg.get("encoder_steps", 252)
    dec = model_cfg.get("decoder_steps", 63)

    # Load dataset
    dataset = CrisisDataset(
        split=args.split, data_dir=args.data_dir,
        encoder_steps=enc, decoder_steps=dec,
        synthetic=(args.data_dir is None),
        exclude_crisis_windows=False,  # Evaluate on all windows including crisis
    )

    # Use saved model config, but override feature dims from dataset to be safe
    feat_dims = dataset.get_feature_dims()
    cfg_dict = {k: v for k, v in model_cfg.items() if hasattr(TFTConfig, k)}
    cfg_dict.update(feat_dims)
    model_config = TFTConfig(**cfg_dict)

    # Reconstruct model (try RA-TFT first, fall back to vanilla TFT)
    try:
        model = RegimeAwareTFT(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception:
        model = TemporalFusionTransformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(
        model, dataset, task="B",
        device=args.device, batch_size=args.batch_size,
    )

    print("\nEvaluation Results:")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
