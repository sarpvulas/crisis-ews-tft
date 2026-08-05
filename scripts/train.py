#!/usr/bin/env python
"""CLI for training market crash prediction models.

Usage:
    python scripts/train.py --model ra-tft --seed 42
    python scripts/train.py --model lstm --epochs 50
"""

import argparse
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.baselines.lstm_baseline import LSTMBaseline
from models.crisis_loss import CrisisAwareLoss
from training.dataset import CrisisDataset
from training.trainer import Trainer


# Default feature configuration
DEFAULT_CONFIG = dict(
    num_historical_numeric=10,
    num_static_numeric=0,
    num_future_numeric=2,
    encoder_steps=252,
    decoder_steps=63,
    task_type="ews",
    num_outputs=1,
)


def build_model(model_name: str, config_overrides: dict = None):
    """Build model from name."""
    task_config = DEFAULT_CONFIG.copy()
    if config_overrides:
        task_config.update(config_overrides)

    if model_name in ("tft", "ra-tft"):
        is_regime = model_name == "ra-tft"
        cfg = TFTConfig(
            **task_config,
            state_size=task_config.get("state_size", 64),
            hidden_size=task_config.get("hidden_size", 64),
            attention_heads=task_config.get("attention_heads", 4),
            lstm_layers=task_config.get("lstm_layers", 2),
            dropout=task_config.get("dropout", 0.1),
            num_regime_states=3,
            use_regime_module=is_regime,
            use_regime_attention=is_regime,
        )
        if model_name == "ra-tft":
            return RegimeAwareTFT(cfg)
        return TemporalFusionTransformer(cfg)

    elif model_name == "lstm":
        return LSTMBaseline(
            input_dim=task_config["num_historical_numeric"],
            hidden_dim=64,
            num_layers=2,
            task="B",
            encoder_steps=task_config["encoder_steps"],
            decoder_steps=task_config["decoder_steps"],
            num_quantiles=task_config.get("num_outputs", 1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train market crash prediction models")
    parser.add_argument("--model", type=str, default="ra-tft",
                        choices=["tft", "ra-tft", "lstm", "xgboost", "logistic"],
                        help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (None for synthetic)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="crisis-prediction")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = DEFAULT_CONFIG["encoder_steps"]
    dec = DEFAULT_CONFIG["decoder_steps"]

    print(f"Training {args.model} (seed={args.seed}, device={args.device})")

    # Handle non-torch baselines
    if args.model in ("xgboost", "logistic"):
        from models.baselines.xgboost_baseline import XGBoostBaseline
        from models.baselines.logistic_baseline import LogisticBaseline

        ds_train = CrisisDataset(
            split="train", data_dir=args.data_dir,
            encoder_steps=enc, decoder_steps=dec,
            synthetic=(args.data_dir is None),
        )
        ds_val = CrisisDataset(
            split="val", data_dir=args.data_dir,
            encoder_steps=enc, decoder_steps=dec,
            synthetic=(args.data_dir is None),
            feature_scaler=ds_train.feature_scaler,
        )

        from torch.utils.data import DataLoader
        loader = DataLoader(ds_train, batch_size=len(ds_train), collate_fn=ds_train.collate_fn)
        batch, targets = next(iter(loader))
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()

        y = targets["ews_label"].numpy()

        if args.model == "xgboost":
            model = XGBoostBaseline(task="B", encoder_steps=enc, decoder_steps=dec)
        else:
            model = LogisticBaseline(task="B", encoder_steps=enc, decoder_steps=dec)

        model.fit(X_hist, y, X_static=X_static)
        print(f"Baseline {args.model} trained successfully.")
        return

    # PyTorch models — create dataset first to detect feature counts
    ds_train = CrisisDataset(
        split="train", data_dir=args.data_dir,
        encoder_steps=enc, decoder_steps=dec,
        synthetic=(args.data_dir is None),
    )
    ds_val = CrisisDataset(
        split="val", data_dir=args.data_dir,
        encoder_steps=enc, decoder_steps=dec,
        synthetic=(args.data_dir is None),
        feature_scaler=ds_train.feature_scaler,
    )

    # Auto-detect feature counts from loaded data
    config_overrides = {
        "num_historical_numeric": ds_train._num_historical,
        "num_static_numeric": ds_train._num_static,
        "num_future_numeric": ds_train._num_future,
    }
    print(f"  Features: {ds_train._num_historical} historical, {ds_train._num_static} static, {ds_train._num_future} future")

    model = build_model(args.model, config_overrides=config_overrides)

    loss_fn = CrisisAwareLoss()

    trainer_config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": 1.0,
        "task": "B",
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
    }

    wandb_config = {"project": args.wandb_project} if args.wandb else None

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataset=ds_train,
        val_dataset=ds_val,
        loss_fn=loss_fn,
        use_wandb=args.wandb,
        wandb_config=wandb_config,
    )

    history = trainer.train(epochs=args.epochs)
    print(f"Training complete. Final train loss: {history['train_loss'][-1]:.4f}, "
          f"val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
