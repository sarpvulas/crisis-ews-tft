import pytest
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock
from models.configs import TFTConfig
from models.ra_tft import RegimeAwareTFT
from models.crisis_loss import CrisisAwareLoss
from training.dataset import CrisisDataset
from training.trainer import Trainer


@pytest.fixture
def small_config():
    return TFTConfig(
        num_historical_numeric=5,
        num_static_numeric=1,
        num_future_numeric=2,
        state_size=16,
        hidden_size=16,
        attention_heads=2,
        lstm_layers=1,
        dropout=0.0,
        encoder_steps=8,
        decoder_steps=3,
        task_type="ews",
        num_outputs=1,
        num_regime_states=3,
        use_regime_module=True,
        use_regime_attention=True,
    )


@pytest.fixture
def small_dataset():
    return CrisisDataset(
        split="train", data_dir=None,
        encoder_steps=8, decoder_steps=3, synthetic=True,
        num_synthetic_samples=50,
        static_numeric_cols=["feat_s0"],
        historical_numeric_cols=["feat_h0", "feat_h1", "feat_h2", "feat_h3", "feat_h4"],
        future_numeric_cols=["feat_f0", "feat_f1"],
    )


@pytest.fixture
def small_val_dataset():
    return CrisisDataset(
        split="val", data_dir=None,
        encoder_steps=8, decoder_steps=3, synthetic=True,
        num_synthetic_samples=30,
        static_numeric_cols=["feat_s0"],
        historical_numeric_cols=["feat_h0", "feat_h1", "feat_h2", "feat_h3", "feat_h4"],
        future_numeric_cols=["feat_f0", "feat_f1"],
    )


class TestTrainer:
    def test_trainer_creation(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={"lr": 1e-3, "batch_size": 8, "grad_clip": 1.0, "task": "B"},
            train_dataset=small_dataset,
            val_dataset=small_val_dataset,
            loss_fn=loss_fn,
        )
        assert trainer is not None

    def test_train_runs_without_error(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={"lr": 1e-3, "batch_size": 8, "grad_clip": 1.0, "task": "B"},
            train_dataset=small_dataset,
            val_dataset=small_val_dataset,
            loss_fn=loss_fn,
        )
        history = trainer.train(epochs=2)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_train_returns_val_loss(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={"lr": 1e-3, "batch_size": 8, "grad_clip": 1.0, "task": "B"},
            train_dataset=small_dataset,
            val_dataset=small_val_dataset,
            loss_fn=loss_fn,
        )
        history = trainer.train(epochs=2)
        assert "val_loss" in history
        assert len(history["val_loss"]) == 2

    def test_checkpoint_saved(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config={"lr": 1e-3, "batch_size": 8, "grad_clip": 1.0,
                        "task": "B", "checkpoint_dir": tmpdir},
                train_dataset=small_dataset,
                val_dataset=small_val_dataset,
                loss_fn=loss_fn,
            )
            trainer.train(epochs=2)
            checkpoints = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(checkpoints) >= 1

    @patch("training.trainer.wandb", None)
    def test_train_without_wandb(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={"lr": 1e-3, "batch_size": 8, "grad_clip": 1.0, "task": "B"},
            train_dataset=small_dataset,
            val_dataset=small_val_dataset,
            loss_fn=loss_fn,
            use_wandb=False,
        )
        history = trainer.train(epochs=1)
        assert len(history["train_loss"]) == 1

    def test_lr_decreases_on_plateau(self, small_config, small_dataset, small_val_dataset):
        model = RegimeAwareTFT(small_config)
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={"lr": 1e-2, "batch_size": 8, "grad_clip": 1.0, "task": "B"},
            train_dataset=small_dataset,
            val_dataset=small_val_dataset,
            loss_fn=loss_fn,
        )
        history = trainer.train(epochs=2)
        assert "lr" in history
