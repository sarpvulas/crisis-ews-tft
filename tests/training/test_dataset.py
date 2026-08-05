import pytest
import torch
import os
import tempfile
import pandas as pd
import numpy as np
from training.dataset import CrisisDataset


@pytest.fixture
def synthetic_data_dir():
    """Create a temporary directory with synthetic Parquet files matching the EWS schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(42)
        n_samples = 200

        task_b_data = {
            "date": pd.date_range("2000-01-01", periods=n_samples, freq="W"),
            # Historical numeric (market indicators)
            "log_return_1d": np.random.randn(n_samples),
            "log_return_5d": np.random.randn(n_samples),
            "log_return_21d": np.random.randn(n_samples),
            "realized_vol_5d": np.abs(np.random.randn(n_samples)),
            "realized_vol_21d": np.abs(np.random.randn(n_samples)),
            "realized_vol_63d": np.abs(np.random.randn(n_samples)),
            "drawdown_52w": np.random.randn(n_samples),
            "hurst": np.random.randn(n_samples),
            "vix": np.abs(np.random.randn(n_samples)) * 10 + 15,
            "hy_spread": np.random.randn(n_samples),
            # Future numeric
            "week_sin": np.sin(np.arange(n_samples) * 2 * np.pi / 52),
            "week_cos": np.cos(np.arange(n_samples) * 2 * np.pi / 52),
            # EWS Labels
            "ews_label": np.random.binomial(1, 0.1, n_samples).astype(float),
            "in_crisis": np.random.binomial(1, 0.05, n_samples).astype(bool),
            "regime_label": np.random.randint(0, 3, n_samples),
        }
        df_b = pd.DataFrame(task_b_data)
        os.makedirs(os.path.join(tmpdir, "task_b"), exist_ok=True)
        df_b.to_parquet(os.path.join(tmpdir, "task_b", "train.parquet"), index=False)
        df_b.to_parquet(os.path.join(tmpdir, "task_b", "val.parquet"), index=False)
        df_b.to_parquet(os.path.join(tmpdir, "task_b", "test.parquet"), index=False)

        yield tmpdir


class TestCrisisDataset:
    def test_loads_and_returns_correct_types(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        batch, targets = ds[0]
        assert isinstance(batch, dict)
        assert isinstance(targets, dict)

    def test_batch_dict_keys(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        batch, targets = ds[0]
        assert "historical_ts_numeric" in batch
        assert "future_ts_numeric" in batch

    def test_targets_dict_keys(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        _, targets = ds[0]
        assert "ews_label" in targets
        assert "regime_label" in targets

    def test_historical_shape(self, synthetic_data_dir):
        enc, dec = 10, 3
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=enc, decoder_steps=dec)
        batch, _ = ds[0]
        hist = batch["historical_ts_numeric"]
        assert hist.shape[0] == enc
        assert hist.dim() == 2

    def test_future_shape(self, synthetic_data_dir):
        enc, dec = 10, 3
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=enc, decoder_steps=dec)
        batch, _ = ds[0]
        fut = batch["future_ts_numeric"]
        assert fut.shape[0] == dec

    def test_ews_label_shape(self, synthetic_data_dir):
        enc, dec = 10, 3
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=enc, decoder_steps=dec)
        _, targets = ds[0]
        assert targets["ews_label"].shape == (dec,)

    def test_regime_label_shape(self, synthetic_data_dir):
        enc, dec = 10, 3
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=enc, decoder_steps=dec)
        _, targets = ds[0]
        assert targets["regime_label"].shape == (enc + dec,)

    def test_len(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        assert len(ds) > 0

    def test_tensors_are_float(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        batch, targets = ds[0]
        assert batch["historical_ts_numeric"].dtype == torch.float32
        assert targets["ews_label"].dtype == torch.float32

    def test_dataloader_compatible(self, synthetic_data_dir):
        ds = CrisisDataset(split="train", data_dir=synthetic_data_dir,
                           encoder_steps=10, decoder_steps=3)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
        batch, targets = next(iter(loader))
        assert batch["historical_ts_numeric"].shape[0] == 4


class TestCrisisDatasetSynthetic:
    def test_synthetic_mode(self):
        """Dataset should work without real data by generating synthetic samples."""
        ds = CrisisDataset(split="train", data_dir=None,
                           encoder_steps=10, decoder_steps=3, synthetic=True,
                           num_synthetic_samples=50)
        assert len(ds) > 0
        batch, targets = ds[0]
        assert "historical_ts_numeric" in batch
        assert "ews_label" in targets


class TestCrisisDatasetSplits:
    def test_different_splits_load(self, synthetic_data_dir):
        for split in ["train", "val", "test"]:
            ds = CrisisDataset(split=split, data_dir=synthetic_data_dir,
                               encoder_steps=10, decoder_steps=3)
            assert len(ds) > 0
