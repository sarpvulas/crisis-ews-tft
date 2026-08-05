"""Tests for TopologyAugmentedDataset.

Uses the fallback (summary-statistic) topology path so the test runs without
giotto-tda. The wrapped dataset is a synthetic CrisisDataset.
"""

import numpy as np
import pytest
import torch

from features.topology import TopologyConfig
from training.dataset import CrisisDataset
from training.topology_dataset import TopologyAugmentedDataset


@pytest.fixture
def synthetic_dataset():
    return CrisisDataset(
        split="train",
        synthetic=True,
        num_synthetic_samples=200,
        encoder_steps=40,
        decoder_steps=10,
    )


class TestTopologyAugmentedDataset:

    def test_feature_dim_inflated(self, synthetic_dataset):
        cfg = TopologyConfig(window=20, embed_resolution=4, homology_dims=(0,))
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        dims = wrapped.get_feature_dims()
        assert dims["num_historical_numeric"] == synthetic_dataset._num_historical + cfg.embed_dim

    def test_window_shape_matches_inflated_dim(self, synthetic_dataset):
        cfg = TopologyConfig(window=20, embed_resolution=4, homology_dims=(0,))
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        batch, targets = wrapped[0]
        hist = batch["historical_ts_numeric"]
        assert hist.shape == (synthetic_dataset.encoder_steps,
                              synthetic_dataset._num_historical + cfg.embed_dim)
        # decoder targets unchanged
        assert targets["ews_label"].shape == (synthetic_dataset.decoder_steps,)

    def test_topology_channels_are_finite(self, synthetic_dataset):
        cfg = TopologyConfig(window=20, embed_resolution=4, homology_dims=(0,))
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        batch, _ = wrapped[5]
        assert torch.isfinite(batch["historical_ts_numeric"]).all()

    def test_collate_fn_stacks_windows(self, synthetic_dataset):
        cfg = TopologyConfig(window=20, embed_resolution=4, homology_dims=(0,))
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        items = [wrapped[i] for i in range(3)]
        batch, targets = wrapped.collate_fn(items)
        assert batch["historical_ts_numeric"].shape[0] == 3
        assert targets["ews_label"].shape[0] == 3

    def test_passthrough_properties(self, synthetic_dataset):
        cfg = TopologyConfig(window=20, embed_resolution=4)
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        assert wrapped.encoder_steps == synthetic_dataset.encoder_steps
        assert wrapped.decoder_steps == synthetic_dataset.decoder_steps
        assert wrapped.feature_scaler is synthetic_dataset.feature_scaler

    def test_topology_alignment_with_burn_in(self, synthetic_dataset):
        """First few timesteps in a window before the rolling window has filled
        should be zero-filled topology features."""
        cfg = TopologyConfig(window=20, embed_resolution=4)
        wrapped = TopologyAugmentedDataset(synthetic_dataset, cfg, use_real_topology=False)
        # Pick a window starting at the very beginning of the series.
        # The first (window - 1) topology rows are zero.
        batch, _ = wrapped[0]
        hist = batch["historical_ts_numeric"].numpy()
        F = synthetic_dataset._num_historical
        # Topology columns over the first (window - 1) timesteps should be zero
        # *if* the window's starting index is 0. (Synthetic data has start=0 for
        # idx=0 because valid_indices = list(range(...)).)
        if wrapped.valid_indices[0] == 0:
            np.testing.assert_array_equal(hist[: cfg.window - 1, F:], 0.0)

    def test_non_crisis_dataset_rejected(self):
        """Wrapping a dataset that lacks `historical_numeric` raises TypeError."""

        class FakeDataset:
            pass

        with pytest.raises(TypeError):
            TopologyAugmentedDataset(FakeDataset(), use_real_topology=False)
