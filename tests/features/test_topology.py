"""Tests for features.topology.

The real giotto-tda import path is exercised only when the optional dep is
installed (CI may not have it). The fallback summary-statistic path is always
tested so the model wiring can be validated end-to-end.
"""

import os
import tempfile

import numpy as np
import pytest

from features.topology import (
    TopologyConfig,
    _fallback_topology_features,
    _rolling_corr_distance,
    compute_topology_features,
)


class TestRollingCorrDistance:

    def test_shape(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 6)
        d = _rolling_corr_distance(X, window=20)
        assert d.shape == (81, 6, 6)

    def test_diagonal_is_zero(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 4)
        d = _rolling_corr_distance(X, window=20)
        np.testing.assert_allclose(np.diagonal(d, axis1=-2, axis2=-1), 0.0, atol=1e-6)

    def test_perfectly_correlated_features_have_zero_distance(self):
        rng = np.random.RandomState(0)
        x = rng.randn(50, 1)
        X = np.concatenate([x, x, -x], axis=1)  # cols 0 & 1 perfectly +corr; 0 & 2 perfectly -corr
        d = _rolling_corr_distance(X, window=20)
        # |corr| = 1 -> distance ~ 0
        np.testing.assert_allclose(d[:, 0, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(d[:, 0, 2], 0.0, atol=1e-4)

    def test_window_larger_than_data_raises(self):
        with pytest.raises(ValueError):
            _rolling_corr_distance(np.zeros((10, 3)), window=20)


class TestTopologyConfig:

    def test_embed_dim_h0_only(self):
        c = TopologyConfig(homology_dims=(0,), embed_resolution=8)
        assert c.embed_dim == 64

    def test_embed_dim_h0_h1(self):
        c = TopologyConfig(homology_dims=(0, 1), embed_resolution=8)
        assert c.embed_dim == 128

    def test_cache_key_is_stable(self):
        c = TopologyConfig()
        k1 = c.cache_key((100, 5), "abc")
        k2 = c.cache_key((100, 5), "abc")
        assert k1 == k2

    def test_cache_key_changes_on_shape(self):
        c = TopologyConfig()
        assert c.cache_key((100, 5), "abc") != c.cache_key((100, 6), "abc")


class TestFallbackPipeline:
    """The fallback uses summary stats only — keeps the test suite TDA-dep-free."""

    def test_output_shape_aligned_with_input(self):
        rng = np.random.RandomState(0)
        X = rng.randn(60, 5)
        cfg = TopologyConfig(window=20)
        out = _fallback_topology_features(X, cfg)
        assert out.shape == (60, cfg.embed_dim)

    def test_burn_in_rows_are_zero(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 4)
        cfg = TopologyConfig(window=15)
        out = _fallback_topology_features(X, cfg)
        np.testing.assert_array_equal(out[: cfg.window - 1], 0.0)


class TestRealPipeline:
    """Run the real giotto-tda pipeline only if the optional dep is installed."""

    def test_shape_and_caching(self):
        pytest.importorskip("gtda")
        rng = np.random.RandomState(0)
        X = rng.randn(80, 4).astype(np.float32)
        cfg = TopologyConfig(window=20, embed_resolution=4, homology_dims=(0,))
        with tempfile.TemporaryDirectory() as cache_dir:
            out1 = compute_topology_features(X, cfg, cache_dir=cache_dir)
            assert out1.shape == (80, cfg.embed_dim)
            # Burn-in zero-filled.
            np.testing.assert_array_equal(out1[: cfg.window - 1], 0.0)
            # Cache hit produces identical output.
            out2 = compute_topology_features(X, cfg, cache_dir=cache_dir)
            np.testing.assert_array_equal(out1, out2)
            # Cache file exists.
            files = [f for f in os.listdir(cache_dir) if f.startswith("topo_")]
            assert len(files) == 1
