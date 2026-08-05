"""Tests for features.causal.

Uses small synthetic data to exercise the pure-numpy NOTEARS path. The
real-dagma path is invoked only if the optional dep is installed.
"""

import os
import tempfile

import numpy as np
import pytest

from features.causal import (
    CausalConfig,
    _fit_dag_numpy,
    _h_acyclic,
    adjacency_to_attention_mask,
    learn_regime_dags,
)


@pytest.fixture
def synthetic_with_known_edges():
    """Two-regime data with intentionally different causal structure:
      regime 0: f0 -> f1 -> f2 (chain)
      regime 1: f0 -> f1, f0 -> f2 (fork)
    """
    rng = np.random.RandomState(0)
    n_per = 400
    F = 3

    # Regime 0: chain
    X0 = np.zeros((n_per, F))
    X0[:, 0] = rng.normal(0, 1, n_per)
    X0[:, 1] = 0.8 * X0[:, 0] + rng.normal(0, 0.5, n_per)
    X0[:, 2] = 0.7 * X0[:, 1] + rng.normal(0, 0.5, n_per)

    # Regime 1: fork
    X1 = np.zeros((n_per, F))
    X1[:, 0] = rng.normal(0, 1, n_per)
    X1[:, 1] = 0.8 * X1[:, 0] + rng.normal(0, 0.5, n_per)
    X1[:, 2] = 0.6 * X1[:, 0] + rng.normal(0, 0.5, n_per)

    X = np.vstack([X0, X1])
    regimes = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    return X, regimes


class TestAcyclicity:

    def test_zero_matrix_is_acyclic(self):
        W = np.zeros((5, 5))
        assert _h_acyclic(W) == pytest.approx(0.0, abs=1e-6)

    def test_chain_is_acyclic(self):
        # Upper-triangular -> DAG
        W = np.zeros((4, 4))
        W[0, 1] = W[1, 2] = W[2, 3] = 0.5
        h = _h_acyclic(W)
        assert h < 0.5

    def test_cycle_is_not_acyclic(self):
        # 2-cycle
        W = np.zeros((3, 3))
        W[0, 1] = W[1, 0] = 0.5
        assert _h_acyclic(W) > 0.0


class TestNotearsFit:

    def test_recovers_chain_direction(self):
        rng = np.random.RandomState(0)
        n = 800
        x0 = rng.normal(0, 1, n)
        x1 = 0.9 * x0 + rng.normal(0, 0.3, n)
        x2 = 0.8 * x1 + rng.normal(0, 0.3, n)
        X = np.stack([x0, x1, x2], axis=1)
        # Standardise (mirrors learn_regime_dags's preprocessing)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        cfg = CausalConfig(lambda1=0.02, threshold=0.05, max_outer=4, max_inner=200)
        W = _fit_dag_numpy(X, cfg)
        # The strongest edges should be x0->x1 and x1->x2.
        # We accept up to one false-positive edge; main thing: those two are present.
        assert abs(W[0, 1]) > 0.2, f"x0->x1 weak: {W[0, 1]:.3f}"
        assert abs(W[1, 2]) > 0.2, f"x1->x2 weak: {W[1, 2]:.3f}"

    def test_no_self_loops(self):
        rng = np.random.RandomState(0)
        X = rng.randn(400, 4)
        cfg = CausalConfig(max_outer=2, max_inner=100)
        W = _fit_dag_numpy(X, cfg)
        np.testing.assert_array_equal(np.diag(W), 0.0)


class TestLearnRegimeDags:

    def test_shape_and_caching(self, synthetic_with_known_edges):
        X, regimes = synthetic_with_known_edges
        cfg = CausalConfig(lambda1=0.02, threshold=0.05, max_outer=3, max_inner=150,
                           use_dagma=False)
        with tempfile.TemporaryDirectory() as cache_dir:
            out1 = learn_regime_dags(X, regimes, K=2, config=cfg, cache_dir=cache_dir)
            assert out1["adjacency"].shape == (2, X.shape[1], X.shape[1])
            assert out1["n_per_regime"].tolist() == [400, 400]
            # Hit the cache the second time.
            out2 = learn_regime_dags(X, regimes, K=2, config=cfg, cache_dir=cache_dir)
            np.testing.assert_array_equal(out1["adjacency"], out2["adjacency"])

    def test_per_regime_structures_differ(self, synthetic_with_known_edges):
        X, regimes = synthetic_with_known_edges
        cfg = CausalConfig(lambda1=0.02, threshold=0.05, max_outer=4, max_inner=250,
                           use_dagma=False)
        out = learn_regime_dags(X, regimes, K=2, config=cfg)
        A0, A1 = out["adjacency"]
        # The two regimes have different structures by construction, so the
        # learned adjacencies must not be identical.
        assert not np.allclose(A0, A1, atol=1e-3), "regime DAGs collapsed to same matrix"

    def test_unknown_regime_left_zero(self):
        rng = np.random.RandomState(0)
        X = rng.randn(120, 4)
        regimes = np.zeros(120, dtype=int)  # only regime 0 present
        cfg = CausalConfig(max_outer=2, max_inner=100, use_dagma=False)
        out = learn_regime_dags(X, regimes, K=3, config=cfg)
        # Regime 1 and 2 have no data — adjacency must be zero.
        np.testing.assert_array_equal(out["adjacency"][1], 0.0)
        np.testing.assert_array_equal(out["adjacency"][2], 0.0)
        assert out["n_per_regime"].tolist() == [120, 0, 0]


class TestAttentionMask:

    def test_shape_for_batched_regime_probs(self):
        K, F = 3, 5
        adjacency = np.random.rand(K, F, F).astype(np.float32)
        probs = np.random.rand(4, K).astype(np.float32)
        probs /= probs.sum(axis=-1, keepdims=True)
        mask = adjacency_to_attention_mask(adjacency, probs)
        assert mask.shape == (4, F, F)

    def test_shape_for_time_resolved_regime_probs(self):
        K, F = 3, 4
        adjacency = np.random.rand(K, F, F).astype(np.float32)
        probs = np.random.rand(2, 7, K).astype(np.float32)
        probs /= probs.sum(axis=-1, keepdims=True)
        mask = adjacency_to_attention_mask(adjacency, probs)
        assert mask.shape == (2, 7, F, F)

    def test_pure_regime_returns_that_dag(self):
        K, F = 3, 4
        adjacency = np.random.rand(K, F, F).astype(np.float32)
        # One-hot regime 1
        probs = np.zeros((1, K), dtype=np.float32)
        probs[0, 1] = 1.0
        mask = adjacency_to_attention_mask(adjacency, probs, floor=-np.inf)
        np.testing.assert_allclose(mask[0], adjacency[1])
