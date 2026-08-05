"""Per-regime causal DAG learning (Option A preprocessing).

What this does
--------------
Given an (T, F) macro/market feature matrix and (T,) regime labels in
{0=calm, 1=stress, 2=crisis}, learn one differentiable DAG over the F
variables *per regime* using NOTEARS-style continuous optimisation. The
output is a stack of K adjacency matrices of shape (K, F, F), saved to disk
and consumed as an attention-mask prior inside TDT's denoising blocks.

Why per-regime
--------------
The thesis claim TDT is built around — "regimes aren't just attention biases,
they're distinct causal worlds" — is supported by learning a *separate* graph
per regime. In calm regimes you'd expect sparse, mostly real-economy edges
(credit → output, inflation → rates). In crisis regimes the literature
predicts denser, faster-feedback edges within the financial sector (bank
balance sheets → credit spreads → equity prices → bank balance sheets). The
per-regime DAGs visualise this if it exists, and the model gets a structural
prior either way.

NOTEARS-style continuous formulation
------------------------------------
We minimise:
    L(W) = 1/(2n) || X - X W ||_F^2  +  lambda1 * ||W||_1  +  rho/2 * h(W)^2
    where  h(W) = tr( exp(W ∘ W) ) - F     (acyclicity constraint)

We do this with augmented Lagrangian: outer loop increases `rho` until h(W)
is small enough, inner loop is a few hundred Adam steps. This is the
canonical recipe from Zheng et al. (2018). We DO use the `dagma` library
when available (it's faster and more numerically stable than from-scratch
NOTEARS) but we ship a pure-numpy fallback so the pipeline runs without the
optional dep.

Output
------
A dict-like object with:
  adjacency: (K, F, F) float array — A[k, i, j] = strength of edge i -> j in regime k
  feature_names: list[str] of length F (for visualisation)
  config: hyperparameters used

Caching is the same pattern as topology.py — content hash + frozen config.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CausalConfig:
    """Hyperparameters for per-regime DAG learning.

    lambda1 : L1 penalty on edges (sparsity).
    threshold : after fit, set |W| < threshold to zero.
    max_outer : augmented Lagrangian outer iterations.
    max_inner : Adam steps per outer iteration.
    rho_init  : initial penalty for the acyclicity constraint.
    rho_mult  : multiplier per outer iter.
    h_tol     : early-stop tolerance for acyclicity.
    """

    lambda1: float = 0.05
    threshold: float = 0.1
    max_outer: int = 8
    max_inner: int = 300
    rho_init: float = 1.0
    rho_mult: float = 10.0
    h_tol: float = 1e-8
    use_dagma: bool = True

    def cache_key(self, X_shape: Tuple[int, int], X_hash: str, regimes_hash: str, K: int) -> str:
        payload = {
            "v": 1,
            "lambda1": self.lambda1,
            "threshold": self.threshold,
            "max_outer": self.max_outer,
            "max_inner": self.max_inner,
            "rho_init": self.rho_init,
            "rho_mult": self.rho_mult,
            "h_tol": self.h_tol,
            "use_dagma": self.use_dagma,
            "X_shape": list(X_shape),
            "X_hash": X_hash,
            "regimes_hash": regimes_hash,
            "K": K,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


# -------------- Pure-numpy NOTEARS (fallback) --------------

def _h_acyclic(W: np.ndarray) -> float:
    """Acyclicity penalty h(W) = tr(exp(W ∘ W)) − F. Zero iff W is a DAG."""
    F = W.shape[0]
    M = W * W
    return float(np.trace(_matrix_exp_psd(M)) - F)


def _matrix_exp_psd(M: np.ndarray) -> np.ndarray:
    """Truncated matrix exponential — sufficient since M = W∘W ≥ 0 with small
    spectral radius after a few Adam steps. Stable for F up to ~50."""
    # Scaling and squaring with order-6 Taylor — adequate for our F.
    norm = np.linalg.norm(M, ord=1)
    s = max(0, int(np.ceil(np.log2(max(norm, 1.0)))))
    A = M / (2 ** s)
    exp = np.eye(M.shape[0]) + A
    A_k = A
    for k in range(2, 7):
        A_k = A_k @ A / k
        exp = exp + A_k
    for _ in range(s):
        exp = exp @ exp
    return exp


def _grad_h(W: np.ndarray) -> np.ndarray:
    """dh/dW. Standard NOTEARS expression."""
    M = W * W
    expM = _matrix_exp_psd(M)
    return 2 * expM.T * W


def _fit_dag_numpy(X: np.ndarray, cfg: CausalConfig) -> np.ndarray:
    """Vanilla NOTEARS via projected gradient with augmented Lagrangian.

    X: (n, F) standardised data.
    Returns W: (F, F) with W[i, j] interpreted as edge i -> j (parents -> children).
    """
    n, F = X.shape
    W = np.zeros((F, F), dtype=np.float64)
    rho = cfg.rho_init
    alpha = 0.0
    lr = 1e-2
    h_prev = np.inf

    for outer in range(cfg.max_outer):
        # m1/m2 for Adam.
        m1 = np.zeros_like(W)
        m2 = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(1, cfg.max_inner + 1):
            R = X - X @ W
            grad_mse = -X.T @ R / n
            grad_l1 = cfg.lambda1 * np.sign(W)
            h = _h_acyclic(W)
            grad_h = _grad_h(W)
            grad_h_full = (rho * h + alpha) * grad_h
            grad = grad_mse + grad_l1 + grad_h_full
            # zero diagonal — no self loops
            np.fill_diagonal(grad, 0.0)
            m1 = b1 * m1 + (1 - b1) * grad
            m2 = b2 * m2 + (1 - b2) * grad * grad
            m1_hat = m1 / (1 - b1 ** step)
            m2_hat = m2 / (1 - b2 ** step)
            W = W - lr * m1_hat / (np.sqrt(m2_hat) + eps)
            np.fill_diagonal(W, 0.0)
        h_curr = _h_acyclic(W)
        if h_curr <= cfg.h_tol:
            break
        if h_curr > 0.25 * h_prev:
            rho *= cfg.rho_mult
        alpha = alpha + rho * h_curr
        h_prev = h_curr

    # Threshold small edges.
    W[np.abs(W) < cfg.threshold] = 0.0
    return W.astype(np.float32)


def _fit_dag_dagma(X: np.ndarray, cfg: CausalConfig) -> np.ndarray:
    """Use the `dagma` library if installed. Faster, more numerically stable."""
    from dagma.linear import DagmaLinear  # lazy import — optional dep

    model = DagmaLinear(loss_type="l2")
    W = model.fit(X, lambda1=cfg.lambda1, T=cfg.max_outer, warm_iter=cfg.max_inner)
    W[np.abs(W) < cfg.threshold] = 0.0
    return W.astype(np.float32)


# -------------- Public API --------------

def learn_regime_dags(
    X: np.ndarray,
    regimes: np.ndarray,
    K: int = 3,
    feature_names: Optional[List[str]] = None,
    config: Optional[CausalConfig] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Learn one DAG per regime.

    Args:
        X: (T, F) standardised feature matrix.
        regimes: (T,) int array of regime labels in [0, K).
        K: number of regimes (default 3 = calm/stress/crisis).
        feature_names: optional list of length F for downstream visualisation.
        config: CausalConfig.
        cache_dir: directory to cache the (K, F, F) adjacency array. None disables.

    Returns:
        dict with keys "adjacency" (K, F, F), "feature_names", "config",
        "n_per_regime" (K,). Always returns; raises only on invalid inputs.
    """
    cfg = config or CausalConfig()
    X = np.asarray(X, dtype=np.float64)
    regimes = np.asarray(regimes, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError(f"X must be (T, F), got shape {X.shape}")
    if regimes.shape != (X.shape[0],):
        raise ValueError(f"regimes shape {regimes.shape} != (T={X.shape[0]},)")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    T, F = X.shape
    feature_names = feature_names or [f"f{i}" for i in range(F)]

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = cfg.cache_key((T, F), _hash_array(X), _hash_array(regimes), K)
        cache_path = os.path.join(cache_dir, f"dag_{key}.npz")
        if os.path.exists(cache_path):
            blob = np.load(cache_path, allow_pickle=True)
            return {
                "adjacency": blob["adjacency"],
                "feature_names": list(blob["feature_names"]),
                "config": cfg,
                "n_per_regime": blob["n_per_regime"],
            }

    adjacency = np.zeros((K, F, F), dtype=np.float32)
    n_per_regime = np.zeros(K, dtype=np.int64)

    fit_fn = _fit_dag_dagma if cfg.use_dagma else _fit_dag_numpy
    for k in range(K):
        mask = regimes == k
        n_per_regime[k] = int(mask.sum())
        if n_per_regime[k] < F + 2:
            # Not enough data — leave that regime as a zero matrix (no prior).
            continue
        Xk = X[mask]
        # Standardise within regime to put each variable on the same scale.
        Xk = (Xk - Xk.mean(axis=0)) / np.where(Xk.std(axis=0) < 1e-8, 1.0, Xk.std(axis=0))
        try:
            W = fit_fn(Xk, cfg)
        except ImportError:
            # dagma not installed — fall back to numpy NOTEARS.
            W = _fit_dag_numpy(Xk, cfg)
        adjacency[k] = W

    if cache_path is not None:
        np.savez(
            cache_path,
            adjacency=adjacency,
            feature_names=np.array(feature_names),
            n_per_regime=n_per_regime,
        )

    return {
        "adjacency": adjacency,
        "feature_names": feature_names,
        "config": cfg,
        "n_per_regime": n_per_regime,
    }


def adjacency_to_attention_mask(
    adjacency: np.ndarray,
    regime_probs: np.ndarray,
    floor: float = 0.0,
) -> np.ndarray:
    """Convert per-regime DAGs to a regime-mixed attention bias.

    Args:
        adjacency: (K, F, F) per-regime adjacency.
        regime_probs: (B, K) or (B, T, K) regime probability vector(s).
        floor: minimum mask value (so a non-edge isn't completely blocked).

    Returns:
        Mask of shape (B, F, F) or (B, T, F, F). The mask is meant to be
        ADDED to attention logits before softmax — positive values encourage
        attention to causally-related variates, zero/negative dampen others.
    """
    K, F1, F2 = adjacency.shape
    assert F1 == F2, "DAG adjacency must be square"
    if regime_probs.ndim == 2:
        # (B, K) @ (K, F*F) -> (B, F*F) -> (B, F, F)
        mask = regime_probs @ adjacency.reshape(K, -1)
        mask = mask.reshape(-1, F1, F1)
    elif regime_probs.ndim == 3:
        B, T, _ = regime_probs.shape
        mask = regime_probs.reshape(-1, K) @ adjacency.reshape(K, -1)
        mask = mask.reshape(B, T, F1, F1)
    else:
        raise ValueError(f"regime_probs must be (B,K) or (B,T,K); got {regime_probs.shape}")
    return np.maximum(mask, floor)
