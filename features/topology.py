"""Persistent-homology embeddings for crisis prediction (Option C).

What this computes
------------------
At each time t (after a burn-in equal to `window`), we build a weighted graph
whose nodes are the F features and edge weights are `1 - |corr_t(i, j)|`
(rolling-window Pearson correlation, so smaller weight = more correlated).
We then compute persistent homology (H0 only by default, optionally H1) of
the resulting metric space using the Vietoris-Rips filtration.

The persistence diagram is summarised into a fixed-length vector via the
persistence image of Adams et al. (2017). That vector is the "topology
embedding" used as additional input channels in `TopologicalRATFT`.

Why this is useful for crisis prediction
----------------------------------------
Gidea & Polychronakou (2018, 2020) show that the topology of cross-asset /
cross-feature correlation networks undergoes a phase transition before
financial crises — the network becomes more connected, H0 components merge
earlier in the filtration, and characteristic loops appear and disappear in
H1. Classical TFT/LSTM/transformer architectures process channels
independently or via dot-product attention and have no direct access to this
geometric structure. By precomputing it and feeding it as side information,
the model gets crisis-precursor signal that XGBoost likely also cannot
represent (XGB's strength is feature-interaction trees, not network-topology
phase transitions).

Caching
-------
Persistence computation is the slow step (~O(F^3) per timestep). We hash the
inputs and cache the resulting (T_valid, embed_dim) array to disk. Subsequent
runs on the same data with the same parameters reuse the cache. Use
`cache_dir=None` to disable caching (slower; useful in tests).

Dependencies
------------
This module needs giotto-tda. It is NOT in the Stage-0 install — install via
    pip install giotto-tda
when first using this module on the cluster (or via requirements/cluster.txt
when ready for the full Stage 1 environment). The import is lazy so the rest
of the project still works without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TopologyConfig:
    """Hyperparameters for the persistence pipeline.

    window           : rolling window length used to compute correlation.
    homology_dims    : which homology dimensions to compute. (0,) is much faster
                       and usually sufficient for crisis signal; (0,1) is the
                       full picture but ~5x slower per timestep.
    embed_resolution : persistence-image resolution per axis. Total embed
                       dimension = embed_resolution ** 2 * len(homology_dims).
    embed_sigma      : Gaussian smoothing in the persistence image.
    """

    window: int = 60
    homology_dims: Tuple[int, ...] = (0,)
    embed_resolution: int = 8
    embed_sigma: float = 0.1

    @property
    def embed_dim(self) -> int:
        return self.embed_resolution * self.embed_resolution * len(self.homology_dims)

    def cache_key(self, data_shape: Tuple[int, int], data_hash: str) -> str:
        payload = {
            "v": 1,
            "window": self.window,
            "homology_dims": list(self.homology_dims),
            "embed_resolution": self.embed_resolution,
            "embed_sigma": self.embed_sigma,
            "shape": list(data_shape),
            "data_hash": data_hash,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _hash_array(arr: np.ndarray) -> str:
    """Fast-but-stable content hash of an ndarray. Uses raw bytes — order matters."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def _rolling_corr_distance(X: np.ndarray, window: int) -> np.ndarray:
    """For each valid t, return distance matrix d_t[i,j] = 1 - |corr(x_i, x_j)|.

    Shape: (T - window + 1, F, F). Diagonal is 0. Distances clipped to [0, 1].
    """
    T, F = X.shape
    if T < window:
        raise ValueError(f"need at least window={window} timesteps, got T={T}")
    out = np.zeros((T - window + 1, F, F), dtype=np.float32)
    # Numpy-vectorised rolling Pearson. Constant-feature columns (zero std) get
    # zero correlation rather than NaN; that's fine — they encode "no signal."
    for t in range(window - 1, T):
        win = X[t - window + 1 : t + 1]  # (window, F)
        # Center + scale
        c = win - win.mean(axis=0, keepdims=True)
        s = c.std(axis=0, keepdims=True)
        s = np.where(s < 1e-8, 1.0, s)
        z = c / s
        corr = (z.T @ z) / window
        d = 1.0 - np.abs(corr)
        np.fill_diagonal(d, 0.0)
        out[t - window + 1] = np.clip(d, 0.0, 1.0)
    return out


def _persistence_to_image(
    diagrams,
    homology_dims: Tuple[int, ...],
    resolution: int,
    sigma: float,
) -> np.ndarray:
    """Embed a batch of diagrams into persistence images.

    `diagrams` has shape (n_diagrams, n_points, 3) where the last axis is
    (birth, death, dim). We split by `dim`, run PersistenceImage per dim, and
    concatenate.
    """
    from gtda.diagrams import PersistenceImage  # lazy import — optional dep

    out_per_dim = []
    for h in homology_dims:
        # Mask points by dimension; PersistenceImage already tolerates this if
        # we keep the (b, d, h) triple representation.
        single_dim = diagrams.copy()
        single_dim[..., 2] = h  # tell PI to treat everything as one dim
        # Zero-out points whose true dim is not h
        mask = diagrams[..., 2] != h
        single_dim[mask] = 0.0
        pi = PersistenceImage(sigma=sigma, n_bins=resolution)
        img = pi.fit_transform(single_dim)  # (n, 1, R, R)
        out_per_dim.append(img.reshape(img.shape[0], -1))
    return np.concatenate(out_per_dim, axis=1).astype(np.float32)


def compute_topology_features(
    X: np.ndarray,
    config: Optional[TopologyConfig] = None,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Compute persistence-image embeddings of rolling correlation graphs.

    Args:
        X: (T, F) numeric feature matrix. Rows are timesteps, columns features.
        config: TopologyConfig hyperparameters. Defaults to TopologyConfig().
        cache_dir: directory for `<hash>.npy` caches. None disables caching.

    Returns:
        (T, embed_dim) array. Timesteps [0, window-1) are zero-filled so the
        output aligns with the input timeline — that lets callers concatenate
        topology features to the historical channels of a CrisisDataset window
        without index gymnastics.
    """
    config = config or TopologyConfig()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, F), got shape {X.shape}")
    T, F = X.shape

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = config.cache_key((T, F), _hash_array(X))
        cache_path = os.path.join(cache_dir, f"topo_{key}.npy")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            if cached.shape == (T, config.embed_dim):
                return cached

    # 1. Distance matrices.
    distances = _rolling_corr_distance(X, config.window)  # (T_valid, F, F)

    # 2. Vietoris-Rips persistence per timestep.
    from gtda.homology import VietorisRipsPersistence  # lazy import

    vrp = VietorisRipsPersistence(
        metric="precomputed",
        homology_dimensions=list(config.homology_dims),
        n_jobs=1,
    )
    diagrams = vrp.fit_transform(distances)  # (T_valid, n_points, 3)

    # 3. Persistence images.
    embeds = _persistence_to_image(
        diagrams,
        homology_dims=config.homology_dims,
        resolution=config.embed_resolution,
        sigma=config.embed_sigma,
    )  # (T_valid, embed_dim)

    # 4. Pad burn-in with zeros so output aligns with timestep axis of X.
    full = np.zeros((T, config.embed_dim), dtype=np.float32)
    full[config.window - 1 :] = embeds

    if cache_path is not None:
        np.save(cache_path, full)

    return full


def _fallback_topology_features(X: np.ndarray, config: TopologyConfig) -> np.ndarray:
    """Pure-numpy fallback used in tests / when giotto-tda is unavailable.

    Not a real persistence embedding — uses summary statistics of the rolling
    correlation distribution as a stand-in. Lets us exercise the model
    pipeline before installing TDA deps.
    """
    T, F = X.shape
    distances = _rolling_corr_distance(X, config.window)
    # Summary stats: mean off-diagonal corr, std of corrs, min, max, frac > 0.5
    out = np.zeros((T, config.embed_dim), dtype=np.float32)
    n_valid = distances.shape[0]
    for t in range(n_valid):
        d = distances[t]
        offdiag = d[~np.eye(F, dtype=bool)]
        corr = 1.0 - offdiag  # back to corr space for interpretability
        stats = np.array([
            corr.mean(),
            corr.std(),
            corr.min(),
            corr.max(),
            float((corr > 0.5).mean()),
        ], dtype=np.float32)
        # Tile to embed_dim
        reps = config.embed_dim // 5 + 1
        out[config.window - 1 + t] = np.tile(stats, reps)[: config.embed_dim]
    return out
