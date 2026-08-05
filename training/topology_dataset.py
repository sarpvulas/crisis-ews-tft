"""TopologyAugmentedDataset — wraps a CrisisDataset with persistence features.

This is the dataset side of Option C (Topological + Transformer). It precomputes
persistence-image embeddings over the underlying time series exactly once, then
splices a slice of those embeddings onto every window's historical channels.

Why a wrapper instead of subclassing CrisisDataset
--------------------------------------------------
The base CrisisDataset is loadable from real data, synthetic data, or a
walk-forward fold. A subclass would need to override three constructors. A
wrapper is generic — it works for any dataset that exposes the protocol used
by the existing training loop (a `historical_numeric` array, a `valid_indices`
list, a `collate_fn`, and the standard `(batch, targets)` __getitem__ return).

Caching
-------
Persistence computation is O(T * F^3); for a 25-year monthly Task-A dataset
with F=30 macro variables and T=300 it's a few seconds, but for a daily Task-B
series with T=5000 and F=27 it's several minutes. We cache to disk keyed by a
content hash of the underlying ndarray + topology config (see
features.topology.TopologyConfig.cache_key). On the cluster, cache lives in
/scratch/users/$USER/ra-tft/features/topology/.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from features.topology import (
    TopologyConfig,
    _fallback_topology_features,
    compute_topology_features,
)


class TopologyAugmentedDataset(Dataset):
    """Wraps a CrisisDataset and appends topology-embedding channels to the
    historical features of every window.

    The wrapped dataset's `__getitem__` is called as-is; we then concatenate a
    slice of the precomputed topology array to `batch["historical_ts_numeric"]`.

    Attributes mirrored from the underlying dataset (so existing code reading
    `feature_scaler`, `_num_static`, etc. still works):
        - feature_scaler
        - get_feature_dims (overridden to advertise the inflated historical
          channel count)
    """

    def __init__(
        self,
        base: Dataset,
        topology_config: Optional[TopologyConfig] = None,
        cache_dir: Optional[str] = None,
        use_real_topology: bool = True,
    ):
        if not hasattr(base, "historical_numeric"):
            raise TypeError(
                "TopologyAugmentedDataset requires the wrapped dataset to expose "
                "`historical_numeric` (a (T, F) array). CrisisDataset does; "
                "PanelCrisisDataset stores arrays per market and needs a separate wrapper."
            )
        self.base = base
        self.config = topology_config or TopologyConfig()
        self.cache_dir = cache_dir
        self.use_real_topology = use_real_topology

        # Compute (or load) topology embedding for the underlying series.
        self._topology = self._build_topology(base.historical_numeric)

        # Inflated channel count.
        self._num_historical_total = (
            int(base._num_historical) + int(self.config.embed_dim)
        )

    # ----------- core API -----------

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, targets = self.base[idx]
        # Look up which window we are at. The base dataset's __getitem__ uses
        # valid_indices[idx] internally; we need the same start so the topology
        # slice aligns timestep-for-timestep with the historical slice.
        start = self.base.valid_indices[idx]
        enc_end = start + self.base.encoder_steps
        topo_slice = self._topology[start:enc_end]
        topo_tensor = torch.tensor(topo_slice, dtype=torch.float32)
        # Concat along the feature axis. Shape: (encoder_steps, F + embed_dim)
        batch["historical_ts_numeric"] = torch.cat(
            [batch["historical_ts_numeric"], topo_tensor], dim=-1
        )
        return batch, targets

    @staticmethod
    def collate_fn(batch):
        # Same collation as the base; we only changed feature width per item.
        from training.dataset import CrisisDataset

        return CrisisDataset.collate_fn(batch)

    # ----------- mirrored properties -----------

    @property
    def feature_scaler(self):
        return self.base.feature_scaler

    @property
    def encoder_steps(self) -> int:
        return self.base.encoder_steps

    @property
    def decoder_steps(self) -> int:
        return self.base.decoder_steps

    @property
    def valid_indices(self) -> List[int]:
        return self.base.valid_indices

    @property
    def _num_historical(self) -> int:
        return self._num_historical_total

    @property
    def _num_static(self) -> int:
        return int(self.base._num_static)

    @property
    def _num_future(self) -> int:
        return int(self.base._num_future)

    def get_feature_dims(self) -> Dict[str, int]:
        dims = self.base.get_feature_dims()
        dims["num_historical_numeric"] = self._num_historical_total
        return dims

    # ----------- internals -----------

    def _build_topology(self, X: np.ndarray) -> np.ndarray:
        """Compute or load persistence-image embedding for the full series."""
        if self.use_real_topology:
            try:
                return compute_topology_features(X, self.config, cache_dir=self.cache_dir)
            except ImportError:
                # giotto-tda missing — fall back to summary stats so training
                # still works (useful for tests and pre-Stage-1 environments).
                pass
        return _fallback_topology_features(X, self.config)
