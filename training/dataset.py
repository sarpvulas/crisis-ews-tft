"""PyTorch Datasets for crisis Early Warning System.

CrisisDataset: single-market sequential windows.
PanelCrisisDataset: multi-market panel with market_id as static categorical.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Default column groups
HISTORICAL_NUMERIC = [
    # Price features
    "log_ret_1d", "log_ret_5d", "log_ret_21d",
    "rvol_5d", "rvol_21d", "rvol_63d",
    # Short-horizon drawdown + vol regime indicators
    "drawdown_63d", "vol_of_vol_63d", "return_skew_63d",
    "hurst_63d",
    # Liquidity
    "amihud_21d", "volume_zscore_63d",
    # External
    "vix", "vix3m", "hy_spread", "ig_spread", "ted_spread",
    "gold", "dxy", "skew", "move_index",
    # Derived
    "vix_term_structure", "implied_realized_spread",
    "credit_spread_diff",
]
FUTURE_NUMERIC = ["day_sin", "day_cos"]
LABEL_COLS = ["ews_label", "in_crisis", "drawdown", "regime_label"]


class CrisisDataset(Dataset):
    """Dataset for crisis Early Warning System (EWS).

    Returns (batch_dict, targets_dict) tuples compatible with TFT forward() and CrisisAwareLoss.

    batch_dict keys:
        - static_feats_numeric: (num_static_feats,) — if any static features exist
        - historical_ts_numeric: (encoder_steps, num_hist_feats)
        - future_ts_numeric: (decoder_steps, num_future_feats)

    targets_dict keys:
        ews_label (decoder_steps,), regime_label (encoder+decoder,)
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[str] = None,
        encoder_steps: int = 252,
        decoder_steps: int = 63,
        synthetic: bool = False,
        num_synthetic_samples: int = 500,
        static_numeric_cols: Optional[List[str]] = None,
        historical_numeric_cols: Optional[List[str]] = None,
        future_numeric_cols: Optional[List[str]] = None,
        feature_scaler: Optional[dict] = None,
        exclude_crisis_windows: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.task = "B"
        self.split = split
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.window_size = encoder_steps + decoder_steps
        self._feature_scaler = feature_scaler  # None = fit new, dict = reuse

        self.static_numeric_cols = static_numeric_cols or []
        self.historical_numeric_cols = historical_numeric_cols or HISTORICAL_NUMERIC
        self.future_numeric_cols = future_numeric_cols or FUTURE_NUMERIC
        self.label_cols = LABEL_COLS

        if synthetic or data_dir is None:
            self._build_synthetic(num_synthetic_samples)
        else:
            self._load_parquet(data_dir)

        # Build valid window indices — exclude windows where "today" is already in crisis
        all_windows = list(range(max(0, len(self.df) - self.window_size + 1)))

        if exclude_crisis_windows and not synthetic:
            self.valid_indices = [
                i for i in all_windows
                if not self.in_crisis[i + self.encoder_steps - 1]
            ]
            # Fallback for crisis-dominated splits: some walk-forward folds place
            # the val (or train) window on a PRIOR crisis (e.g. eurozone's val
            # overlaps the GFC; selloff2018's overlaps China-2015), so excluding
            # in-crisis anchors can wipe out ALL windows and crash the run with
            # "need at least one array to concatenate". Relax the exclusion only
            # when the calm-only count is too small. Calm splits keep ~100+
            # windows and are unaffected; never relaxes the test split (eval
            # semantics unchanged).
            if self.split != "test" and len(self.valid_indices) < 32:
                self.valid_indices = all_windows
        else:
            self.valid_indices = all_windows

        self.num_windows = len(self.valid_indices)

    def _load_fold(self, fold_dir, fold_name: str, split: str, feature_scaler=None) -> None:
        """Load data from a walk-forward fold directory."""
        path = os.path.join(fold_dir, f"{fold_name}_{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fold file not found: {path}")
        self._feature_scaler = feature_scaler
        self.df = pd.read_parquet(path)
        self.static_numeric_cols = [c for c in self.static_numeric_cols if c in self.df.columns]
        self.historical_numeric_cols = [c for c in self.historical_numeric_cols if c in self.df.columns]
        self.future_numeric_cols = [c for c in self.future_numeric_cols if c in self.df.columns]
        self._prepare_tensors()

        # Rebuild valid indices
        all_windows = list(range(max(0, len(self.df) - self.window_size + 1)))
        if self.split != "test":
            self.valid_indices = [
                i for i in all_windows
                if not self.in_crisis[i + self.encoder_steps - 1]
            ]
        else:
            self.valid_indices = all_windows
        self.num_windows = len(self.valid_indices)

    def _load_parquet(self, data_dir: str) -> None:
        """Load data from Parquet file."""
        # Try both naming conventions
        path = os.path.join(data_dir, "task_b", f"{self.split}.parquet")
        if not os.path.exists(path):
            path = os.path.join(data_dir, f"task_b_{self.split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No parquet file found for split={self.split} in {data_dir}")
        self.df = pd.read_parquet(path)
        # Filter to columns that actually exist in the dataframe
        self.static_numeric_cols = [c for c in self.static_numeric_cols if c in self.df.columns]
        self.historical_numeric_cols = [c for c in self.historical_numeric_cols if c in self.df.columns]
        self.future_numeric_cols = [c for c in self.future_numeric_cols if c in self.df.columns]
        self._prepare_tensors()

    def _build_synthetic(self, n: int) -> None:
        """Generate synthetic data matching the expected schema."""
        np.random.seed(42)
        total_len = n + self.window_size  # Ensure enough data for windows

        data = {}
        for col in self.static_numeric_cols:
            data[col] = np.random.randn(total_len)
        for col in self.historical_numeric_cols:
            data[col] = np.random.randn(total_len)
        for col in self.future_numeric_cols:
            data[col] = np.random.randn(total_len)

        data["regime_label"] = np.random.randint(0, 3, total_len)
        data["ews_label"] = np.random.binomial(1, 0.1, total_len).astype(float)
        data["in_crisis"] = np.random.binomial(1, 0.05, total_len).astype(bool)

        self.df = pd.DataFrame(data)
        self._prepare_tensors()

    def _prepare_tensors(self) -> None:
        """Convert DataFrame columns to numpy arrays for fast indexing."""
        n = len(self.df)

        # Static features (may not exist in real data)
        avail_static = [c for c in self.static_numeric_cols if c in self.df.columns]
        self.static_numeric = self.df[avail_static].values.astype(np.float32) if avail_static else np.zeros((n, 0), dtype=np.float32)
        # Update num static for model config
        self._num_static = self.static_numeric.shape[1]

        # Historical features — use all numeric columns except labels/metadata/raw levels
        exclude = {"date", "country", "crisis_label", "crash_label", "regime_label", "crisis_mask",
                   "forward_drawdown", "close", "volume",
                   # label/metadata columns — must NOT leak into the feature set
                   # (matches PanelCrisisDataset's exclude list; their omission here let
                   #  ews_label/drawdown leak as inputs and inverted RA-TFT at test time)
                   "ews_label", "in_crisis", "drawdown"}
        avail_hist = [c for c in self.historical_numeric_cols if c in self.df.columns]
        # Always add any numeric columns from real data that aren't in the predefined list
        extra_hist = [c for c in self.df.columns if c not in exclude and c not in avail_hist
                      and self.df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        avail_hist = avail_hist + extra_hist
        raw_hist = self.df[avail_hist].values.astype(np.float32)

        # Feature normalization: z-score standardization
        if self._feature_scaler is not None:
            # Reuse scaler from training set (val/test)
            mean = self._feature_scaler["mean"]
            std = self._feature_scaler["std"]
        else:
            # Fit scaler on this data (training set)
            mean = np.nanmean(raw_hist, axis=0)
            std = np.nanstd(raw_hist, axis=0)
            std[std < 1e-8] = 1.0  # Avoid division by zero for constant features
            self._feature_scaler = {"mean": mean, "std": std}

        self.historical_numeric = (raw_hist - mean) / std
        # Replace any remaining NaN/Inf after normalization
        self.historical_numeric = np.nan_to_num(self.historical_numeric, nan=0.0, posinf=0.0, neginf=0.0)
        self._num_historical = self.historical_numeric.shape[1]

        # Future features (may not exist — generate calendar features)
        avail_future = [c for c in self.future_numeric_cols if c in self.df.columns]
        if avail_future:
            self.future_numeric = self.df[avail_future].values.astype(np.float32)
        else:
            # Generate simple positional features
            idx = np.arange(n, dtype=np.float32)
            self.future_numeric = np.stack([np.sin(idx * 2 * np.pi / 252), np.cos(idx * 2 * np.pi / 252)], axis=1)
        self._num_future = self.future_numeric.shape[1]

        # Regime labels (auxiliary target for regime module)
        if "regime_label" in self.df.columns:
            self.regime_labels = self.df["regime_label"].values.astype(np.int64)
        else:
            self.regime_labels = np.zeros(n, dtype=np.int64)

        # EWS labels
        if "ews_label" in self.df.columns:
            self.ews_labels = self.df["ews_label"].values.astype(np.float32)
        else:
            self.ews_labels = np.zeros(n, dtype=np.float32)

        # In-crisis flag (for excluding windows during training)
        if "in_crisis" in self.df.columns:
            self.in_crisis = self.df["in_crisis"].values.astype(bool)
        else:
            self.in_crisis = np.zeros(n, dtype=bool)

        # Drawdown series — auxiliary regression target (excluded from features).
        if "drawdown" in self.df.columns:
            self.drawdown = np.nan_to_num(self.df["drawdown"].values.astype(np.float32), nan=0.0)
        else:
            self.drawdown = np.zeros(n, dtype=np.float32)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        start = self.valid_indices[idx]
        enc_end = start + self.encoder_steps
        dec_end = enc_end + self.decoder_steps

        # Build batch dict
        batch = {}

        # Static features: use mean over encoder window (static per entity in real data)
        if self.static_numeric.shape[1] > 0:
            static = self.static_numeric[start:enc_end].mean(axis=0)
            batch["static_feats_numeric"] = torch.tensor(static, dtype=torch.float32)

        # Historical: encoder window
        batch["historical_ts_numeric"] = torch.tensor(
            self.historical_numeric[start:enc_end], dtype=torch.float32
        )

        # Future: decoder window
        batch["future_ts_numeric"] = torch.tensor(
            self.future_numeric[enc_end:dec_end], dtype=torch.float32
        )

        # Build targets dict
        targets = {}
        targets["regime_label"] = torch.tensor(
            self.regime_labels[start:dec_end], dtype=torch.long
        )
        targets["ews_label"] = torch.tensor(
            self.ews_labels[enc_end:dec_end], dtype=torch.float32
        )
        # Auxiliary forward-drawdown target (decoder window). Harmless when the
        # model has no aux head — CrisisAwareLoss only uses it if aux_pred exists.
        targets["drawdown_target"] = torch.tensor(
            self.drawdown[enc_end:dec_end], dtype=torch.float32
        )

        return batch, targets

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Custom collate function to stack dicts of tensors."""
        batch_dicts, target_dicts = zip(*batch)

        collated_batch = {}
        for key in batch_dicts[0]:
            collated_batch[key] = torch.stack([b[key] for b in batch_dicts])

        collated_targets = {}
        for key in target_dicts[0]:
            collated_targets[key] = torch.stack([t[key] for t in target_dicts])

        return collated_batch, collated_targets

    @property
    def feature_scaler(self) -> Optional[dict]:
        """Return the fitted scaler dict (mean/std) for reuse by val/test datasets."""
        return self._feature_scaler

    def get_feature_dims(self) -> Dict[str, int]:
        """Return feature dimensions for model config construction."""
        return {
            "num_static_numeric": self.static_numeric.shape[1],
            "num_historical_numeric": self.historical_numeric.shape[1],
            "num_future_numeric": self.future_numeric.shape[1],
        }


class PanelCrisisDataset(Dataset):
    """Multi-market panel dataset for crisis EWS.

    Loads a panel parquet with a `market_id` column. Each market's time series is
    windowed independently (no cross-market windows). Market identity is provided as
    a static categorical feature for TFT's entity embedding.

    batch_dict keys:
        - static_feats_categorical: (1,) — market ID integer
        - historical_ts_numeric: (encoder_steps, num_hist_feats)
        - future_ts_numeric: (decoder_steps, num_future_feats)

    targets_dict keys:
        - ews_label: (decoder_steps,)
        - regime_label: (encoder_steps + decoder_steps,)
    """

    def __init__(
        self,
        panel_path: str,
        split: str = "train",
        encoder_steps: int = 252,
        decoder_steps: int = 63,
        historical_numeric_cols: Optional[List[str]] = None,
        future_numeric_cols: Optional[List[str]] = None,
        feature_scaler: Optional[dict] = None,
        exclude_crisis_windows: bool = True,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
        test_end: Optional[str] = None,
        embargo_days: int = 63,
        per_market_norm: bool = False,
    ):
        super().__init__()
        self.split = split
        self.per_market_norm = per_market_norm
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.window_size = encoder_steps + decoder_steps
        self._feature_scaler = feature_scaler

        self.historical_numeric_cols = historical_numeric_cols or HISTORICAL_NUMERIC

        df = pd.read_parquet(panel_path)
        df["date"] = pd.to_datetime(df["date"])

        # Temporal split
        if train_end and val_end:
            train_end_ts = pd.Timestamp(train_end)
            val_end_ts = pd.Timestamp(val_end)
            if split == "train":
                df = df[df["date"] <= train_end_ts]
            elif split == "val":
                df = df[(df["date"] > train_end_ts) & (df["date"] <= val_end_ts)]
            else:  # test
                df = df[df["date"] > val_end_ts]
                if test_end:  # bound the test window (walk-forward folds)
                    df = df[df["date"] <= pd.Timestamp(test_end)]

        # Embargo (purge): drop the last `embargo_days` rows per market from
        # train/val so their forward (decoder-horizon) labels don't bleed across
        # the split boundary into the next period (val-based model selection
        # peeking at test). Test is left intact.
        if embargo_days > 0 and split != "test" and len(df):
            df = (df.sort_values(["market_id", "date"])
                    .groupby("market_id", group_keys=False)
                    .apply(lambda g: g.iloc[:-embargo_days] if len(g) > embargo_days else g.iloc[0:0]))

        # Build market ID mapping
        markets = sorted(df["market_id"].unique())
        self.market_to_idx = {m: i for i, m in enumerate(markets)}
        self.num_markets = len(markets)

        # Filter historical columns to those present
        self.historical_numeric_cols = [c for c in self.historical_numeric_cols if c in df.columns]

        # Also auto-detect extra numeric columns
        exclude = {"date", "market_id", "ews_label", "in_crisis", "drawdown",
                    "regime_label", "close", "volume"}
        extra = [c for c in df.columns
                 if c not in exclude
                 and c not in self.historical_numeric_cols
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        self.historical_numeric_cols = self.historical_numeric_cols + extra

        # Build per-market arrays and window indices
        self._build_per_market(df, exclude_crisis_windows)

    @classmethod
    def train_val_split(
        cls,
        panel_path: str,
        train_end: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> tuple:
        """Create train/val datasets with random holdout from the same time period.

        Instead of temporal val (which may lack crises), randomly holds out
        val_ratio of training windows. Both splits share the same crisis distribution.
        """
        # Load full training period
        full_train = cls(panel_path, split="train", train_end=train_end,
                         val_end=train_end, **kwargs)

        # Random split of window indices
        rng = np.random.RandomState(seed)
        n = len(full_train.windows)
        indices = rng.permutation(n)
        val_size = int(n * val_ratio)

        val_indices = set(indices[:val_size].tolist())
        train_indices = [i for i in range(n) if i not in val_indices]

        # Create shallow copies with different window lists
        import copy
        ds_train = copy.copy(full_train)
        ds_val = copy.copy(full_train)

        ds_train.windows = [full_train.windows[i] for i in train_indices]
        ds_val.windows = [full_train.windows[i] for i in sorted(val_indices)]
        ds_val.split = "val"

        return ds_train, ds_val

    def _build_per_market(self, df: pd.DataFrame, exclude_crisis: bool) -> None:
        """Build normalized feature arrays and valid window indices per market."""
        all_market_dfs = {}
        raw_by_market = {}

        # Collect raw features for scaler fitting
        for market in sorted(df["market_id"].unique()):
            mdf = df[df["market_id"] == market].sort_values("date").reset_index(drop=True)
            all_market_dfs[market] = mdf
            raw_by_market[market] = mdf[self.historical_numeric_cols].values.astype(np.float32)

        per_mkt = getattr(self, "per_market_norm", False)
        # Fit scaler on TRAIN only (reused for val/test via feature_scaler).
        if self._feature_scaler is None:
            comb = np.concatenate(list(raw_by_market.values()), axis=0)
            gm = np.nanmean(comb, axis=0); gs = np.nanstd(comb, axis=0); gs[gs < 1e-8] = 1.0
            self._feature_scaler = {"mean": gm, "std": gs}
            if per_mkt:
                # Per-market (RevIN-style) scaler: each entity normalised by its own
                # train statistics, so heterogeneous markets (EM/small-cap/Europe)
                # are made comparable. Global mean/std kept as fallback.
                pm = {}
                for market, raw in raw_by_market.items():
                    m = np.nanmean(raw, axis=0); s = np.nanstd(raw, axis=0); s[s < 1e-8] = 1.0
                    pm[market] = {"mean": m, "std": s}
                self._feature_scaler["per_market"] = pm

        def _scaler_for(market):
            fs = self._feature_scaler
            if "per_market" in fs and market in fs["per_market"]:
                return fs["per_market"][market]["mean"], fs["per_market"][market]["std"]
            return fs["mean"], fs["std"]

        # Build per-market tensors and windows
        self.windows = []  # list of (market_idx, start_idx, market_key)
        self.market_data = {}  # market_key -> {hist, future, ews, regime, in_crisis}

        for market, mdf in all_market_dfs.items():
            n = len(mdf)
            if n < self.window_size:
                continue

            market_idx = self.market_to_idx[market]

            # Normalize features (per-market scaler if enabled, else global)
            mean, std = _scaler_for(market)
            raw = mdf[self.historical_numeric_cols].values.astype(np.float32)
            hist = (raw - mean) / std
            hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

            # Calendar features
            idx = np.arange(n, dtype=np.float32)
            future = np.stack([
                np.sin(idx * 2 * np.pi / 252),
                np.cos(idx * 2 * np.pi / 252),
            ], axis=1)

            # Labels
            ews = mdf["ews_label"].values.astype(np.float32) if "ews_label" in mdf.columns else np.zeros(n, dtype=np.float32)
            regime = mdf["regime_label"].values.astype(np.int64) if "regime_label" in mdf.columns else np.zeros(n, dtype=np.int64)
            in_crisis = mdf["in_crisis"].values.astype(bool) if "in_crisis" in mdf.columns else np.zeros(n, dtype=bool)
            drawdown = (np.nan_to_num(mdf["drawdown"].values.astype(np.float32), nan=0.0)
                        if "drawdown" in mdf.columns else np.zeros(n, dtype=np.float32))

            self.market_data[market] = {
                "hist": hist,
                "future": future,
                "ews": ews,
                "regime": regime,
                "in_crisis": in_crisis,
                "drawdown": drawdown,
                "market_idx": market_idx,
            }

            # Build valid window indices for this market
            for i in range(n - self.window_size + 1):
                enc_end_idx = i + self.encoder_steps - 1
                if exclude_crisis and in_crisis[enc_end_idx]:
                    continue
                self.windows.append((market, i))

        self._num_historical = len(self.historical_numeric_cols)
        self._num_future = 2
        self._num_static = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        market, start = self.windows[idx]
        md = self.market_data[market]
        enc_end = start + self.encoder_steps
        dec_end = enc_end + self.decoder_steps

        batch = {
            "static_feats_categorical": torch.tensor([md["market_idx"]], dtype=torch.long),
            "historical_ts_numeric": torch.tensor(md["hist"][start:enc_end], dtype=torch.float32),
            "future_ts_numeric": torch.tensor(md["future"][enc_end:dec_end], dtype=torch.float32),
        }

        targets = {
            "ews_label": torch.tensor(md["ews"][enc_end:dec_end], dtype=torch.float32),
            "regime_label": torch.tensor(md["regime"][start:dec_end], dtype=torch.long),
            "drawdown_target": torch.tensor(md["drawdown"][enc_end:dec_end], dtype=torch.float32),
        }

        return batch, targets

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_dicts, target_dicts = zip(*batch)
        collated_batch = {key: torch.stack([b[key] for b in batch_dicts]) for key in batch_dicts[0]}
        collated_targets = {key: torch.stack([t[key] for t in target_dicts]) for key in target_dicts[0]}
        return collated_batch, collated_targets

    @property
    def feature_scaler(self) -> Optional[dict]:
        return self._feature_scaler

    def get_feature_dims(self) -> Dict[str, int]:
        return {
            "num_static_numeric": 0,
            "num_static_categorical": 1,
            "static_categorical_cardinalities": [self.num_markets],
            "num_historical_numeric": self._num_historical,
            "num_future_numeric": self._num_future,
        }


