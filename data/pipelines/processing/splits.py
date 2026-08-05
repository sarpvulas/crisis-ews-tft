# data/pipelines/processing/splits.py
"""Temporal splitting strategies for crisis EWS evaluation.

Implements walk-forward CV anchored on crisis episodes,
following Bluwstein et al. (2023) and BIS WP 878 methodology.
"""
import logging
from typing import List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe temporally with no leakage."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df[date_col] <= train_end_ts].copy()
    val = df[(df[date_col] > train_end_ts) & (df[date_col] <= val_end_ts)].copy()
    test = df[df[date_col] > val_end_ts].copy()

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# Crisis episodes for S&P 500 (onsets confirmed by the dd10 rolling-peak labeler).
# dot-com stays as the bootstrap (index 0): it cannot be a test fold because the
# data starts in 2000 (no prior crisis to train on). Each later episode becomes a
# walk-forward fold. Eurozone-2011 / China-2015 / 2018-Q4 added to raise the number
# of independent crisis episodes (the real power lever for significance testing).
CRISIS_EPISODES = [
    {"name": "dot-com",     "start": "2001-03-01", "end": "2003-05-01"},
    {"name": "gfc",         "start": "2008-07-01", "end": "2009-10-01"},
    {"name": "eurozone",    "start": "2011-05-01", "end": "2012-06-01"},
    {"name": "china2015",   "start": "2015-08-01", "end": "2016-02-01"},
    {"name": "selloff2018", "start": "2018-10-01", "end": "2018-12-31"},
    {"name": "covid",       "start": "2020-03-01", "end": "2020-05-01"},
    {"name": "bear2022",    "start": "2022-06-01", "end": "2023-02-01"},
]


def walk_forward_crisis_splits(
    df: pd.DataFrame,
    date_col: str,
    crisis_episodes: List[dict] = None,
    val_fraction: float = 0.3,
    min_train_years: int = 5,
    embargo_days: int = 63,
) -> List[dict]:
    """Walk-forward CV anchored on crisis episodes.

    For each crisis episode (starting from the 2nd), creates a fold:
      - Train: all data before the gap period
      - Val: the gap between previous crisis end and test pre-crisis window
      - Test: 1 year before crisis start through crisis end (captures
              both the pre-crisis warning period and the crisis itself)

    This ensures:
      1. Every test fold contains a genuine crisis to predict
      2. Training always includes at least one prior crisis
      3. No future information leaks into training
      4. Val set covers the calm period (for calibrating false alarm rate)

    Args:
        df: DataFrame with date column and features.
        date_col: Name of the date column.
        crisis_episodes: List of dicts with 'name', 'start', 'end' keys.
        val_fraction: Fraction of the gap between train and test to use as val.
        min_train_years: Minimum years of training data.

    Returns:
        List of fold dicts with 'name', 'train', 'val', 'test' DataFrames
        and metadata about the split dates and crisis counts.
    """
    if crisis_episodes is None:
        crisis_episodes = CRISIS_EPISODES

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    data_start = df[date_col].min()

    # Minimum rows needed for a dataset window (encoder + decoder)
    min_window = 315  # 252 + 63

    folds = []

    for i in range(1, len(crisis_episodes)):
        test_crisis = crisis_episodes[i]
        prev_crisis = crisis_episodes[i - 1]

        test_crisis_start = pd.Timestamp(test_crisis["start"])
        test_crisis_end = pd.Timestamp(test_crisis["end"])

        # Test: 2 years before crisis start through 6 months after crisis end
        # Needs to be wide enough to form sliding windows (>315 rows)
        test_start = test_crisis_start - pd.DateOffset(years=2)
        test_end = test_crisis_end + pd.DateOffset(months=6)
        # Clamp to data range
        test_end = min(test_end, df[date_col].max())

        # Training must end before test starts (no overlap)
        train_end = test_start

        # Check minimum training requirement
        train_duration = (train_end - data_start).days / 365.25
        if train_duration < min_train_years:
            logger.info(f"Skipping {test_crisis['name']}: only {train_duration:.1f}y of training data")
            continue

        # Val: last 2 years of training period (for early stopping)
        val_start = train_end - pd.DateOffset(years=2)

        # Build splits
        train = df[df[date_col] < val_start].copy()
        val = df[(df[date_col] >= val_start) & (df[date_col] < train_end)].copy()
        test = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)].copy()

        # ---- Embargo (purge) ----
        # ews_label looks `decoder_steps` (~63) trading days AHEAD, so the last
        # `embargo_days` rows of train/val carry labels that bleed into the next
        # split (val/test). Dropping them removes the only forward-label leak —
        # critically, it stops val-based model selection from peeking at the test
        # crisis. The encoder lookback reaching back across a boundary is NOT a
        # leak (using history to predict the future is the point), so test is
        # left intact. (López de Prado purging.)
        if embargo_days > 0:
            train = train.iloc[:-embargo_days] if len(train) > embargo_days else train.iloc[:0]
            val = val.iloc[:-embargo_days] if len(val) > embargo_days else val.iloc[:0]

        # Skip if any split is too small for windowing
        if len(train) < min_window or len(val) < min_window or len(test) < min_window:
            logger.info(
                f"Skipping {test_crisis['name']}: too few rows "
                f"(train={len(train)}, val={len(val)}, test={len(test)}, need>{min_window})"
            )
            continue

        # Count crises in each split
        def _count_ews(split_df):
            if "ews_label" in split_df.columns:
                return int(split_df["ews_label"].sum())
            return 0

        fold = {
            "name": test_crisis["name"],
            "train": train,
            "val": val,
            "test": test,
            "meta": {
                "train_dates": f"{train[date_col].min().date()} to {train[date_col].max().date()}",
                "val_dates": f"{val[date_col].min().date()} to {val[date_col].max().date()}",
                "test_dates": f"{test[date_col].min().date()} to {test[date_col].max().date()}",
                "train_n": len(train),
                "val_n": len(val),
                "test_n": len(test),
                "train_ews": _count_ews(train),
                "val_ews": _count_ews(val),
                "test_ews": _count_ews(test),
                "embargo_days": embargo_days,
            },
        }
        folds.append(fold)

        logger.info(
            f"Fold {len(folds)} ({test_crisis['name']}): "
            f"train={fold['meta']['train_dates']} ({len(train)} rows, {fold['meta']['train_ews']} warnings), "
            f"val={fold['meta']['val_dates']} ({len(val)} rows, {fold['meta']['val_ews']} warnings), "
            f"test={fold['meta']['test_dates']} ({len(test)} rows, {fold['meta']['test_ews']} warnings)"
        )

    return folds
