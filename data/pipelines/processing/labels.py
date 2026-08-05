# data/pipelines/processing/labels.py
"""EWS labeling following Dichtl, Drobetz & Otto (2023) and BIS methodology.

Crisis definition: drawdown from 252-day rolling peak >= 20% (bear market).
EWS label: y_t = 1 if a crisis ONSET occurs within [t+1, t+h], else 0.
Observations already in crisis are excluded from training (flagged via in_crisis).
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _identify_crisis_periods(
    prices: pd.Series,
    threshold: float = 0.20,
    peak_window: int = 252,
) -> pd.Series:
    """Identify crisis periods as drawdown from rolling peak >= threshold.

    Returns a boolean Series: True on days that are in crisis.
    """
    peak = prices.rolling(window=peak_window, min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.abs() >= threshold


def _find_crisis_onsets(in_crisis: pd.Series) -> pd.Series:
    """Find crisis onset dates: first day of each new crisis episode.

    An onset is a day where in_crisis transitions from False to True.
    """
    shifted = in_crisis.shift(1, fill_value=False)
    return in_crisis & ~shifted


def build_ews_labels(
    prices: pd.Series,
    threshold: float = 0.20,
    horizon_days: int = 63,
    peak_window: int = 252,
) -> pd.DataFrame:
    """Build Early Warning System labels.

    For each time t:
      1. Crisis = drawdown from 252-day rolling peak >= 20%
      2. Crisis onset = first day entering a crisis episode
      3. ews_label = 1 if any crisis onset in [t+1, t+horizon_days], else 0
      4. in_crisis = True if t is already in a crisis period (excluded from training)

    Args:
        prices: Daily price series (e.g., S&P 500 close).
        threshold: Drawdown threshold for crisis definition (default 0.20 = 20%).
        horizon_days: Forward warning horizon in trading days (default 63 ~ 3 months).
        peak_window: Rolling window for peak calculation (default 252 ~ 1 year).

    Returns:
        DataFrame with columns: date, ews_label, in_crisis, drawdown.
    """
    in_crisis = _identify_crisis_periods(prices, threshold, peak_window)
    onsets = _find_crisis_onsets(in_crisis)

    # For each day, check if any onset occurs in the next horizon_days
    # Use rolling sum on reversed onset series for efficiency
    onset_int = onsets.astype(int).values
    n = len(onset_int)
    ews = np.zeros(n, dtype=int)

    # Forward-looking: any onset in [t+1, t+horizon_days]
    # Compute reverse cumsum with a sliding window
    onset_cumsum = np.cumsum(onset_int)
    for i in range(n):
        future_start = i + 1
        future_end = min(i + horizon_days, n - 1)
        if future_start > future_end:
            continue
        onsets_in_window = onset_cumsum[future_end] - (onset_cumsum[future_start - 1] if future_start > 0 else 0)
        if onsets_in_window > 0:
            ews[i] = 1

    peak = prices.rolling(window=peak_window, min_periods=1).max()
    drawdown = (prices - peak) / peak

    labels = pd.DataFrame({
        "date": prices.index,
        "ews_label": ews,
        "in_crisis": in_crisis.values,
        "drawdown": drawdown.values,
    })

    n_crisis_days = in_crisis.sum()
    n_onsets = onsets.sum()
    n_warning = (ews == 1).sum()
    logger.info(
        f"EWS labels: {n_onsets} crisis onsets, {n_crisis_days} crisis days, "
        f"{n_warning} warning days (label=1), {n - n_warning - n_crisis_days} calm days"
    )

    return labels


def build_regime_labels(
    prices: pd.Series,
    vix: pd.Series,
    drawdown_threshold: float = 0.20,
    vix_stress: float = 25.0,
    peak_window: int = 252,
) -> pd.DataFrame:
    """Build regime labels: calm=0, stress=1, crisis=2.

    Crisis: drawdown from rolling 252-day peak >= 20%.
    Stress: VIX >= 25 (but not in crisis).
    Calm: everything else.
    """
    peak = prices.rolling(window=peak_window, min_periods=1).max()
    drawdown = (prices - peak) / peak

    vix_aligned = vix.reindex(prices.index, method="ffill")
    dd_aligned = drawdown.reindex(vix_aligned.index)

    regimes = pd.DataFrame({"date": vix_aligned.index, "regime_label": 0})

    # Crisis = in drawdown >= threshold
    crisis_mask = dd_aligned.abs().values >= drawdown_threshold
    regimes.loc[crisis_mask, "regime_label"] = 2

    # Stress = elevated VIX (but not already crisis)
    stress_mask = (vix_aligned.values >= vix_stress) & (~crisis_mask)
    regimes.loc[stress_mask, "regime_label"] = 1

    return regimes
