# data/pipelines/processing/crisis_labels.py
"""Drawdown-based crisis labeling with hysteresis and post-crisis buffer.

This is a stricter alternative to ``labels.build_ews_labels``:
  * peak reference is configurable via ``peak_window``: None = running
    (cumulative) all-time peak (default, legacy); an int = trailing rolling
    peak over that many days (e.g. 252 = 52-week-high drawdown), which avoids
    the post-crash multi-year "in drawdown" artifact
  * a crisis starts only on the FIRST crossing below ``entry_threshold``
  * a crisis remains active until DD stays above ``exit_threshold`` for
    ``recovery_days`` consecutive days (hysteresis -- avoids on/off flicker)
  * a ``post_crisis_buffer`` of trading days is also excluded from training
  * the ML target ``y`` is a nullable Int (0/1/NA), with NA on rows that
    fall inside an active crisis or buffer, or where t+horizon overruns the
    end of the series.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_running_drawdown(prices: pd.Series, peak_window: int | None = None) -> pd.Series:
    """Causal drawdown from the peak.

    If ``peak_window`` is None, the reference is the running all-time high
    (``prices.cummax()``). If an int, it is a trailing rolling maximum over
    ``peak_window`` trading days (e.g. 252 = 52-week-high drawdown), matching
    the convention in ``labels.build_ews_labels`` / ``build_regime_labels``.
    Both are causal (past/current prices only), so safe to use as a feature.

    The rolling-peak form avoids the all-time-high artifact whereby a series
    stays "in drawdown" for years after a deep crash (e.g. SPX 2002-2007),
    which spawns phantom onsets during recoveries.
    """
    if peak_window is None:
        peak = prices.cummax()
    else:
        peak = prices.rolling(window=peak_window, min_periods=1).max()
    return (prices - peak) / peak


def identify_crisis_episodes(
    prices: pd.Series,
    *,
    entry_threshold: float = -0.20,
    exit_threshold: float = -0.10,
    recovery_days: int = 30,
    recovery_mode: str = "peak",
    rebound_threshold: float = 0.20,
    peak_window: int | None = None,
) -> List[Tuple[int, int]]:
    """Identify crisis episodes as (start_idx, last_active_idx) pairs.

    Start is the first index where DD crosses below ``entry_threshold``
    (DD_t <= entry AND DD_{t-1} > entry).

    The episode ends after a ``recovery_days``-long recovery streak. Two ways
    to define "recovered", controlled by ``recovery_mode``:

    * ``"peak"`` (default): drawdown from the running all-time peak climbs back
      above ``exit_threshold`` (e.g. -0.10 = within 10% of the old high). After
      a deep crash this requires a near-full recovery, so episodes stay active
      for years (dot-com -> 2006, GFC -> 2012).
    * ``"trough"``: price has rebounded at least ``rebound_threshold`` above the
      lowest price seen since the crisis started (e.g. +0.20 = +20% off the low).
      Declares the acute phase over once recovery is underway, regardless of how
      far below the old high we still are. Frees the long grind back to highs.

    ``last_active_idx`` is the final day of the recovery streak; if it never
    completes before the series ends, the episode runs to the last index.
    """
    if recovery_mode not in ("peak", "trough"):
        raise ValueError(f"recovery_mode must be 'peak' or 'trough', got {recovery_mode!r}")
    dd = compute_running_drawdown(prices, peak_window=peak_window).to_numpy()
    p = prices.to_numpy()
    n = len(dd)
    episodes: List[Tuple[int, int]] = []

    i = 0
    while i < n:
        crossed = dd[i] <= entry_threshold and (i == 0 or dd[i - 1] > entry_threshold)
        if not crossed:
            i += 1
            continue

        start = i
        streak = 0
        last_active = n - 1  # default: crisis runs to end of data
        trough = p[start]
        j = start + 1
        while j < n:
            if recovery_mode == "trough":
                if p[j] < trough:
                    trough = p[j]
                recovered = trough > 0 and (p[j] - trough) / trough >= rebound_threshold
            else:
                recovered = dd[j] > exit_threshold
            if recovered:
                streak += 1
                if streak >= recovery_days:
                    last_active = j
                    break
            else:
                streak = 0
            j += 1

        episodes.append((start, last_active))
        i = last_active + 1

    return episodes


def build_crisis_labels(
    prices: pd.Series,
    *,
    horizon: int = 60,
    entry_threshold: float = -0.20,
    exit_threshold: float = -0.10,
    recovery_days: int = 30,
    post_crisis_buffer: int = 30,
    recovery_mode: str = "peak",
    rebound_threshold: float = 0.20,
    peak_window: int | None = None,
) -> pd.DataFrame:
    """Build the crisis-period labeling described in the spec.

    Returns a DataFrame indexed positionally (same length as ``prices``) with
    columns: ``date, price, peak, drawdown, crisis_start, crisis_active,
    in_buffer, y``.  ``y`` is a nullable ``Int64`` (0/1/NA).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n = len(prices)
    if peak_window is None:
        peak = prices.cummax()
    else:
        peak = prices.rolling(window=peak_window, min_periods=1).max()
    drawdown = (prices - peak) / peak

    episodes = identify_crisis_episodes(
        prices,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        recovery_days=recovery_days,
        recovery_mode=recovery_mode,
        rebound_threshold=rebound_threshold,
        peak_window=peak_window,
    )

    crisis_start = np.zeros(n, dtype=bool)
    crisis_active = np.zeros(n, dtype=bool)
    in_buffer = np.zeros(n, dtype=bool)

    for start, last_active in episodes:
        crisis_start[start] = True
        crisis_active[start : last_active + 1] = True
        buf_start = last_active + 1
        buf_end = min(last_active + post_crisis_buffer, n - 1)
        if buf_start <= buf_end:
            in_buffer[buf_start : buf_end + 1] = True

    # cumulative count of crisis_starts -- O(n) lookahead.
    # starts_cumsum[k] = number of starts in indices [0, k-1].
    starts_cumsum = np.concatenate([[0], np.cumsum(crisis_start.astype(np.int64))])

    y = np.full(n, fill_value=pd.NA, dtype=object)
    for t in range(n):
        if crisis_active[t] or in_buffer[t]:
            continue
        future_end = t + horizon
        if future_end >= n:
            continue  # insufficient lookahead -- leave as NA
        # Count starts in [t+1, t+horizon] inclusive.
        count = starts_cumsum[future_end + 1] - starts_cumsum[t + 1]
        y[t] = 1 if count > 0 else 0

    labels = pd.DataFrame(
        {
            "date": prices.index,
            "price": prices.to_numpy(),
            "peak": peak.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "crisis_start": crisis_start,
            "crisis_active": crisis_active,
            "in_buffer": in_buffer,
            "y": pd.array(y, dtype="Int64"),
        }
    )

    n_starts = int(crisis_start.sum())
    n_active = int(crisis_active.sum())
    n_buffer = int(in_buffer.sum())
    n_ones = int((labels["y"] == 1).sum())
    n_zeros = int((labels["y"] == 0).sum())
    n_nan = int(labels["y"].isna().sum())
    logger.info(
        "crisis_labels: %d episode(s), %d active days, %d buffer days; "
        "y: %d ones, %d zeros, %d NaN",
        n_starts, n_active, n_buffer, n_ones, n_zeros, n_nan,
    )
    return labels
