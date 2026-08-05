# tests/data/test_crisis_labels.py
"""Tests for the drawdown-based crisis labeling system.

Spec recap:
  - Crisis starts when DD first crosses below -20% (only the first crossing).
  - Crisis remains active until DD stays above -10% for 30 consecutive days.
  - After the crisis ends, a 30-day post-crisis buffer is also excluded.
  - Y_t = 1 if a NEW crisis start falls in [t+1, t+h], 0 otherwise,
    NaN while in active crisis or buffer (or when t+h exceeds the series).
"""
import numpy as np
import pandas as pd
import pytest

from data.pipelines.processing.crisis_labels import (
    build_crisis_labels,
    identify_crisis_episodes,
    compute_running_drawdown,
)


def _worked_example_prices(n_days: int = 300) -> pd.Series:
    """Construct the worked-example price path from the spec.

    Index 0..98   (days 1..99):    price = 100         -> DD = 0
    Index 99      (day 100):        price = 79          -> DD = -21% (crisis start)
    Index 99..149 (days 100..150):  price = 79          -> DD = -21%
    Index 150..179(days 151..180):  price = 91          -> DD = -9%  (30-day recovery)
    Index 180..209(days 181..210):  price = 91          -> buffer
    Index 210..   (days 211..):     price = 91          -> normal again
    """
    price = np.empty(n_days)
    price[:99] = 100.0
    price[99:150] = 79.0
    price[150:] = 91.0
    return pd.Series(price, index=pd.bdate_range("2020-01-01", periods=n_days))


def test_worked_example_from_spec():
    """The exact example given in the spec must match label-for-label."""
    prices = _worked_example_prices(300)
    df = build_crisis_labels(
        prices,
        horizon=30,
        entry_threshold=-0.20,
        exit_threshold=-0.10,
        recovery_days=30,
        post_crisis_buffer=30,
    )

    y = df["y"]

    # Day 1..69 (idx 0..68): Y = 0
    assert (y.iloc[0:69] == 0).all(), "Days 1-69 should all be Y=0"
    # Day 70..99 (idx 69..98): Y = 1
    assert (y.iloc[69:99] == 1).all(), "Days 70-99 should all be Y=1"
    # Day 100..180 (idx 99..179): Y = NaN (active crisis)
    assert y.iloc[99:180].isna().all(), "Days 100-180 should be NaN (active crisis)"
    # Day 181..210 (idx 180..209): Y = NaN (post-crisis buffer)
    assert y.iloc[180:210].isna().all(), "Days 181-210 should be NaN (buffer)"
    # Day 211..270 (idx 210..269): Y = 0 (no new crisis in lookahead)
    assert (y.iloc[210:270] == 0).all(), "Days 211-270 should be Y=0"
    # Day 271..300 (idx 270..299): Y = NaN (insufficient lookahead for h=30)
    assert y.iloc[270:].isna().all(), "Tail rows with t+h>=n should be NaN"


def test_crisis_start_fires_only_on_first_crossing():
    """A multi-day spell below -20% must produce exactly one crisis_start."""
    prices = _worked_example_prices(300)
    df = build_crisis_labels(prices, horizon=30)
    # crisis_start is True only on day 100 (idx 99)
    assert df["crisis_start"].sum() == 1
    assert df["crisis_start"].iloc[99]


def test_hysteresis_prevents_spurious_recovery():
    """A brief dip back below -10% during recovery must keep crisis active."""
    n = 400
    price = np.empty(n)
    price[:99] = 100.0
    price[99:150] = 79.0       # crisis (DD = -21%)
    price[150:170] = 91.0      # 20 days above -10% (DD = -9%)
    price[170:180] = 85.0      # dip back below -10% (DD = -15%) — resets streak
    price[180:230] = 91.0      # 50 more days above -10% — recovery completes at idx 209
    price[230:] = 91.0
    prices = pd.Series(price, index=pd.bdate_range("2020-01-01", periods=n))

    df = build_crisis_labels(
        prices, horizon=30, recovery_days=30, post_crisis_buffer=30,
    )
    # Only one crisis start
    assert df["crisis_start"].sum() == 1
    # Crisis must still be active during the failed-recovery dip (idx 170-179)
    assert df["crisis_active"].iloc[175]
    # Crisis ends after 30 consecutive days above -10% starting idx 180 -> last active idx 209
    assert df["crisis_active"].iloc[209]
    assert not df["crisis_active"].iloc[210]
    # Buffer: idx 210..239
    assert df["in_buffer"].iloc[210]
    assert df["in_buffer"].iloc[239]
    assert not df["in_buffer"].iloc[240]


def test_no_crisis_yields_all_zero_labels():
    """Monotonically rising prices: no NaN-from-crisis, no warning labels."""
    prices = pd.Series(
        np.linspace(100, 200, 300),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    df = build_crisis_labels(prices, horizon=30)
    assert df["crisis_start"].sum() == 0
    assert df["crisis_active"].sum() == 0
    assert df["in_buffer"].sum() == 0
    # Tail rows are still NaN (insufficient lookahead), but body should be 0
    assert (df["y"].iloc[:-30] == 0).all()


def test_drawdown_is_causal():
    """compute_running_drawdown at index t must depend only on prices[:t+1]."""
    prices = pd.Series([100, 110, 90, 80, 95, 120], dtype=float)
    dd = compute_running_drawdown(prices)
    # Peaks: 100, 110, 110, 110, 110, 120
    # DD:    0,   0,  -20/110, -30/110, -15/110, 0
    expected = np.array([0.0, 0.0, -20 / 110, -30 / 110, -15 / 110, 0.0])
    np.testing.assert_allclose(dd.values, expected, atol=1e-12)


def test_crisis_without_recovery_runs_to_end_of_series():
    """If recovery never completes before data ends, crisis remains active."""
    n = 200
    price = np.empty(n)
    price[:99] = 100.0
    price[99:] = 70.0  # never recovers above -10%
    prices = pd.Series(price, index=pd.bdate_range("2020-01-01", periods=n))

    df = build_crisis_labels(prices, horizon=30, post_crisis_buffer=30)
    assert df["crisis_start"].sum() == 1
    assert df["crisis_active"].iloc[99:].all()
    assert df["in_buffer"].sum() == 0  # never reached buffer phase
    # Y for the run-up should still be valid (1 near the crisis, 0 earlier)
    assert (df["y"].iloc[:69] == 0).all()
    assert (df["y"].iloc[69:99] == 1).all()
    assert df["y"].iloc[99:].isna().all()


def test_identify_episodes_returns_start_and_end_indices():
    """The lower-level episode helper returns (start, last_active) pairs."""
    prices = _worked_example_prices(300)
    episodes = identify_crisis_episodes(
        prices,
        entry_threshold=-0.20,
        exit_threshold=-0.10,
        recovery_days=30,
    )
    assert episodes == [(99, 179)]
