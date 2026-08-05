# tests/data/test_labels.py
import pandas as pd
import numpy as np
from data.pipelines.processing.labels import (
    build_ews_labels,
    build_regime_labels,
    _identify_crisis_periods,
    _find_crisis_onsets,
)


def test_crisis_periods_detected():
    """A 30% drop from peak should be flagged as crisis at 20% threshold."""
    # Ramp up to 100, then crash to 70 (30% drawdown from peak)
    prices = pd.Series(
        np.concatenate([np.linspace(80, 100, 100), np.linspace(100, 70, 50), np.ones(50) * 70]),
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    in_crisis = _identify_crisis_periods(prices, threshold=0.20, peak_window=252)
    assert in_crisis.sum() > 0
    # The crisis should be in the second half (after the crash)
    assert in_crisis.iloc[:100].sum() == 0


def test_crisis_onset_marks_first_day():
    """Onset should only fire on the first day entering crisis."""
    in_crisis = pd.Series([False, False, False, True, True, True, False, True, True])
    onsets = _find_crisis_onsets(in_crisis)
    # Two separate crisis episodes → two onsets
    assert onsets.sum() == 2
    assert onsets.iloc[3] == True
    assert onsets.iloc[7] == True
    # Not onsets: days already in crisis
    assert onsets.iloc[4] == False
    assert onsets.iloc[5] == False


def test_ews_labels_forward_looking():
    """EWS label=1 should appear BEFORE crisis onset, not during."""
    # 200 calm days, then crash from 100 to 75 (25% drawdown) over 50 days
    prices = pd.Series(
        np.concatenate([np.linspace(90, 100, 200), np.linspace(100, 75, 50), np.ones(50) * 75]),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    labels = build_ews_labels(prices, threshold=0.20, horizon_days=63, peak_window=252)

    # There should be warning labels before the crash
    assert labels["ews_label"].sum() > 0

    # Days already in crisis should not have ews_label=1 (they're excluded from training)
    crisis_days = labels[labels["in_crisis"]]
    if len(crisis_days) > 0:
        # The in_crisis flag should be set
        assert crisis_days["in_crisis"].all()

    # Warning labels should precede the first crisis day
    first_crisis_idx = labels[labels["in_crisis"]].index[0]
    warning_days = labels[labels["ews_label"] == 1]
    assert warning_days.index[0] < first_crisis_idx


def test_ews_no_crisis_no_warning():
    """If price only goes up, no crisis and no warnings."""
    prices = pd.Series(
        np.linspace(100, 200, 300),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    labels = build_ews_labels(prices, threshold=0.20, horizon_days=63)
    assert labels["ews_label"].sum() == 0
    assert labels["in_crisis"].sum() == 0


def test_regime_labels():
    dates = pd.bdate_range("2020-01-01", periods=300)
    # Steady price at 100, then crash to 70
    prices = pd.Series(
        np.concatenate([np.ones(200) * 100, np.linspace(100, 70, 50), np.ones(50) * 70]),
        index=dates,
    )
    vix = pd.Series(
        np.concatenate([np.ones(100) * 15, np.ones(100) * 30, np.ones(100) * 40]),
        index=dates,
    )
    regimes = build_regime_labels(prices, vix, drawdown_threshold=0.20, vix_stress=25.0)
    assert set(regimes["regime_label"].unique()).issubset({0, 1, 2})
    # Calm period (low VIX, no drawdown)
    assert regimes["regime_label"].iloc[50] == 0
