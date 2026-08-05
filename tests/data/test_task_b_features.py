# tests/data/test_task_b_features.py
import pandas as pd
import numpy as np
from data.pipelines.processing.task_b_features import (
    compute_realized_vol,
    compute_drawdown_from_peak,
    compute_hurst_exponent,
    resample_to_weekly,
    build_task_b_features,
)

def test_realized_vol_correct():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008] * 20)
    vol_5 = compute_realized_vol(returns, window=5)
    assert len(vol_5) == len(returns)
    assert vol_5.iloc[4:].notna().all()
    assert vol_5.iloc[4] > 0

def test_drawdown_from_peak():
    prices = pd.Series([100, 105, 103, 95, 98, 110, 100])
    dd = compute_drawdown_from_peak(prices)
    assert dd.iloc[0] == 0.0  # First price is the peak
    assert dd.iloc[3] < 0     # 95 < peak of 105
    assert dd.min() < 0

def test_weekly_resample():
    dates = pd.bdate_range("2020-01-01", periods=20)
    df = pd.DataFrame({
        "close": np.random.randn(20).cumsum() + 100,
        "volume": np.random.randint(1e6, 1e7, 20),
        "vix": np.random.randn(20) + 20,
    }, index=dates)
    weekly = resample_to_weekly(df)
    assert len(weekly) >= 4  # 20 business days ~ 4-5 weeks depending on alignment
    assert "close" in weekly.columns
    assert "volume" in weekly.columns
