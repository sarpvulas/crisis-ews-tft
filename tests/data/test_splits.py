# tests/data/test_splits.py
import pandas as pd
import numpy as np
from data.pipelines.processing.splits import temporal_split, rolling_window_splits

def test_temporal_split_no_leakage():
    dates = pd.date_range("2000-01-01", "2020-12-01", freq="MS")
    df = pd.DataFrame({"date": dates, "value": np.random.randn(len(dates))})

    train, val, test = temporal_split(
        df, date_col="date",
        train_end="2005-12-31", val_end="2012-12-31",
    )
    assert train["date"].max() <= pd.Timestamp("2005-12-31")
    assert val["date"].min() >= pd.Timestamp("2006-01-01")
    assert val["date"].max() <= pd.Timestamp("2012-12-31")
    assert test["date"].min() >= pd.Timestamp("2013-01-01")
    assert len(train) + len(val) + len(test) == len(df)

def test_rolling_window_splits():
    dates = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    df = pd.DataFrame({"date": dates, "value": np.random.randn(len(dates))})
    splits = rolling_window_splits(
        df, date_col="date",
        train_years=15, val_years=3, test_years=5, step_years=3,
    )
    assert len(splits) >= 4  # Should produce multiple folds
    for train, val, test in splits:
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()
