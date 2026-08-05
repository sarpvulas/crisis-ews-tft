import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from analysis.graphs.data_exploration import (
    de1_crisis_timeline,
    de2_correlation_heatmap,
    de3_class_imbalance,
    de4_event_study,
    de5_rolling_statistics,
    de6_pca_tsne,
)


@pytest.fixture
def sample_crisis_db():
    return pd.DataFrame({
        "country": ["USA", "USA", "UK", "Germany"],
        "start": pd.to_datetime(["2007-12-01", "2020-03-01", "2008-01-01", "2008-06-01"]),
        "end": pd.to_datetime(["2009-06-01", "2020-09-01", "2009-03-01", "2009-09-01"]),
    })


@pytest.fixture
def sample_crash_events():
    return pd.DataFrame({
        "name": ["GFC", "COVID"],
        "start": pd.to_datetime(["2008-09-01", "2020-02-15"]),
        "end": pd.to_datetime(["2009-03-01", "2020-04-15"]),
    })


@pytest.fixture
def sample_features():
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=200, freq="ME")
    return pd.DataFrame({
        "credit_gap": np.random.randn(200),
        "yield_spread": np.random.randn(200),
        "vix": np.abs(np.random.randn(200)) * 10 + 15,
        "gdp_growth": np.random.randn(200) * 2 + 2,
    }, index=dates)


@pytest.fixture
def sample_regime_labels():
    np.random.seed(42)
    return np.random.choice([0, 1, 2], size=200, p=[0.7, 0.2, 0.1])


def test_de1_returns_figure(sample_crisis_db, sample_crash_events):
    fig = de1_crisis_timeline(sample_crisis_db, sample_crash_events)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de1_without_crash_events(sample_crisis_db):
    fig = de1_crisis_timeline(sample_crisis_db)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de2_returns_figure(sample_features, sample_regime_labels):
    fig = de2_correlation_heatmap(sample_features, sample_regime_labels)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de3_returns_figure(sample_regime_labels):
    fig = de3_class_imbalance(sample_regime_labels)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de4_returns_figure(sample_features):
    crisis_dates = [sample_features.index[50], sample_features.index[120]]
    fig = de4_event_study(sample_features, crisis_dates, window=10)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de5_returns_figure(sample_features):
    crisis_periods = [
        (sample_features.index[40], sample_features.index[60]),
        (sample_features.index[130], sample_features.index[150]),
    ]
    fig = de5_rolling_statistics(sample_features, crisis_periods)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_de6_returns_figure(sample_features, sample_regime_labels):
    fig = de6_pca_tsne(sample_features, sample_regime_labels)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)
