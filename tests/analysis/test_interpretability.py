import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from analysis.graphs.interpretability import (
    int1_variable_importance_over_time,
    int2_attention_heatmap,
    int3_regime_transition_diagram,
    int4_regime_timeline,
    int5_attention_calm_vs_crisis,
    int6_feature_attribution_per_regime,
)


@pytest.fixture
def feature_names():
    return ["credit_gap", "yield_spread", "vix", "gdp_growth",
            "inflation", "npl_ratio", "leverage", "m2_growth"]


@pytest.fixture
def vsn_weights_over_time():
    np.random.seed(42)
    # (n_timesteps, n_features)
    return np.abs(np.random.randn(100, 8))


@pytest.fixture
def dates():
    import pandas as pd
    return pd.date_range("2000-01-01", periods=100, freq="ME").values


@pytest.fixture
def attention_3d():
    np.random.seed(42)
    # (num_heads, decoder_steps, total_steps)
    attn = np.abs(np.random.randn(4, 12, 72))
    # Normalize to valid attention weights
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn


@pytest.fixture
def transition_matrix():
    return np.array([
        [0.95, 0.04, 0.01],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])


@pytest.fixture
def regime_probs():
    np.random.seed(42)
    raw = np.abs(np.random.randn(100, 3))
    return raw / raw.sum(axis=1, keepdims=True)


def test_int1_returns_figure(vsn_weights_over_time, dates, feature_names):
    fig = int1_variable_importance_over_time(vsn_weights_over_time, dates, feature_names)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int2_returns_figure_3d(attention_3d):
    fig = int2_attention_heatmap(attention_3d, crisis_name="GFC 2008")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int2_returns_figure_2d(attention_3d):
    attn_2d = attention_3d.mean(axis=0)
    fig = int2_attention_heatmap(attn_2d, crisis_name="COVID 2020")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int3_returns_figure(transition_matrix):
    fig = int3_regime_transition_diagram(transition_matrix)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int4_returns_figure(regime_probs, dates):
    import pandas as pd
    crisis_dates = [dates[30], dates[70]]
    crisis_names = ["Event A", "Event B"]
    fig = int4_regime_timeline(regime_probs, dates, crisis_dates, crisis_names)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int5_returns_figure():
    np.random.seed(42)
    attn_calm = np.abs(np.random.randn(12, 72))
    attn_calm = attn_calm / attn_calm.sum(axis=-1, keepdims=True)
    attn_crisis = np.abs(np.random.randn(12, 72))
    attn_crisis = attn_crisis / attn_crisis.sum(axis=-1, keepdims=True)
    fig = int5_attention_calm_vs_crisis(attn_calm, attn_crisis)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_int6_returns_figure(feature_names):
    np.random.seed(42)
    n = 300
    vsn_weights = np.abs(np.random.randn(n, 8))
    regime_labels = np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])
    fig = int6_feature_attribution_per_regime(vsn_weights, regime_labels, feature_names)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)
