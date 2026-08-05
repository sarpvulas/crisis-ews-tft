import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from analysis.graphs.model_performance import (
    mp1_roc_curves,
    mp2_pr_curves,
    mp3_lead_time_boxplot,
    mp4_calibration_plot,
    mp5_cap_curves,
    mp6_comparison_bars,
    mp7_temporal_generalization,
    mp8_ablation_bars,
)


@pytest.fixture
def results_dict():
    """Synthetic results for 3 models."""
    np.random.seed(42)
    n = 500
    y_true = np.concatenate([np.zeros(450), np.ones(50)])  # 10% positive
    return {
        "RA-TFT": {
            "y_true": y_true,
            "y_score": np.clip(y_true * 0.7 + np.random.randn(n) * 0.2, 0, 1),
        },
        "Vanilla TFT": {
            "y_true": y_true,
            "y_score": np.clip(y_true * 0.5 + np.random.randn(n) * 0.25, 0, 1),
        },
        "LSTM": {
            "y_true": y_true,
            "y_score": np.clip(y_true * 0.3 + np.random.randn(n) * 0.3, 0, 1),
        },
    }


@pytest.fixture
def lead_times_dict():
    np.random.seed(42)
    return {
        "RA-TFT": np.random.exponential(8, size=20) + 2,
        "Vanilla TFT": np.random.exponential(5, size=20) + 1,
        "LSTM": np.random.exponential(3, size=20) + 0.5,
    }


@pytest.fixture
def metrics_dict():
    return {
        "RA-TFT": {"AUC": 0.92, "PR-AUC": 0.78, "F1": 0.73, "Lead Time": 8.2},
        "Vanilla TFT": {"AUC": 0.87, "PR-AUC": 0.65, "F1": 0.62, "Lead Time": 5.4},
        "LSTM": {"AUC": 0.80, "PR-AUC": 0.52, "F1": 0.55, "Lead Time": 3.1},
    }


@pytest.fixture
def rolling_results():
    return {
        "RA-TFT": [{"fold": i, "pr_auc": 0.75 + np.random.randn() * 0.05} for i in range(5)],
        "Vanilla TFT": [{"fold": i, "pr_auc": 0.60 + np.random.randn() * 0.08} for i in range(5)],
    }


@pytest.fixture
def ablation_results():
    return {
        "Full RA-TFT": {"PR-AUC": 0.78, "AUC": 0.92},
        "- Regime Module": {"PR-AUC": 0.65, "AUC": 0.87},
        "- Regime Attention": {"PR-AUC": 0.71, "AUC": 0.90},
        "- Early Warning Loss": {"PR-AUC": 0.74, "AUC": 0.91},
        "- Regime Aux Task": {"PR-AUC": 0.72, "AUC": 0.89},
    }


def test_mp1_roc_curves(results_dict):
    fig = mp1_roc_curves(results_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp2_pr_curves(results_dict):
    fig = mp2_pr_curves(results_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp3_lead_time_boxplot(lead_times_dict):
    fig = mp3_lead_time_boxplot(lead_times_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp4_calibration_plot(results_dict):
    fig = mp4_calibration_plot(results_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp5_cap_curves(results_dict):
    fig = mp5_cap_curves(results_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp6_comparison_bars(metrics_dict):
    fig = mp6_comparison_bars(metrics_dict)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp7_temporal_generalization(rolling_results):
    fig = mp7_temporal_generalization(rolling_results)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_mp8_ablation_bars(ablation_results):
    fig = mp8_ablation_bars(ablation_results)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)
