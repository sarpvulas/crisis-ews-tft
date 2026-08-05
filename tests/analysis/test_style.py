import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from analysis.graphs.style import (
    apply_thesis_style,
    CALM_COLOR, STRESS_COLOR, CRISIS_COLOR,
    MODEL_COLORS, get_model_color, regime_color, regime_name,
    THESIS_RCPARAMS,
)


def test_apply_thesis_style_context_manager():
    """Verify the context manager applies rcParams and restores them."""
    original_dpi = plt.rcParams["figure.dpi"]
    with apply_thesis_style():
        assert plt.rcParams["figure.dpi"] == 300
        assert "serif" in plt.rcParams["font.family"]
        assert plt.rcParams["axes.spines.top"] is False
    # Should be restored
    assert plt.rcParams["figure.dpi"] == original_dpi


def test_regime_colors_defined():
    assert CALM_COLOR.startswith("#")
    assert STRESS_COLOR.startswith("#")
    assert CRISIS_COLOR.startswith("#")


def test_model_colors_has_all_models():
    expected = ["RA-TFT", "Vanilla TFT", "LSTM", "XGBoost", "Logistic Regression"]
    for name in expected:
        assert name in MODEL_COLORS


def test_get_model_color_fallback():
    color = get_model_color("UnknownModel")
    assert color.startswith("#")


def test_regime_color_and_name():
    assert regime_color(0) == CALM_COLOR
    assert regime_color(2) == CRISIS_COLOR
    assert regime_name(0) == "Calm"
    assert regime_name(2) == "Crisis"
