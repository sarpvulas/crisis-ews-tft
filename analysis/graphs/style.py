"""Thesis-quality matplotlib styling for crisis prediction graphs.

Provides consistent visual styling across all 20 thesis figures:
- Serif fonts at publication-appropriate sizes
- 300 DPI for print quality
- Three-regime color scheme (calm/stress/crisis)
- Model comparison palette
"""

from contextlib import contextmanager
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

# === Regime Colors ===
CALM_COLOR = "#3274A1"       # Blue
STRESS_COLOR = "#E1812C"     # Orange
CRISIS_COLOR = "#C03D3E"     # Red
REGIME_COLORS = [CALM_COLOR, STRESS_COLOR, CRISIS_COLOR]
REGIME_NAMES = ["Calm", "Stress", "Crisis"]

# === Model Comparison Palette ===
MODEL_COLORS: Dict[str, str] = {
    "RA-TFT": "#2D6A4F",          # Dark green (primary model)
    "Vanilla TFT": "#3274A1",     # Blue
    "LSTM": "#E1812C",            # Orange
    "XGBoost": "#9467BD",         # Purple
    "Logistic Regression": "#8C564B",  # Brown
}

# Ablation palette (variations of green to distinguish from baselines)
ABLATION_COLORS: Dict[str, str] = {
    "Full RA-TFT": "#2D6A4F",
    "- Regime Module": "#52B788",
    "- Regime Attention": "#74C69D",
    "- Early Warning Loss": "#95D5B2",
    "- Regime Aux Task": "#B7E4C7",
    "Vanilla TFT": "#3274A1",
}

# === Thesis rcParams ===
THESIS_RCPARAMS = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,

    # Figure
    "figure.figsize": (7, 5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,

    # Axes
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Grid
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # Legend
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",

    # Patch (bars, etc.)
    "patch.linewidth": 0.5,
}


@contextmanager
def apply_thesis_style():
    """Context manager that applies thesis-quality matplotlib styling.

    Usage:
        with apply_thesis_style():
            fig, ax = plt.subplots()
            ax.plot(x, y)
    """
    with mpl.rc_context(THESIS_RCPARAMS):
        yield


def get_model_color(model_name: str) -> str:
    """Get consistent color for a model name, with fallback."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    if model_name in ABLATION_COLORS:
        return ABLATION_COLORS[model_name]
    # Fallback: cycle through a default palette
    fallback = ["#D62728", "#FF7F0E", "#2CA02C", "#1F77B4", "#9467BD",
                "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    idx = hash(model_name) % len(fallback)
    return fallback[idx]


def regime_color(regime_idx: int) -> str:
    """Get color for regime index (0=calm, 1=stress, 2=crisis)."""
    return REGIME_COLORS[regime_idx % len(REGIME_COLORS)]


def regime_name(regime_idx: int) -> str:
    """Get name for regime index (0=calm, 1=stress, 2=crisis)."""
    return REGIME_NAMES[regime_idx % len(REGIME_NAMES)]
