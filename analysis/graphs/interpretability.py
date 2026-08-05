"""Interpretability graphs (INT-1 through INT-6) for thesis.

Each function takes extracted model internals and returns a matplotlib Figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Optional, Tuple

from analysis.graphs.style import (
    apply_thesis_style, REGIME_COLORS, REGIME_NAMES,
    regime_color, regime_name,
)


def int1_variable_importance_over_time(
    vsn_weights: np.ndarray,
    dates: np.ndarray,
    feature_names: List[str],
    top_k: int = 8,
) -> Figure:
    """INT-1: Variable importance over time (stacked area of VSN weights).

    Args:
        vsn_weights: Array of shape (n_timesteps, n_features) — time-averaged
            VSN weights per period.
        dates: Array of datetime-like values for the x-axis.
        feature_names: Names for each feature.
        top_k: Number of top features to show (rest grouped as "Other").

    Returns:
        Figure with stacked area chart.
    """
    with apply_thesis_style():
        # Normalize weights per timestep
        row_sums = vsn_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        normalized = vsn_weights / row_sums

        # Select top-k by mean importance
        mean_importance = normalized.mean(axis=0)
        top_indices = np.argsort(-mean_importance)[:top_k]

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
        selected_data = normalized[:, top_indices]
        selected_names = [feature_names[i] for i in top_indices]

        ax.stackplot(
            dates, selected_data.T,
            labels=selected_names, colors=colors, alpha=0.8,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Relative Importance")
        ax.set_title("INT-1: Variable Importance Over Time (VSN Weights)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig


def int2_attention_heatmap(
    attention: np.ndarray,
    crisis_name: str = "Crisis Event",
    time_labels: Optional[List[str]] = None,
) -> Figure:
    """INT-2: Attention heatmap for a specific crisis event.

    Args:
        attention: Array of shape (num_heads, decoder_steps, total_steps) or
            (decoder_steps, total_steps) — attention weights for one sample.
        crisis_name: Name of the crisis for the title.
        time_labels: Optional labels for the time axis.

    Returns:
        Figure with attention heatmap.
    """
    with apply_thesis_style():
        if attention.ndim == 3:
            # Average over heads
            attn_avg = attention.mean(axis=0)
        else:
            attn_avg = attention

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(attn_avg, cmap="YlOrRd", aspect="auto", interpolation="nearest")

        ax.set_xlabel("Input Time Steps (Encoder + Decoder)")
        ax.set_ylabel("Decoder Time Steps")
        ax.set_title(f"INT-2: Attention Heatmap — {crisis_name}")

        if time_labels is not None:
            step = max(1, len(time_labels) // 10)
            ax.set_xticks(range(0, len(time_labels), step))
            ax.set_xticklabels(time_labels[::step], rotation=45, ha="right", fontsize=8)

        fig.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)
        fig.tight_layout()
        return fig


def int3_regime_transition_diagram(
    transition_matrix: np.ndarray,
) -> Figure:
    """INT-3: Regime transition diagram (state diagram with edge weights).

    Args:
        transition_matrix: (3, 3) learned transition probability matrix.

    Returns:
        Figure with state diagram showing transition probabilities.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        # State positions (triangle layout)
        positions = {
            0: (-0.8, -0.5),   # Calm (bottom left)
            1: (0.8, -0.5),    # Stress (bottom right)
            2: (0.0, 0.7),     # Crisis (top center)
        }

        # Draw states as circles
        for state_idx, (x, y) in positions.items():
            circle = plt.Circle(
                (x, y), 0.3, facecolor=regime_color(state_idx),
                edgecolor="black", linewidth=1.5, alpha=0.8, zorder=3,
            )
            ax.add_patch(circle)
            ax.text(x, y + 0.05, regime_name(state_idx),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white", zorder=4)
            # Self-transition probability
            self_prob = transition_matrix[state_idx, state_idx]
            ax.text(x, y - 0.12, f"({self_prob:.2f})",
                    ha="center", va="center", fontsize=9, color="white", zorder=4)

        # Draw transitions between states
        for src in range(3):
            for dst in range(3):
                if src == dst:
                    continue
                prob = transition_matrix[src, dst]
                if prob < 0.005:
                    continue

                sx, sy = positions[src]
                dx, dy = positions[dst]

                # Shorten arrows to not overlap circles
                vec = np.array([dx - sx, dy - sy])
                length = np.linalg.norm(vec)
                unit = vec / length
                start = np.array([sx, sy]) + unit * 0.32
                end = np.array([dx, dy]) - unit * 0.32

                arrow = FancyArrowPatch(
                    start, end,
                    arrowstyle="-|>", mutation_scale=15,
                    linewidth=max(0.5, prob * 5),
                    color="gray", alpha=0.7, zorder=2,
                )
                ax.add_patch(arrow)

                # Label
                mid = (start + end) / 2
                # Offset label perpendicular to arrow
                perp = np.array([-unit[1], unit[0]]) * 0.12
                ax.text(mid[0] + perp[0], mid[1] + perp[1], f"{prob:.3f}",
                        ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="gray", alpha=0.8))

        ax.set_title("INT-3: Learned Regime Transition Probabilities")
        fig.tight_layout()
        return fig


def int4_regime_timeline(
    regime_probs: np.ndarray,
    dates: np.ndarray,
    crisis_dates: Optional[List] = None,
    crisis_names: Optional[List[str]] = None,
) -> Figure:
    """INT-4: Regime probability timeline (stacked area + event markers).

    Args:
        regime_probs: (n_timesteps, 3) regime probabilities.
        dates: Array of datetime-like values.
        crisis_dates: Optional list of crisis onset dates for vertical markers.
        crisis_names: Optional names for each crisis date.

    Returns:
        Figure with stacked area chart of regime probabilities.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(12, 4.5))

        ax.stackplot(
            dates,
            regime_probs[:, 0], regime_probs[:, 1], regime_probs[:, 2],
            labels=REGIME_NAMES,
            colors=REGIME_COLORS,
            alpha=0.8,
        )

        if crisis_dates is not None:
            for i, cd in enumerate(crisis_dates):
                label = crisis_names[i] if crisis_names and i < len(crisis_names) else None
                ax.axvline(cd, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                if label:
                    ax.text(cd, 1.02, label, rotation=45, ha="left", va="bottom",
                            fontsize=8, transform=ax.get_xaxis_transform())

        ax.set_xlabel("Date")
        ax.set_ylabel("Regime Probability")
        ax.set_title("INT-4: Regime Probability Timeline")
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig


def int5_attention_calm_vs_crisis(
    attn_calm: np.ndarray,
    attn_crisis: np.ndarray,
) -> Figure:
    """INT-5: Attention pattern comparison — calm vs crisis (side-by-side).

    Args:
        attn_calm: (decoder_steps, total_steps) mean attention during calm.
        attn_crisis: (decoder_steps, total_steps) mean attention during crisis.

    Returns:
        Figure with side-by-side heatmaps.
    """
    with apply_thesis_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        vmin = min(attn_calm.min(), attn_crisis.min())
        vmax = max(attn_calm.max(), attn_crisis.max())

        im1 = ax1.imshow(attn_calm, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title("Calm Regime")
        ax1.set_xlabel("Input Time Steps")
        ax1.set_ylabel("Decoder Time Steps")

        im2 = ax2.imshow(attn_crisis, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title("Crisis Regime")
        ax2.set_xlabel("Input Time Steps")
        ax2.set_ylabel("Decoder Time Steps")

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Attention Weight")
        fig.suptitle("INT-5: Attention Patterns — Calm vs Crisis", y=1.02)
        fig.tight_layout()
        return fig


def int6_feature_attribution_per_regime(
    vsn_weights: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> Figure:
    """INT-6: Feature attribution per regime (grouped bar, top features).

    Args:
        vsn_weights: (n_samples, n_features) VSN weights.
        regime_labels: (n_samples,) regime index per sample.
        feature_names: Feature names.
        top_k: Number of top features to show.

    Returns:
        Figure with grouped horizontal bar chart.
    """
    with apply_thesis_style():
        top_k = min(top_k, len(feature_names))

        # Compute mean importance per regime
        regime_importance = {}
        for r in range(3):
            mask = regime_labels == r
            if mask.sum() > 0:
                regime_importance[r] = vsn_weights[mask].mean(axis=0)
            else:
                regime_importance[r] = np.zeros(len(feature_names))

        # Select top-k features by overall importance
        overall = sum(regime_importance.values()) / 3
        top_indices = np.argsort(-overall)[:top_k]
        top_names = [feature_names[i] for i in top_indices]

        fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.4 + 1)))

        y = np.arange(top_k)
        height = 0.25

        for r in range(3):
            values = regime_importance[r][top_indices]
            ax.barh(
                y + (r - 1) * height, values, height,
                label=regime_name(r), color=regime_color(r), alpha=0.8,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(top_names)
        ax.set_xlabel("Mean VSN Weight")
        ax.set_title("INT-6: Feature Attribution by Regime")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig
