"""Data exploration graphs (DE-1 through DE-6) for thesis.

Each function takes processed data as input and returns a matplotlib Figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from typing import List, Optional, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from analysis.graphs.style import (
    apply_thesis_style, CALM_COLOR, STRESS_COLOR, CRISIS_COLOR,
    REGIME_COLORS, REGIME_NAMES, regime_color, regime_name,
)


def de1_crisis_timeline(
    crisis_db: pd.DataFrame,
    crash_events: Optional[pd.DataFrame] = None,
) -> Figure:
    """DE-1: Crisis timeline Gantt chart.

    Args:
        crisis_db: DataFrame with columns [country, start, end] where start/end
            are datetime-like. Rows represent crisis periods per country.
        crash_events: Optional DataFrame with columns [name, start, end] for
            market crash events (Task B overlay).

    Returns:
        matplotlib Figure with Gantt chart of crises across countries.
    """
    with apply_thesis_style():
        countries = crisis_db["country"].unique()
        n_countries = len(countries)

        fig_height = max(4, 0.35 * n_countries + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        country_map = {c: i for i, c in enumerate(sorted(countries))}

        for _, row in crisis_db.iterrows():
            y = country_map[row["country"]]
            start = pd.to_datetime(row["start"])
            end = pd.to_datetime(row["end"])
            duration = (end - start).days
            ax.barh(
                y, duration, left=start, height=0.6,
                color=CRISIS_COLOR, alpha=0.8, edgecolor="none",
            )

        if crash_events is not None and len(crash_events) > 0:
            for _, row in crash_events.iterrows():
                start = pd.to_datetime(row["start"])
                end = pd.to_datetime(row["end"])
                ax.axvspan(start, end, alpha=0.15, color=STRESS_COLOR, zorder=0)

        ax.set_yticks(range(n_countries))
        ax.set_yticklabels(sorted(countries))
        ax.set_xlabel("Year")
        ax.set_title("DE-1: Systemic Crisis Timeline")
        ax.invert_yaxis()

        legend_elements = [Patch(facecolor=CRISIS_COLOR, alpha=0.8, label="Banking Crisis")]
        if crash_events is not None:
            legend_elements.append(
                Patch(facecolor=STRESS_COLOR, alpha=0.15, label="Market Crash")
            )
        ax.legend(handles=legend_elements, loc="lower right")
        fig.tight_layout()
        return fig


def de2_correlation_heatmap(
    features: pd.DataFrame,
    regime_labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Figure:
    """DE-2: Feature correlation heatmap, split by regime (calm vs crisis).

    Args:
        features: DataFrame of shape (n_samples, n_features).
        regime_labels: Array of regime indices (0=calm, 1=stress, 2=crisis).
        feature_names: Optional feature names; uses columns if None.

    Returns:
        Figure with two side-by-side clustered heatmaps.
    """
    with apply_thesis_style():
        if feature_names is None:
            feature_names = list(features.columns)

        calm_mask = regime_labels == 0
        crisis_mask = regime_labels == 2

        corr_calm = features[calm_mask].corr()
        corr_crisis = features[crisis_mask].corr()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        vmin, vmax = -1, 1
        im1 = ax1.imshow(corr_calm.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax1.set_title("Calm Regime")
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax1.set_yticklabels(feature_names, fontsize=8)

        im2 = ax2.imshow(corr_crisis.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax2.set_title("Crisis Regime")
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax2.set_yticklabels(feature_names, fontsize=8)

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Correlation")
        fig.suptitle("DE-2: Feature Correlations by Regime", y=1.02)
        fig.tight_layout()
        return fig


def de3_class_imbalance(regime_labels: np.ndarray) -> Figure:
    """DE-3: Class imbalance horizontal bar chart.

    Args:
        regime_labels: Array of regime indices (0=calm, 1=stress, 2=crisis).

    Returns:
        Figure with horizontal bars showing class distribution.
    """
    with apply_thesis_style():
        unique, counts = np.unique(regime_labels, return_counts=True)
        total = len(regime_labels)
        percentages = counts / total * 100

        fig, ax = plt.subplots(figsize=(7, 3.5))

        bars = ax.barh(
            [regime_name(int(u)) for u in unique],
            counts,
            color=[regime_color(int(u)) for u in unique],
            edgecolor="white",
            height=0.5,
        )

        for bar, pct, cnt in zip(bars, percentages, counts):
            ax.text(
                bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=10,
            )

        ax.set_xlabel("Number of Observations")
        ax.set_title("DE-3: Regime Class Distribution")
        ax.set_xlim(0, max(counts) * 1.2)
        fig.tight_layout()
        return fig


def de4_event_study(
    features: pd.DataFrame,
    crisis_dates: List[pd.Timestamp],
    window: int = 24,
    feature_cols: Optional[List[str]] = None,
) -> Figure:
    """DE-4: Event study — mean feature trajectories aligned to crisis onset.

    Args:
        features: DataFrame indexed by date with feature columns.
        crisis_dates: List of crisis onset dates.
        window: Number of periods before and after onset.
        feature_cols: Which features to plot (top 4 if None).

    Returns:
        Figure with aligned mean trajectories.
    """
    with apply_thesis_style():
        if feature_cols is None:
            feature_cols = list(features.columns[:4])

        n_features = len(feature_cols)
        fig, axes = plt.subplots(
            n_features, 1, figsize=(8, 2.5 * n_features), sharex=True,
        )
        if n_features == 1:
            axes = [axes]

        time_axis = np.arange(-window, window + 1)

        for feat_idx, col in enumerate(feature_cols):
            trajectories = []
            for onset in crisis_dates:
                loc = features.index.get_indexer([onset], method="nearest")[0]
                start = loc - window
                end = loc + window + 1
                if start >= 0 and end <= len(features):
                    vals = features[col].iloc[start:end].values
                    if len(vals) == len(time_axis):
                        trajectories.append(vals)

            if len(trajectories) > 0:
                traj_array = np.array(trajectories)
                mean_traj = np.mean(traj_array, axis=0)
                std_traj = np.std(traj_array, axis=0)

                ax = axes[feat_idx]
                ax.plot(time_axis, mean_traj, color=CRISIS_COLOR, linewidth=1.5)
                ax.fill_between(
                    time_axis, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.2, color=CRISIS_COLOR,
                )
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_ylabel(col)
            else:
                axes[feat_idx].set_ylabel(col)
                axes[feat_idx].text(0.5, 0.5, "No data", transform=axes[feat_idx].transAxes,
                                    ha="center", va="center")

        axes[-1].set_xlabel("Periods Relative to Crisis Onset")
        fig.suptitle("DE-4: Event Study — Feature Trajectories Around Crisis Onset")
        fig.tight_layout()
        return fig


def de5_rolling_statistics(
    features: pd.DataFrame,
    crisis_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    feature_cols: Optional[List[str]] = None,
    rolling_window: int = 12,
) -> Figure:
    """DE-5: Rolling statistics with crisis period shading.

    Args:
        features: DataFrame indexed by date with feature columns.
        crisis_periods: List of (start, end) tuples for crisis shading.
        feature_cols: Which features to plot (top 4 if None).
        rolling_window: Window size for rolling mean/std.

    Returns:
        Multi-panel figure with rolling mean, std, and crisis shading.
    """
    with apply_thesis_style():
        if feature_cols is None:
            feature_cols = list(features.columns[:4])

        n_features = len(feature_cols)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 2.5 * n_features), sharex=True)
        if n_features == 1:
            axes = [axes]

        for feat_idx, col in enumerate(feature_cols):
            ax = axes[feat_idx]
            series = features[col]
            rolling_mean = series.rolling(rolling_window, min_periods=1).mean()
            rolling_std = series.rolling(rolling_window, min_periods=1).std()

            ax.plot(features.index, rolling_mean, color=CALM_COLOR, linewidth=1.2, label="Rolling Mean")
            ax.fill_between(
                features.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.15, color=CALM_COLOR,
            )

            for start, end in crisis_periods:
                ax.axvspan(start, end, alpha=0.2, color=CRISIS_COLOR, zorder=0)

            ax.set_ylabel(col)
            if feat_idx == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Date")
        fig.suptitle("DE-5: Rolling Statistics with Crisis Periods")
        fig.tight_layout()
        return fig


def de6_pca_tsne(
    features: pd.DataFrame,
    regime_labels: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Figure:
    """DE-6: PCA and t-SNE scatter plots colored by regime.

    Args:
        features: DataFrame of shape (n_samples, n_features).
        regime_labels: Array of regime indices (0=calm, 1=stress, 2=crisis).
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Figure with side-by-side PCA and t-SNE scatter plots.
    """
    with apply_thesis_style():
        X = features.values if isinstance(features, pd.DataFrame) else features
        X_clean = np.nan_to_num(X, nan=0.0)

        # PCA
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_clean)

        # t-SNE
        tsne = TSNE(
            n_components=2, perplexity=min(perplexity, max(5, len(X_clean) // 4)),
            random_state=random_state, max_iter=300,
        )
        X_tsne = tsne.fit_transform(X_clean)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for idx in range(3):
            mask = regime_labels == idx
            if mask.sum() > 0:
                ax1.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    c=regime_color(idx), label=regime_name(idx),
                    alpha=0.5, s=15, edgecolors="none",
                )
                ax2.scatter(
                    X_tsne[mask, 0], X_tsne[mask, 1],
                    c=regime_color(idx), label=regime_name(idx),
                    alpha=0.5, s=15, edgecolors="none",
                )

        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax1.set_title("PCA")
        ax1.legend()

        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_title("t-SNE")
        ax2.legend()

        fig.suptitle("DE-6: Feature Space by Regime")
        fig.tight_layout()
        return fig
