"""Model performance graphs (MP-1 through MP-8) for thesis.

Each function takes results dicts (model_name -> predictions/metrics)
and returns a matplotlib Figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from analysis.graphs.style import (
    apply_thesis_style, get_model_color, ABLATION_COLORS,
)


def mp1_roc_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
) -> Figure:
    """MP-1: ROC curves for all models with AUC in legend.

    Args:
        results_dict: {model_name: {"y_true": array, "y_score": array}}.

    Returns:
        Figure with overlaid ROC curves.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, data in results_dict.items():
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_score"])
            roc_auc = auc(fpr, tpr)
            color = get_model_color(model_name)
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"{model_name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("MP-1: ROC Curves")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        return fig


def mp2_pr_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
) -> Figure:
    """MP-2: Precision-Recall curves with PR-AUC in legend.

    Args:
        results_dict: {model_name: {"y_true": array, "y_score": array}}.

    Returns:
        Figure with overlaid PR curves.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, data in results_dict.items():
            precision, recall, _ = precision_recall_curve(data["y_true"], data["y_score"])
            pr_auc = average_precision_score(data["y_true"], data["y_score"])
            color = get_model_color(model_name)
            ax.plot(recall, precision, color=color, linewidth=1.5,
                    label=f"{model_name} (PR-AUC={pr_auc:.3f})")

        # Baseline: prevalence
        if results_dict:
            first = next(iter(results_dict.values()))
            prevalence = np.mean(first["y_true"])
            ax.axhline(prevalence, color="gray", linestyle="--", linewidth=0.8,
                        alpha=0.5, label=f"Baseline ({prevalence:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("MP-2: Precision-Recall Curves")
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        return fig


def mp3_lead_time_boxplot(
    lead_times_dict: Dict[str, np.ndarray],
) -> Figure:
    """MP-3: Early warning lead time box plot per model.

    Args:
        lead_times_dict: {model_name: array of lead times (months or days)}.

    Returns:
        Figure with grouped box plots.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(8, 5))

        names = list(lead_times_dict.keys())
        data = [lead_times_dict[n] for n in names]
        colors = [get_model_color(n) for n in names]

        bp = ax.boxplot(
            data, tick_labels=names, patch_artist=True,
            widths=0.5, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=5),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Lead Time (periods before onset)")
        ax.set_title("MP-3: Early Warning Lead Time")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        return fig


def mp4_calibration_plot(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    n_bins: int = 10,
) -> Figure:
    """MP-4: Calibration (reliability) diagram.

    Args:
        results_dict: {model_name: {"y_true": array, "y_score": array}}.
        n_bins: Number of calibration bins.

    Returns:
        Figure with reliability diagram and histogram of predictions.
    """
    with apply_thesis_style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]},
        )

        ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfectly Calibrated")

        for model_name, data in results_dict.items():
            y_true = data["y_true"]
            y_score = data["y_score"]
            color = get_model_color(model_name)

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_means = []

            for i in range(n_bins):
                mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_means.append(y_true[mask].mean())

            ax1.plot(bin_centers, bin_means, "o-", color=color, linewidth=1.2,
                     markersize=4, label=model_name)
            ax2.hist(y_score, bins=bin_edges, alpha=0.4, color=color, label=model_name)

        ax1.set_ylabel("Observed Frequency")
        ax1.set_title("MP-4: Calibration Plot")
        ax1.legend(loc="upper left")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        fig.tight_layout()
        return fig


def mp5_cap_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
) -> Figure:
    """MP-5: Cumulative accuracy profile (CAP) curves.

    Args:
        results_dict: {model_name: {"y_true": array, "y_score": array}}.

    Returns:
        Figure with CAP curves.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, data in results_dict.items():
            y_true = data["y_true"]
            y_score = data["y_score"]
            color = get_model_color(model_name)

            # Sort by score descending
            sorted_idx = np.argsort(-y_score)
            sorted_true = y_true[sorted_idx]

            # Cumulative positive rate
            cum_positives = np.cumsum(sorted_true)
            total_positives = cum_positives[-1] if cum_positives[-1] > 0 else 1
            cum_pos_rate = cum_positives / total_positives
            x_axis = np.arange(1, len(y_true) + 1) / len(y_true)

            ax.plot(x_axis, cum_pos_rate, color=color, linewidth=1.5, label=model_name)

        # Random baseline
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
        # Perfect model
        if results_dict:
            first = next(iter(results_dict.values()))
            prevalence = np.mean(first["y_true"])
            if prevalence > 0:
                ax.plot([0, prevalence, 1], [0, 1, 1], "k:", linewidth=0.8,
                        alpha=0.3, label="Perfect")

        ax.set_xlabel("Fraction of Population")
        ax.set_ylabel("Fraction of Positives Captured")
        ax.set_title("MP-5: Cumulative Accuracy Profile")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        return fig


def mp6_comparison_bars(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> Figure:
    """MP-6: Model comparison grouped bar chart.

    Args:
        metrics_dict: {model_name: {metric_name: value}}.
        metric_names: Which metrics to include (uses all if None).

    Returns:
        Figure with grouped bars.
    """
    with apply_thesis_style():
        models = list(metrics_dict.keys())
        if metric_names is None:
            metric_names = list(next(iter(metrics_dict.values())).keys())

        n_models = len(models)
        n_metrics = len(metric_names)
        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))

        for i, model in enumerate(models):
            values = [metrics_dict[model].get(m, 0) for m in metric_names]
            color = get_model_color(model)
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("MP-6: Model Comparison")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig


def mp7_temporal_generalization(
    rolling_results: Dict[str, List[Dict[str, float]]],
    metric: str = "pr_auc",
) -> Figure:
    """MP-7: Temporal generalization across rolling windows.

    Args:
        rolling_results: {model_name: [{"fold": i, metric: value, "test_start": str, ...}]}.
        metric: Which metric to plot.

    Returns:
        Figure with line plot per model across folds.
    """
    with apply_thesis_style():
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name, folds in rolling_results.items():
            fold_ids = [f.get("fold", i) for i, f in enumerate(folds)]
            values = [f[metric] for f in folds]
            color = get_model_color(model_name)

            ax.plot(fold_ids, values, "o-", color=color, linewidth=1.5,
                    markersize=6, label=model_name)

        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"MP-7: Temporal Generalization ({metric})")
        ax.legend()
        fig.tight_layout()
        return fig


def mp8_ablation_bars(
    ablation_results: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> Figure:
    """MP-8: Ablation study grouped bar chart.

    Args:
        ablation_results: {ablation_name: {metric_name: value}}.
        metric_names: Which metrics to include.

    Returns:
        Figure with grouped bars, one group per ablation.
    """
    with apply_thesis_style():
        ablations = list(ablation_results.keys())
        if metric_names is None:
            metric_names = list(next(iter(ablation_results.values())).keys())

        n_ablations = len(ablations)
        n_metrics = len(metric_names)
        x = np.arange(n_ablations)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(8, n_ablations * 1.5), 5))

        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for i, metric in enumerate(metric_names):
            values = [ablation_results[a].get(metric, 0) for a in ablations]
            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(ablations, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("MP-8: Ablation Study")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig
