"""Evaluation metrics for financial crisis prediction."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision)."""
    return float(average_precision_score(y_true, y_scores))


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC."""
    return float(roc_auc_score(y_true, y_scores))


def compute_early_warning_lead_time(
    probabilities: np.ndarray,
    crisis_events: List[int],
    threshold: float = 0.5,
    sustained: int = 3,
) -> List[int]:
    """Compute early warning lead time for each crisis event.

    For each crisis onset time, find the first time the model's probability
    exceeds `threshold` for `sustained` consecutive periods before the crisis.
    Lead time = crisis_onset - first_sustained_alarm_start.

    Args:
        probabilities: 1D array of crisis probabilities over time.
        crisis_events: List of time indices where crises begin.
        threshold: Probability threshold for alarm.
        sustained: Number of consecutive periods above threshold to count as alarm.

    Returns:
        List of lead times (in periods) for each successfully predicted crisis.
    """
    above = probabilities >= threshold
    lead_times = []

    # Sort events to process in order, track previous crisis to avoid overlap
    sorted_events = sorted(crisis_events)

    for i, crisis_t in enumerate(sorted_events):
        # Start searching from previous crisis (or 0) to avoid counting
        # alarms that belong to a previous crisis
        search_start = sorted_events[i - 1] + 1 if i > 0 else 0

        # Look for first sustained alarm in the window before this crisis
        first_alarm = None
        for t in range(search_start, crisis_t):
            if t + sustained <= len(probabilities):
                if all(above[t:t + sustained]):
                    first_alarm = t
                    break

        if first_alarm is not None and first_alarm < crisis_t:
            lead_times.append(crisis_t - first_alarm)

    return lead_times


def compute_calibration_error(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions into n_bins equally-spaced bins by predicted probability,
    computes |mean_predicted - mean_actual| per bin, weighted by bin size.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_scores >= lo) & (y_scores <= hi)
        else:
            mask = (y_scores >= lo) & (y_scores < hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_scores[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Brier score (lower is better).

    Brier = mean((predicted - actual)^2). Decomposes into calibration + discrimination.
    Post-hoc calibration improves the calibration component while preserving discrimination.
    """
    return float(brier_score_loss(y_true, y_scores))


class PostHocCalibrator:
    """Post-hoc probability calibration using Platt scaling or isotonic regression.

    Fits a monotonic mapping from raw model probabilities to calibrated probabilities
    using a held-out validation set. Does not change the model or its ranking —
    ROC-AUC is invariant, but ECE and Brier score improve.

    Usage:
        cal = PostHocCalibrator(method="platt")
        cal.fit(val_probs, val_labels)
        calibrated_test = cal.transform(test_probs)
    """

    METHODS = ("platt", "isotonic")

    def __init__(self, method: str = "platt"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'")
        self.method = method
        self._fitted = False

        if method == "platt":
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
        else:
            self._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, y_scores: np.ndarray, y_true: np.ndarray) -> "PostHocCalibrator":
        """Fit the calibration mapping on validation set predictions.

        Args:
            y_scores: Raw model probabilities (after sigmoid), shape (n,).
            y_true: Binary ground truth labels, shape (n,).
        """
        y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
        y_true = np.asarray(y_true, dtype=np.float64).ravel()

        if self.method == "platt":
            # Platt scaling: logistic regression on log-odds of raw probabilities
            # Clip to avoid log(0) or log(inf)
            eps = 1e-8
            clipped = np.clip(y_scores, eps, 1.0 - eps)
            log_odds = np.log(clipped / (1.0 - clipped))
            self._calibrator.fit(log_odds.reshape(-1, 1), y_true)
        else:
            # Isotonic regression: direct mapping from raw probs to calibrated probs
            self._calibrator.fit(y_scores, y_true)

        self._fitted = True
        return self

    def transform(self, y_scores: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to new predictions.

        Args:
            y_scores: Raw model probabilities, shape (n,).

        Returns:
            Calibrated probabilities, shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("Calibrator must be fit() before transform()")

        y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

        if self.method == "platt":
            eps = 1e-8
            clipped = np.clip(y_scores, eps, 1.0 - eps)
            log_odds = np.log(clipped / (1.0 - clipped))
            return self._calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        else:
            return self._calibrator.transform(y_scores)

    def fit_transform(self, y_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit on data and return calibrated scores (for validation set diagnostics)."""
        self.fit(y_scores, y_true)
        return self.transform(y_scores)


def compute_bootstrap_ci(
    metric_fn: Callable,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function(y_true, y_scores) -> float
        y_true: Ground truth labels.
        y_scores: Predicted scores.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_true = y_true[idx]
        boot_scores = y_scores[idx]
        # Skip if only one class in bootstrap sample
        if len(np.unique(boot_true)) < 2:
            continue
        scores.append(metric_fn(boot_true, boot_scores))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return lower, upper


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    task: str = "B",
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Run full evaluation on a dataset, returning all metrics.

    Args:
        model: Trained model with .forward(batch) -> outputs dict.
        dataset: CrisisDataset instance.
        task: "B" (regression/crash prediction).
        device: Device for inference.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with pr_auc, roc_auc, calibration_error, early_warning_lead_time_mean,
        and bootstrap CIs.
    """
    from torch.utils.data import DataLoader

    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=dataset.collate_fn)

    all_preds = []
    all_labels = []

    for batch_dict, targets_dict in loader:
        batch_dev = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(batch_dev)

        # EWS: single logit output → sigmoid probability
        logits = outputs["predicted"].squeeze(-1)
        probs = torch.sigmoid(logits)
        labels = targets_dict["ews_label"]

        all_preds.append(probs.cpu().numpy().flatten())
        all_labels.append(labels.cpu().numpy().flatten())

    y_scores = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    results = {}

    # Only compute metrics if both classes present
    if len(np.unique(y_true)) >= 2:
        results["pr_auc"] = compute_pr_auc(y_true, y_scores)
        results["roc_auc"] = compute_roc_auc(y_true, y_scores)
        results["calibration_error"] = compute_calibration_error(y_true, y_scores)
        results["brier_score"] = compute_brier_score(y_true, y_scores)

        pr_ci = compute_bootstrap_ci(compute_pr_auc, y_true, y_scores)
        results["pr_auc_ci_lower"] = pr_ci[0]
        results["pr_auc_ci_upper"] = pr_ci[1]

    results["positive_rate"] = float(y_true.mean())
    results["n_samples"] = len(y_true)

    return results
