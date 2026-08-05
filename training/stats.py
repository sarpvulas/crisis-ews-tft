"""Statistical tests for model comparison.

All tests assume aligned predictions on the *same* test windows — never compare
models that ran on different splits.

Three tests, picked for what they tell us:
  - paired_wilcoxon: non-parametric "is model A's per-event score reliably better
    than model B's?" — preferred for our ranking metrics (PR-AUC, ROC-AUC).
  - mcnemar: classification disagreement test — "do models A and B disagree
    asymmetrically at a chosen threshold?" — for binary decisions.
  - diebold_mariano: for continuous forecast errors (used on Task B quantile
    crash predictions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    method: str
    n: int


def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "greater",
) -> TestResult:
    """Per-bootstrap or per-seed paired Wilcoxon signed-rank.

    `scores_a` and `scores_b` are paired sequences — e.g. one entry per seed,
    one entry per bootstrap resample of the test set. `alternative="greater"`
    tests H1: A > B.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return TestResult(
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        method=f"wilcoxon_{alternative}",
        n=len(a),
    )


def mcnemar(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    y_true: np.ndarray,
) -> TestResult:
    """McNemar test on disagreement between two classifiers.

    Inputs are binary predictions (after thresholding). Returns chi-square
    statistic with continuity correction.
    """
    a = np.asarray(preds_a).astype(bool)
    b = np.asarray(preds_b).astype(bool)
    y = np.asarray(y_true).astype(bool)
    correct_a = a == y
    correct_b = b == y
    # b01: A correct, B wrong; b10: A wrong, B correct
    b01 = int(np.sum(correct_a & ~correct_b))
    b10 = int(np.sum(~correct_a & correct_b))
    if b01 + b10 == 0:
        return TestResult(statistic=0.0, pvalue=1.0, method="mcnemar", n=len(y))
    # Continuity-corrected statistic; chi-square with 1 dof
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    pvalue = float(stats.chi2.sf(stat, df=1))
    return TestResult(statistic=float(stat), pvalue=pvalue, method="mcnemar_cc", n=len(y))


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> TestResult:
    """Diebold-Mariano test for equal forecast accuracy.

    `errors_a`, `errors_b` are per-timestep forecast errors (e.g. residuals).
    `power=2` gives mean squared error loss; `power=1` gives MAE.
    `h` is the forecast horizon used for variance correction.
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = np.abs(a) ** power - np.abs(b) ** power
    mean_d = d.mean()
    # Newey-West variance with lag h-1
    n = len(d)
    var = np.var(d, ddof=1)
    if h > 1:
        for lag in range(1, h):
            cov = float(np.cov(d[:-lag], d[lag:])[0, 1])
            var += 2 * (1 - lag / h) * cov
    if var <= 0:
        return TestResult(statistic=0.0, pvalue=1.0, method=f"dm_h{h}_p{power}", n=n)
    stat = mean_d / np.sqrt(var / n)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return TestResult(statistic=float(stat), pvalue=pvalue, method=f"dm_h{h}_p{power}", n=n)


def paired_bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the *difference* metric_fn(y, A) - metric_fn(y, B).

    Resamples paired indices, recomputes both metrics on the resampled set,
    takes the difference, percentiles over bootstraps. Returns (mean_diff,
    lower_bound, upper_bound). Use this instead of comparing independent CIs
    — paired CI is tighter and respects the dependence between models on the
    same windows.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        sa = scores_a[idx]
        sb = scores_b[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(metric_fn(yt, sa) - metric_fn(yt, sb))
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    return (
        float(diffs.mean()),
        float(np.percentile(diffs, 100 * alpha)),
        float(np.percentile(diffs, 100 * (1 - alpha))),
    )
