"""Tests for training.splits and training.stats — the comparison harness."""

import numpy as np
import pytest

from training.splits import (
    PRIMARY_SPLIT_A,
    PRIMARY_SPLIT_B,
    WALKFORWARD_FOLDS,
    get_primary_split,
    get_walkforward_fold,
    split_dates,
)
from training.stats import (
    diebold_mariano,
    mcnemar,
    paired_bootstrap_ci,
    paired_wilcoxon,
)


# ---------- splits ----------

class TestSplits:

    def test_primary_split_dispatch(self):
        assert get_primary_split("A") is PRIMARY_SPLIT_A
        assert get_primary_split("a") is PRIMARY_SPLIT_A
        assert get_primary_split("B") is PRIMARY_SPLIT_B

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_primary_split("C")

    def test_walkforward_folds_present(self):
        expected = {"gfc", "covid", "bear2022", "eurozone2011"}
        assert expected.issubset(set(WALKFORWARD_FOLDS))

    def test_walkforward_dispatch(self):
        fold = get_walkforward_fold("gfc")
        assert fold.test_start.startswith("2008")
        assert fold.test_end.startswith("2009")

    def test_unknown_fold_raises(self):
        with pytest.raises(ValueError):
            get_walkforward_fold("not-a-real-crisis")

    def test_split_dates_returns_train_and_val_endpoints(self):
        train_end, val_end = split_dates(PRIMARY_SPLIT_B)
        assert train_end == PRIMARY_SPLIT_B.train_end
        assert val_end == PRIMARY_SPLIT_B.val_end

    def test_splits_are_monotonic_in_time(self):
        for split in [PRIMARY_SPLIT_A, PRIMARY_SPLIT_B] + list(WALKFORWARD_FOLDS.values()):
            assert split.train_start < split.train_end < split.val_start
            assert split.val_end < split.test_start < split.test_end, (
                f"split {split.name} has overlapping val/test"
            )


# ---------- stats ----------

class TestPairedWilcoxon:

    def test_a_strictly_greater_than_b(self):
        rng = np.random.RandomState(0)
        b = rng.uniform(0.3, 0.5, 30)
        a = b + 0.1  # A always beats B
        res = paired_wilcoxon(a, b, alternative="greater")
        assert res.pvalue < 0.01
        assert res.n == 30

    def test_a_equal_to_b_not_significant(self):
        rng = np.random.RandomState(0)
        a = rng.uniform(0.3, 0.5, 30)
        b = a.copy() + rng.normal(0, 1e-9, 30)
        res = paired_wilcoxon(a, b, alternative="greater")
        assert res.pvalue > 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon(np.zeros(5), np.zeros(6))


class TestMcNemar:

    def test_perfect_agreement_returns_nonsignificant(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        a = y.copy()
        b = y.copy()
        res = mcnemar(a, b, y)
        assert res.pvalue == 1.0

    def test_a_dominates_disagreement(self):
        # 10 cases: A always correct, B always wrong.
        y = np.zeros(10, dtype=bool)
        a = np.zeros(10, dtype=bool)
        b = np.ones(10, dtype=bool)
        res = mcnemar(a, b, y)
        assert res.pvalue < 0.05


class TestDieboldMariano:

    def test_a_lower_error_than_b(self):
        rng = np.random.RandomState(0)
        # A: smaller errors. B: larger.
        a = rng.normal(0, 0.1, 200)
        b = rng.normal(0, 1.0, 200)
        res = diebold_mariano(a, b)
        # A has lower MSE → statistic should be negative and significant.
        assert res.statistic < 0
        assert res.pvalue < 0.05

    def test_equal_errors_not_significant(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0, 0.5, 200)
        b = rng.normal(0, 0.5, 200)
        res = diebold_mariano(a, b)
        assert res.pvalue > 0.05


class TestPairedBootstrapCI:

    def test_ci_brackets_known_diff(self):
        from training.evaluation import compute_roc_auc

        rng = np.random.RandomState(0)
        # Synthetic: A is informative, B is random.
        n = 500
        y = rng.binomial(1, 0.3, n).astype(np.float64)
        scores_a = 0.7 * y + 0.3 * rng.uniform(0, 1, n)  # informative
        scores_b = rng.uniform(0, 1, n)                  # random
        mean_diff, lo, hi = paired_bootstrap_ci(
            compute_roc_auc, y, scores_a, scores_b, n_bootstrap=500,
        )
        # A should beat B by a wide margin.
        assert mean_diff > 0.1
        assert lo > 0  # CI bound away from zero → significant.
