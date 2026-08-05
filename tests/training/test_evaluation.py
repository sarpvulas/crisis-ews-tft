import numpy as np
import pytest
from training.evaluation import (
    compute_pr_auc,
    compute_roc_auc,
    compute_early_warning_lead_time,
    compute_calibration_error,
    compute_brier_score,
    compute_bootstrap_ci,
    PostHocCalibrator,
)


class TestPRAUC:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        auc = compute_pr_auc(y_true, y_scores)
        assert auc > 0.95

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_scores = np.random.rand(100)
        auc = compute_pr_auc(y_true, y_scores)
        assert 0.0 < auc < 1.0

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_pr_auc(y_true, y_scores)
        assert isinstance(result, float)


class TestROCAUC:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc > 0.95

    def test_inverse_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc < 0.1

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])
        assert isinstance(compute_roc_auc(y_true, y_scores), float)


class TestEarlyWarningLeadTime:
    def test_basic_lead_time(self):
        """Crisis at t=50, sustained alarm starts at t=40 → lead time ~10."""
        probabilities = np.zeros(60)
        probabilities[40:] = 0.8  # Sustained alarm from t=40 onward
        crisis_events = [50]
        lead_times = compute_early_warning_lead_time(
            probabilities, crisis_events, threshold=0.5, sustained=3
        )
        assert len(lead_times) == 1
        assert lead_times[0] == 10  # 50 - 40

    def test_no_alarm(self):
        """No alarm above threshold → empty lead times."""
        probabilities = np.full(60, 0.1)
        crisis_events = [50]
        lead_times = compute_early_warning_lead_time(
            probabilities, crisis_events, threshold=0.5, sustained=3
        )
        assert len(lead_times) == 0

    def test_not_sustained(self):
        """Alarm at t=45 but only for 2 periods (not sustained=3) → no lead time."""
        probabilities = np.zeros(60)
        probabilities[45:47] = 0.8  # Only 2 periods
        crisis_events = [50]
        lead_times = compute_early_warning_lead_time(
            probabilities, crisis_events, threshold=0.5, sustained=3
        )
        assert len(lead_times) == 0

    def test_multiple_crises(self):
        probabilities = np.zeros(120)
        probabilities[20:30] = 0.8  # Alarm for crisis at 30 (stops at onset)
        probabilities[80:90] = 0.8  # Alarm for crisis at 90
        crisis_events = [30, 90]
        lead_times = compute_early_warning_lead_time(
            probabilities, crisis_events, threshold=0.5, sustained=3
        )
        assert len(lead_times) == 2
        assert lead_times[0] == 10  # 30 - 20
        assert lead_times[1] == 10  # 90 - 80


class TestCalibrationError:
    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        np.random.seed(42)
        n = 10000
        y_scores = np.random.rand(n)
        y_true = (np.random.rand(n) < y_scores).astype(float)
        ece = compute_calibration_error(y_true, y_scores, n_bins=10)
        assert ece < 0.05

    def test_poorly_calibrated(self):
        """Extreme overconfidence should yield high ECE."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ece = compute_calibration_error(y_true, y_scores, n_bins=10)
        assert ece > 0.3

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.3, 0.7, 0.4, 0.6])
        assert isinstance(compute_calibration_error(y_true, y_scores, n_bins=5), float)


class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_scores = np.array([0.0, 0.0, 1.0, 1.0])
        assert compute_brier_score(y_true, y_scores) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_scores = np.array([1.0, 1.0, 0.0, 0.0])
        assert compute_brier_score(y_true, y_scores) == pytest.approx(1.0)

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.3, 0.7, 0.4, 0.6])
        assert isinstance(compute_brier_score(y_true, y_scores), float)


class TestPostHocCalibrator:
    @pytest.fixture
    def overconfident_data(self):
        """Simulate a model that ranks well but is overconfident."""
        np.random.seed(42)
        n = 2000
        # True probability varies smoothly
        true_prob = np.linspace(0.05, 0.95, n)
        y_true = (np.random.rand(n) < true_prob).astype(float)
        # Model is overconfident: pushes probabilities toward 0 and 1
        y_scores = np.clip(true_prob * 1.5 - 0.25, 0.01, 0.99)
        return y_scores, y_true

    def test_platt_reduces_ece(self, overconfident_data):
        y_scores, y_true = overconfident_data
        raw_ece = compute_calibration_error(y_true, y_scores)

        cal = PostHocCalibrator(method="platt")
        cal.fit(y_scores, y_true)
        y_cal = cal.transform(y_scores)

        cal_ece = compute_calibration_error(y_true, y_cal)
        assert cal_ece < raw_ece

    def test_isotonic_reduces_ece(self, overconfident_data):
        y_scores, y_true = overconfident_data
        raw_ece = compute_calibration_error(y_true, y_scores)

        cal = PostHocCalibrator(method="isotonic")
        cal.fit(y_scores, y_true)
        y_cal = cal.transform(y_scores)

        cal_ece = compute_calibration_error(y_true, y_cal)
        assert cal_ece < raw_ece

    def test_preserves_roc_auc(self, overconfident_data):
        """Calibration is monotonic — ROC-AUC should be invariant."""
        y_scores, y_true = overconfident_data
        raw_roc = compute_roc_auc(y_true, y_scores)

        for method in ("platt", "isotonic"):
            cal = PostHocCalibrator(method=method)
            cal.fit(y_scores, y_true)
            y_cal = cal.transform(y_scores)
            cal_roc = compute_roc_auc(y_true, y_cal)
            assert cal_roc == pytest.approx(raw_roc, abs=0.01)

    def test_output_range(self, overconfident_data):
        """Calibrated probabilities should be in [0, 1]."""
        y_scores, y_true = overconfident_data
        for method in ("platt", "isotonic"):
            cal = PostHocCalibrator(method=method)
            cal.fit(y_scores, y_true)
            y_cal = cal.transform(y_scores)
            assert y_cal.min() >= 0.0
            assert y_cal.max() <= 1.0

    def test_fit_transform(self, overconfident_data):
        y_scores, y_true = overconfident_data
        cal = PostHocCalibrator(method="platt")
        y_cal = cal.fit_transform(y_scores, y_true)
        assert len(y_cal) == len(y_scores)
        assert y_cal.min() >= 0.0

    def test_transform_before_fit_raises(self):
        cal = PostHocCalibrator(method="platt")
        with pytest.raises(RuntimeError, match="must be fit"):
            cal.transform(np.array([0.5]))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            PostHocCalibrator(method="invalid")

    def test_improves_brier_score(self, overconfident_data):
        y_scores, y_true = overconfident_data
        raw_brier = compute_brier_score(y_true, y_scores)

        for method in ("platt", "isotonic"):
            cal = PostHocCalibrator(method=method)
            cal.fit(y_scores, y_true)
            y_cal = cal.transform(y_scores)
            cal_brier = compute_brier_score(y_true, y_cal)
            assert cal_brier < raw_brier


class TestBootstrapCI:
    def test_returns_tuple(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_scores = np.random.rand(100)
        lower, upper = compute_bootstrap_ci(
            compute_roc_auc, y_true, y_scores, n_bootstrap=100, ci=0.95
        )
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_ci_ordering(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_scores = np.random.rand(100)
        lower, upper = compute_bootstrap_ci(
            compute_roc_auc, y_true, y_scores, n_bootstrap=100, ci=0.95
        )
        assert lower <= upper

    def test_ci_contains_point_estimate(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 200)
        y_scores = np.random.rand(200)
        point = compute_roc_auc(y_true, y_scores)
        lower, upper = compute_bootstrap_ci(
            compute_roc_auc, y_true, y_scores, n_bootstrap=500, ci=0.95
        )
        # Point estimate should typically be within CI (not always guaranteed but very likely)
        assert lower <= point + 0.1  # Loose check
        assert upper >= point - 0.1
