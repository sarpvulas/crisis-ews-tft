"""Tests for the Plessis 2025 random-forest baseline."""

import numpy as np
import pytest
import torch

from models.baselines.plessis_baseline import PlessisBaseline, _ConstantClassifier


def _make_synthetic(n_windows: int = 80, T: int = 20, F: int = 5, dec: int = 4, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_hist = rng.randn(n_windows, T, F).astype(np.float32)
    # Make label informative so the RF actually learns something.
    signal = X_hist[:, -1, 0] + 0.5 * X_hist[:, -1, 1]
    y_window = (signal > signal.mean()).astype(int)
    y = np.tile(y_window[:, None], (1, dec))
    return X_hist, y


class TestPlessisBaseline:

    def test_fit_predict_shape(self):
        X, y = _make_synthetic()
        model = PlessisBaseline(encoder_steps=20, decoder_steps=4, n_estimators=20, max_depth=4)
        model.fit(X, y)
        probs = model.predict_from_arrays(X)
        assert probs.shape == (X.shape[0], 4)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_dict_interface(self):
        X, y = _make_synthetic()
        model = PlessisBaseline(encoder_steps=20, decoder_steps=4, n_estimators=20, max_depth=4)
        model.fit(X, y)
        batch = {"historical_ts_numeric": torch.tensor(X)}
        probs = model.predict(batch)
        assert probs.shape == (X.shape[0], 4)

    def test_constant_classifier_fallback(self):
        """Steps where all training labels are one class should fall back to a
        constant predictor — sklearn RFs can't fit single-class targets."""
        X, _ = _make_synthetic()
        y_all_zero = np.zeros((X.shape[0], 4), dtype=int)
        model = PlessisBaseline(encoder_steps=20, decoder_steps=4, n_estimators=20)
        model.fit(X, y_all_zero)
        for step_model in model.models:
            assert isinstance(step_model, _ConstantClassifier)
            assert step_model.constant == 0
        probs = model.predict_from_arrays(X)
        np.testing.assert_array_equal(probs, 0.0)

    def test_class_balance_via_class_weight(self):
        """Heavily imbalanced labels (5% positives) should still produce a
        non-degenerate ranking — class_weight='balanced' is doing its job."""
        rng = np.random.RandomState(0)
        n, T, F = 400, 12, 4
        X = rng.randn(n, T, F).astype(np.float32)
        signal = X[:, -1, 0]
        # ~5% positives, but tied to signal so RF can rank
        y_window = (signal > np.percentile(signal, 95)).astype(int)
        y = np.tile(y_window[:, None], (1, 2))
        model = PlessisBaseline(encoder_steps=T, decoder_steps=2, n_estimators=80, max_depth=6)
        model.fit(X, y)
        probs = model.predict_from_arrays(X)
        # The model should give measurably higher probability to the positives.
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[:, 0], probs[:, 0])
        assert auc > 0.75, f"expected >0.75 ROC-AUC, got {auc:.3f}"

    def test_feature_importances_after_fit(self):
        X, y = _make_synthetic()
        model = PlessisBaseline(encoder_steps=20, decoder_steps=4, n_estimators=20, max_depth=4)
        model.fit(X, y)
        imp = model.feature_importances
        assert imp is not None
        assert imp.shape == (20 * 5,)  # T*F = 100
        assert (imp >= 0).all()
        assert imp.sum() == pytest.approx(1.0, abs=1e-3)

    def test_feature_importances_none_if_no_real_models(self):
        X, _ = _make_synthetic()
        y = np.zeros((X.shape[0], 4), dtype=int)
        model = PlessisBaseline(encoder_steps=20, decoder_steps=4, n_estimators=20)
        model.fit(X, y)
        assert model.feature_importances is None


class TestRegistry:

    def test_plessis_registers_and_builds(self):
        import models.registry  # noqa: F401
        from models.base import build_model, list_models

        assert "plessis-rf" in list_models()
        m = build_model("plessis-rf", {"decoder_steps": 4, "encoder_steps": 20, "seed": 0},
                        {"num_historical_numeric": 5, "num_future_numeric": 0})
        assert hasattr(m, "fit")
        assert hasattr(m, "predict_proba")
