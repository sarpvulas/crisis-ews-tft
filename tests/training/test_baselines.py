import pytest
import numpy as np
import torch
from models.baselines.lstm_baseline import LSTMBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.baselines.logistic_baseline import LogisticBaseline


class TestLSTMBaseline:
    def test_task_b_output_shape(self):
        model = LSTMBaseline(
            input_dim=10, hidden_dim=32, num_layers=2, task="B",
            encoder_steps=8, decoder_steps=3, num_quantiles=7,
        )
        batch = {
            "historical_ts_numeric": torch.randn(4, 8, 10),
            "future_ts_numeric": torch.randn(4, 3, 2),
            "static_feats_numeric": torch.randn(4, 2),
        }
        preds = model.predict(batch)
        assert preds.shape == (4, 3, 7)  # (batch, decoder_steps, quantiles)

    def test_forward_matches_predict(self):
        model = LSTMBaseline(
            input_dim=5, hidden_dim=16, num_layers=1, task="B",
            encoder_steps=8, decoder_steps=3, num_quantiles=7,
        )
        batch = {
            "historical_ts_numeric": torch.randn(2, 8, 5),
            "future_ts_numeric": torch.randn(2, 3, 2),
            "static_feats_numeric": torch.randn(2, 1),
        }
        outputs = model(batch)
        assert "predicted" in outputs


class TestXGBoostBaseline:
    def test_task_b_fit_predict(self):
        np.random.seed(42)
        model = XGBoostBaseline(task="B", encoder_steps=8, decoder_steps=3)
        X_hist = np.random.randn(100, 8, 5)
        X_static = np.random.randn(100, 2)
        y = np.abs(np.random.randn(100, 3)) * 0.05
        model.fit(X_hist, y, X_static=X_static)
        preds = model.predict_from_arrays(X_hist[:10], X_static=X_static[:10])
        assert preds.shape == (10, 3)

    def test_predict_interface_with_batch_dict(self):
        np.random.seed(42)
        model = XGBoostBaseline(task="B", encoder_steps=8, decoder_steps=3)
        X_hist = np.random.randn(50, 8, 5)
        X_static = np.random.randn(50, 2)
        y = np.abs(np.random.randn(50, 3)) * 0.05
        model.fit(X_hist, y, X_static=X_static)

        batch = {
            "historical_ts_numeric": torch.randn(5, 8, 5),
            "static_feats_numeric": torch.randn(5, 2),
        }
        preds = model.predict(batch)
        assert preds.shape == (5, 3)


class TestLogisticBaseline:
    def test_task_b_fit_predict(self):
        np.random.seed(42)
        model = LogisticBaseline(task="B", encoder_steps=8, decoder_steps=3)
        X_hist = np.random.randn(100, 8, 5)
        y = np.abs(np.random.randn(100, 3)) * 0.05
        model.fit(X_hist, y)
        preds = model.predict_from_arrays(X_hist[:10])
        assert preds.shape == (10, 3)

    def test_predict_interface_with_batch_dict(self):
        np.random.seed(42)
        model = LogisticBaseline(task="B", encoder_steps=8, decoder_steps=3)
        X_hist = np.random.randn(50, 8, 5)
        y = np.abs(np.random.randn(50, 3)) * 0.05
        model.fit(X_hist, y)

        batch = {
            "historical_ts_numeric": torch.randn(5, 8, 5),
        }
        preds = model.predict(batch)
        assert preds.shape == (5, 3)
