"""XGBoost baseline for crisis prediction."""

from typing import Dict, Optional

import numpy as np
import torch


class XGBoostBaseline:
    """XGBoost baseline that flattens temporal features.

    Fits one model per decoder step. For Task A uses XGBClassifier,
    for Task B uses XGBRegressor.
    """

    def __init__(
        self,
        task: str = "A",
        encoder_steps: int = 60,
        decoder_steps: int = 12,
        **xgb_kwargs,
    ):
        from xgboost import XGBClassifier, XGBRegressor

        self.task = task
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.xgb_kwargs = xgb_kwargs

        # Use GPU if available (XGBoost 2.0+ uses device="cuda" with tree_method="hist")
        try:
            import torch
            if torch.cuda.is_available():
                xgb_kwargs.setdefault("tree_method", "hist")
                xgb_kwargs.setdefault("device", "cuda")
        except ImportError:
            pass

        # One model per decoder step (kwargs override defaults)
        xgb_kwargs.setdefault("n_estimators", 100)
        xgb_kwargs.setdefault("max_depth", 5)

        self.models = []
        for _ in range(decoder_steps):
            if task == "A":
                self.models.append(XGBClassifier(
                    eval_metric="logloss",
                    **xgb_kwargs,
                ))
            else:
                self.models.append(XGBRegressor(
                    **xgb_kwargs,
                ))

    def _flatten(self, X_hist: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        """Flatten temporal features: (N, T, F) → (N, T*F), optionally concat static."""
        N = X_hist.shape[0]
        flat = X_hist.reshape(N, -1)
        if X_static is not None:
            flat = np.concatenate([flat, X_static], axis=1)
        return flat

    def fit(
        self,
        X_hist: np.ndarray,
        y: np.ndarray,
        X_static: Optional[np.ndarray] = None,
    ) -> None:
        """Fit one XGBoost model per decoder step.

        Args:
            X_hist: (N, encoder_steps, num_features) historical features.
            y: (N, decoder_steps) targets.
            X_static: (N, num_static_features) optional static features.
        """
        X_flat = self._flatten(X_hist, X_static)
        for step in range(self.decoder_steps):
            self.models[step].fit(X_flat, y[:, step])

    def predict_from_arrays(
        self,
        X_hist: np.ndarray,
        X_static: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict from numpy arrays.

        Returns:
            (N, decoder_steps) predictions.
        """
        X_flat = self._flatten(X_hist, X_static)
        preds = []
        for step in range(self.decoder_steps):
            if self.task == "A":
                preds.append(self.models[step].predict_proba(X_flat)[:, 1])
            else:
                preds.append(self.models[step].predict(X_flat))
        return np.stack(preds, axis=1)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Predict from a batch dict (same interface as TFT).

        Returns:
            (N, decoder_steps) numpy array of predictions.
        """
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        return self.predict_from_arrays(X_hist, X_static)
