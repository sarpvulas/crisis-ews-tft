"""Logistic Regression / Ridge baseline for crisis prediction."""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge


class LogisticBaseline:
    """Logistic Regression baseline that flattens temporal features.

    For Task A: LogisticRegression per decoder step.
    For Task B: Ridge Regression per decoder step.
    """

    def __init__(
        self,
        task: str = "A",
        encoder_steps: int = 60,
        decoder_steps: int = 12,
        **kwargs,
    ):
        self.task = task
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps

        self.models = []
        for _ in range(decoder_steps):
            if task == "A":
                self.models.append(LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    **kwargs,
                ))
            else:
                self.models.append(Ridge(**kwargs))

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
        """Fit one model per decoder step."""
        X_flat = self._flatten(X_hist, X_static)
        for step in range(self.decoder_steps):
            self.models[step].fit(X_flat, y[:, step])

    def predict_from_arrays(
        self,
        X_hist: np.ndarray,
        X_static: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict from numpy arrays."""
        X_flat = self._flatten(X_hist, X_static)
        preds = []
        for step in range(self.decoder_steps):
            if self.task == "A":
                preds.append(self.models[step].predict_proba(X_flat)[:, 1])
            else:
                preds.append(self.models[step].predict(X_flat))
        return np.stack(preds, axis=1)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Predict from a batch dict (same interface as TFT)."""
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        return self.predict_from_arrays(X_hist, X_static)
