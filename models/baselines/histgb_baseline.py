"""Histogram gradient-boosting baseline (LightGBM-style, sklearn-native).

A strong modern gradient-boosted-tree baseline that does not require the
external LightGBM/CatBoost wheels (which are not installed in the project
environment). ``HistGradientBoostingClassifier`` is sklearn's histogram-based
GBDT and is algorithmically close to LightGBM. One classifier per decoder step,
same per-step flattening as XGBoostBaseline / PlessisBaseline so the comparison
is controlled (identical input representation, different model class).

Class imbalance is handled with balanced per-sample weights at fit time (robust
across sklearn versions, unlike the ``class_weight`` argument).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGBBaseline:
    """Histogram gradient-boosting crisis-prediction baseline."""

    def __init__(
        self,
        task: str = "A",
        encoder_steps: int = 252,
        decoder_steps: int = 63,
        max_iter: int = 300,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        random_state: int = 0,
        **kwargs,
    ):
        self.task = task
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.models = []
        for step in range(decoder_steps):
            self.models.append(
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=random_state + step,
                )
            )

    def _flatten(self, X_hist: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        N, T, F = X_hist.shape
        flat = X_hist.reshape(N, T * F)
        if X_static is not None and X_static.size > 0:
            flat = np.concatenate([flat, X_static], axis=1)
        return flat

    def fit(self, X_hist: np.ndarray, y: np.ndarray, X_static: Optional[np.ndarray] = None) -> None:
        X_flat = self._flatten(X_hist, X_static)
        for step in range(self.decoder_steps):
            target = y[:, step].astype(int)
            if np.unique(target).size < 2:
                self.models[step] = _ConstantClassifier(constant=int(target[0]))
                continue
            # balanced sample weights: w_c = n / (2 * n_c)
            n = target.size
            pos = max(int(target.sum()), 1)
            neg = max(n - pos, 1)
            w_pos, w_neg = n / (2.0 * pos), n / (2.0 * neg)
            sw = np.where(target == 1, w_pos, w_neg)
            self.models[step].fit(X_flat, target, sample_weight=sw)

    def predict_from_arrays(self, X_hist: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        X_flat = self._flatten(X_hist, X_static)
        preds = []
        for step in range(self.decoder_steps):
            model = self.models[step]
            if isinstance(model, _ConstantClassifier):
                preds.append(np.full(X_flat.shape[0], float(model.constant)))
            else:
                preds.append(model.predict_proba(X_flat)[:, 1])
        return np.stack(preds, axis=1)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        return self.predict_from_arrays(X_hist, X_static)


class _ConstantClassifier:
    def __init__(self, constant: int = 0):
        self.constant = int(constant)
