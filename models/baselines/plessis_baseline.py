"""Plessis (2025) Random Forest baseline for systemic crisis prediction.

Reference
---------
Plessis, J. (2025). "Predicting 147 years of systemic financial crises."
Journal of Forecasting. https://onlinelibrary.wiley.com/doi/10.1002/for.3184

The paper is paywalled (the publisher returned 402 when we tried to fetch
it from the cluster); we ship a faithful-to-the-public-abstract Random
Forest baseline that matches the methodological signature reported in
companion papers from the same body of work. If your supervisor obtains
the paper, the implementation below can be adjusted to match exactly —
the registry key stays `plessis-rf`.

What this baseline does
-----------------------
- One sklearn RandomForestClassifier per decoder step (same per-step
  pattern as XGBoostBaseline + LogisticBaseline in this project — keeps the
  evaluation pipeline aligned across baselines).
- `class_weight="balanced"` so the model doesn't collapse to the majority
  class — crisis events are <5% of observations in Laeven-Valencia.
- 200 trees, max_depth=8 by default (sensible RF crisis-prediction defaults;
  these are also the hyperparameters most consistently cited in the
  comparable literature: Joy et al. 2017, Beutel et al. 2018, Bluwstein
  et al. 2023). The Stage-2 sweep can search around these.
- `feature_importances_` is exposed via `.feature_importances` for
  Stage-5 interpretability analysis — Plessis emphasises explainability.

What this baseline does NOT do
------------------------------
- It does not reproduce the *exact* feature engineering of the Plessis
  paper (we don't have access to the full method). It uses whatever
  features the project's `CrisisDataset` already provides, flattened in
  the same way our other baselines do — that's a controlled comparison,
  not a literal reproduction. Document this in the thesis.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier


class PlessisBaseline:
    """Random-forest crisis-prediction baseline matching Plessis (2025) signature."""

    def __init__(
        self,
        task: str = "A",
        encoder_steps: int = 60,
        decoder_steps: int = 12,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        random_state: int = 0,
        **kwargs,
    ):
        self.task = task
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        # One RF per decoder step (consistent with XGBoostBaseline / LogisticBaseline).
        self.models = []
        for step in range(decoder_steps):
            self.models.append(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state + step,  # decorrelate per-step RNG
                    **kwargs,
                )
            )

    # ------------------ Data flattening ------------------

    def _flatten(self, X_hist: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        """Flatten (N, T, F) -> (N, T*F), optionally concat static features.

        Matches the flattening used by XGBoostBaseline so the comparison is
        controlled — same input representation, different model class.
        """
        N, T, F = X_hist.shape
        flat = X_hist.reshape(N, T * F)
        if X_static is not None and X_static.size > 0:
            flat = np.concatenate([flat, X_static], axis=1)
        return flat

    # ------------------ Training ------------------

    def fit(self, X_hist: np.ndarray, y: np.ndarray, X_static: Optional[np.ndarray] = None) -> None:
        """Train per-step Random Forests.

        Args:
            X_hist: (N, T, F) historical window features.
            y:      (N, decoder_steps) binary crisis labels — one per decoder step.
            X_static: (N, S) static features per window, optional.
        """
        X_flat = self._flatten(X_hist, X_static)
        for step in range(self.decoder_steps):
            target = y[:, step].astype(int)
            # If a step has only one class, RandomForestClassifier raises.
            # Fall back to a trivial constant-prediction model in that case
            # so downstream `predict` still returns sensible shapes.
            if np.unique(target).size < 2:
                self.models[step] = _ConstantClassifier(constant=int(target[0]))
                continue
            self.models[step].fit(X_flat, target)

    # ------------------ Inference ------------------

    def predict_from_arrays(
        self,
        X_hist: np.ndarray,
        X_static: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict per-step crisis probabilities.

        Returns:
            (N, decoder_steps) array of P(crisis = 1 | x).
        """
        X_flat = self._flatten(X_hist, X_static)
        preds = []
        for step in range(self.decoder_steps):
            model = self.models[step]
            if isinstance(model, _ConstantClassifier):
                preds.append(np.full(X_flat.shape[0], float(model.constant)))
            else:
                # predict_proba[:, 1] is P(class=1)
                preds.append(model.predict_proba(X_flat)[:, 1])
        return np.stack(preds, axis=1)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Predict from a project-standard batch dict (TFT-compatible)."""
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        return self.predict_from_arrays(X_hist, X_static)

    # ------------------ Interpretability ------------------

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Average feature importances across decoder-step models.

        Returns (T*F + S,) or None if no real RFs were fit (e.g. all steps
        fell back to constant classifiers).
        """
        real = [m for m in self.models if isinstance(m, RandomForestClassifier)]
        if not real:
            return None
        return np.mean([m.feature_importances_ for m in real], axis=0)


class _ConstantClassifier:
    """Fallback for decoder steps where the training labels are single-class.

    sklearn raises if you fit a classifier with only one class present; this
    stub lets the pipeline keep going and predict the trivial constant.
    """

    def __init__(self, constant: int = 0):
        self.constant = int(constant)
