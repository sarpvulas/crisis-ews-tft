"""Unified model interface for crisis prediction experiments.

Every architecture (TFT, RA-TFT, LSTM, XGBoost, iTransformer, novel architectures...)
implements `CrisisModel`. This decouples the experiment harness from model internals
so `scripts/run_experiment.py --model X` works for any registered architecture.

Two model families are supported:
  - Torch models: train via the existing `training.trainer.Trainer` loop.
  - Sklearn-style models (XGBoost, Logistic Regression): train via `.fit(X, y)`.

The wrapper hides this distinction. Downstream code only needs `.fit()`,
`.predict_proba()`, `.save()`, `.load()`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class FitResult:
    """Returned by CrisisModel.fit().

    `history` is per-epoch metrics (torch models) or empty (sklearn-style).
    `best_val_loss` is the lowest seen value of the trainer's val loss; for
    sklearn-style models it is np.nan because they fit in one shot.
    """

    history: Dict[str, list] = field(default_factory=dict)
    best_val_loss: float = float("nan")
    wall_time_s: float = 0.0
    final_epoch: int = 0


class CrisisModel(ABC):
    """Abstract interface every crisis-prediction architecture implements.

    Implementations are registered via `register_model(name)` and built via
    `build_model(name, config, feature_dims)`.

    Subclasses must override:
        - fit(train_dataset, val_dataset, train_config) -> FitResult
        - predict_proba(dataset) -> np.ndarray of shape (n_samples,) with crisis probabilities
        - save(path) / load(path)
        - n_params (property)

    Subclasses should set `self.name` in __init__ for logging.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train_dataset,
        val_dataset,
        train_config: Dict[str, Any],
    ) -> FitResult:
        ...

    @abstractmethod
    def predict_proba(self, dataset) -> np.ndarray:
        """Return crisis probabilities aligned with `dataset` iteration order.

        For sequence models, the convention is to flatten (n_windows * decoder_steps,)
        the same way `training.evaluation.evaluate_model` does — so a returned vector
        can be paired directly with `labels` for ROC/PR/calibration.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Trainable parameter count. For non-torch models, return a sensible proxy
        (e.g. n_features * n_trees for XGB, or 0 if it's not meaningful)."""
        ...


# ---------- Registry ----------

_REGISTRY: Dict[str, Callable[..., CrisisModel]] = {}


def register_model(name: str) -> Callable[[Callable[..., CrisisModel]], Callable[..., CrisisModel]]:
    """Decorator to register a model factory under a canonical name.

    Usage:
        @register_model("itransformer")
        def _make_itransformer(config, feature_dims):
            return ITransformerCrisisModel(config, feature_dims)
    """

    def decorator(factory: Callable[..., CrisisModel]) -> Callable[..., CrisisModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_model(name: str, config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    """Construct a registered model by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](config, feature_dims)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
