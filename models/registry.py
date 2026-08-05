"""Registers existing and future architectures against the CrisisModel interface.

Two adapter classes wrap the existing concrete implementations:

  - TorchCrisisModel: wraps any nn.Module whose forward(batch_dict) returns a
    dict containing "predicted" (the project's existing convention). Training
    delegates to `training.trainer.Trainer`. Inference delegates to
    `training.evaluation.evaluate_model`.

  - SklearnCrisisModel: wraps the flat-feature XGBoost / Logistic baselines.
    Training extracts (X_hist, X_static, y) once from the dataset and calls
    `.fit(...)`. Inference calls `.predict(batch_dict)`.

Novel architectures (Stage 1) will subclass one of these adapters and add
their own forward / fit logic, then register themselves via @register_model.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import CrisisModel, FitResult, register_model


# ---------- Torch adapter ----------

class TorchCrisisModel(CrisisModel):
    """Adapter for any nn.Module whose forward(batch_dict) -> {"predicted": ...}.

    Subclasses just need to construct the inner module in __init__ and (optionally)
    override `predict_proba` if they don't follow the EWS-logit convention.
    """

    name = "torch-base"

    def __init__(self, module: nn.Module, loss_fn: nn.Module, name: str,
                 predict_from: str = "clf"):
        self.module = module
        self.loss_fn = loss_fn
        self.name = name
        self.mc_samples = 0  # >1 enables MC-dropout averaging at inference
        # "clf" = score with the sigmoid of the classification head (default).
        # "aux" = score with the auxiliary drawdown-regression head, mapped to a
        # probability by a 1-D logistic fitted on the TRAINING set (see
        # _fit_aux_mapping). Used by the regression-only ablation.
        self.predict_from = predict_from
        self._aux_map = None

    def fit(self, train_dataset, val_dataset, train_config: Dict[str, Any]) -> FitResult:
        from training.trainer import Trainer  # local to avoid cycle

        trainer = Trainer(
            model=self.module,
            config={**train_config, "predict_from": self.predict_from},
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_fn=self.loss_fn,
            use_wandb=train_config.get("use_wandb", False),
            wandb_config=train_config.get("wandb_config"),
        )
        epochs = int(train_config.get("epochs", 50))
        t0 = time.time()
        history = trainer.train(epochs=epochs)
        wall = time.time() - t0
        # A regression-only model emits a drawdown, not a probability. Learn the
        # monotone map from predicted drawdown to crisis probability on the
        # TRAIN split — never on val (which selects the checkpoint) or test.
        # Fitting it rather than hardcoding a sign matters: the drawdown/label
        # relationship reverses between crisis-excluded (train/val) and all-window
        # (test) populations, so an assumed direction could bury a real signal.
        if self.predict_from == "aux":
            self._fit_aux_mapping(train_dataset)
        best = float(min(history["val_loss"])) if history.get("val_loss") else float("nan")
        return FitResult(
            history=history,
            best_val_loss=best,
            wall_time_s=wall,
            final_epoch=len(history.get("val_loss", [])),
        )

    @torch.no_grad()
    def _head_outputs(self, dataset, key: str) -> np.ndarray:
        """Flattened raw outputs of one head over a dataset."""
        device = next(self.module.parameters()).device
        self.module.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)
        out = []
        for batch_dict, _targets in loader:
            batch_dev = {k: v.to(device) for k, v in batch_dict.items()}
            out.append(self.module(batch_dev)[key].squeeze(-1).reshape(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def _fit_aux_mapping(self, train_dataset) -> None:
        from sklearn.linear_model import LogisticRegression

        x = self._head_outputs(train_dataset, "aux_pred").reshape(-1, 1)
        loader = DataLoader(train_dataset, batch_size=len(train_dataset),
                            collate_fn=train_dataset.collate_fn)
        _batch, targets = next(iter(loader))
        y = targets["ews_label"].numpy().reshape(-1)
        if np.unique(y).size < 2 or x.shape[0] != y.shape[0]:
            self._aux_map = None
            return
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(x, y)
        self._aux_map = lr
        print(f"  aux->prob map fitted on train: coef={lr.coef_.ravel()[0]:+.3f} "
              f"intercept={lr.intercept_[0]:+.3f} "
              f"(sign learned, not assumed)")

    @torch.no_grad()
    def predict_proba(self, dataset) -> np.ndarray:
        # Regression-only models score through the drawdown head instead.
        if self.predict_from == "aux":
            x = self._head_outputs(dataset, "aux_pred").reshape(-1, 1)
            if self._aux_map is None:
                # Fallback when train had a single class: monotone squash centred
                # on the 10% labelling threshold. Rank-preserving, so PR/ROC are
                # unaffected by the constants.
                return 1.0 / (1.0 + np.exp(-((-x.reshape(-1) - 0.10) * 20.0)))
            return self._aux_map.predict_proba(x)[:, 1].astype(np.float64)

        device = next(self.module.parameters()).device
        mc = int(getattr(self, "mc_samples", 0) or 0)
        # MC-dropout: keep dropout stochastic and average N passes for
        # uncertainty-aware (better-calibrated) probabilities.
        self.module.train() if mc > 1 else self.module.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)
        preds = []
        for batch_dict, _targets in loader:
            batch_dev = {k: v.to(device) for k, v in batch_dict.items()}
            if mc > 1:
                ps = [torch.sigmoid(self.module(batch_dev)["predicted"].squeeze(-1)) for _ in range(mc)]
                prob = torch.stack(ps, 0).mean(0)
            else:
                prob = torch.sigmoid(self.module(batch_dev)["predicted"].squeeze(-1))
            preds.append(prob.cpu().numpy().reshape(-1))
        return np.concatenate(preds)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.module.state_dict(), "name": self.name}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.module.load_state_dict(ckpt["state_dict"])

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters() if p.requires_grad))


# ---------- Sklearn-style adapter ----------

class SklearnCrisisModel(CrisisModel):
    """Adapter for flat-feature baselines (XGBoost, Logistic Regression).

    These have .fit(X_hist, y, X_static=None) and .predict(batch_dict) -> (N, dec).
    """

    name = "sklearn-base"

    def __init__(self, inner, name: str):
        self.inner = inner
        self.name = name

    def _materialise(self, dataset):
        """Pull the whole dataset into numpy. OK because baselines need flat features anyway."""
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)
        batch, targets = next(iter(loader))
        X_hist = batch["historical_ts_numeric"].numpy()
        X_static = batch.get("static_feats_numeric")
        if X_static is not None:
            X_static = X_static.numpy()
        y = targets["ews_label"].numpy()
        return X_hist, X_static, y

    def fit(self, train_dataset, val_dataset, train_config: Dict[str, Any]) -> FitResult:
        X_hist, X_static, y = self._materialise(train_dataset)
        t0 = time.time()
        self.inner.fit(X_hist, y, X_static=X_static)
        wall = time.time() - t0
        return FitResult(history={}, best_val_loss=float("nan"), wall_time_s=wall, final_epoch=1)

    def predict_proba(self, dataset) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)
        batch, _ = next(iter(loader))
        # inner.predict(batch_dict) -> (N, decoder_steps) probabilities for Task A,
        # quantile predictions for Task B. For EWS we always want sigmoid-bounded
        # probabilities — the baselines already return them for task="A".
        preds = self.inner.predict(batch)
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"inner": self.inner, "name": self.name}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.inner = blob["inner"]

    @property
    def n_params(self) -> int:
        # Rough proxy: total leaves across boosters / coefficients.
        try:
            return int(sum(getattr(m, "n_estimators", 0) for m in self.inner.models))
        except Exception:
            return 0


# ---------- Registrations for existing architectures ----------

@register_model("tft")
def _make_tft(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    from models.configs import TFTConfig
    from models.tft import TemporalFusionTransformer
    from models.crisis_loss import CrisisAwareLoss

    cfg = TFTConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_static_numeric=feature_dims.get("num_static_numeric", 0),
        num_static_categorical=feature_dims.get("num_static_categorical", 0),
        static_categorical_cardinalities=feature_dims.get("static_categorical_cardinalities", []),
        num_future_numeric=feature_dims["num_future_numeric"],
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        task_type="ews",
        num_outputs=1,
        state_size=config.get("state_size", 64),
        hidden_size=config.get("hidden_size", 64),
        attention_heads=config.get("attention_heads", 4),
        lstm_layers=config.get("lstm_layers", 2),
        dropout=config.get("dropout", 0.1),
        num_regime_states=config.get("num_regime_states", 3),
        use_regime_module=False,
        use_regime_attention=False,
        use_aux=config.get("use_aux", False),
        use_regime_vsn=config.get("use_regime_vsn", False),
    )
    module = TemporalFusionTransformer(cfg)
    # no regime aux for vanilla TFT; loss_type lets the loss-ablation sweep tft too.
    # beta=0 drops the classification term entirely (regression-only ablation).
    loss = CrisisAwareLoss(
        beta=config.get("beta", 1.0),
        gamma=0.0,
        pos_weight=config.get("pos_weight", 10.0),
        loss_type=config.get("loss_type", "bce"),
        focal_gamma=config.get("focal_gamma", 2.0),
        lambda_aux=config.get("lambda_aux", 0.5),
    )
    return TorchCrisisModel(module, loss, name="tft",
                            predict_from=config.get("predict_from", "clf"))


@register_model("ra-tft")
def _make_ra_tft(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    from models.configs import TFTConfig
    from models.ra_tft import RegimeAwareTFT
    from models.crisis_loss import CrisisAwareLoss

    cfg = TFTConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_static_numeric=feature_dims.get("num_static_numeric", 0),
        num_static_categorical=feature_dims.get("num_static_categorical", 0),
        static_categorical_cardinalities=feature_dims.get("static_categorical_cardinalities", []),
        num_future_numeric=feature_dims["num_future_numeric"],
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        task_type="ews",
        num_outputs=1,
        state_size=config.get("state_size", 64),
        hidden_size=config.get("hidden_size", 64),
        attention_heads=config.get("attention_heads", 4),
        lstm_layers=config.get("lstm_layers", 2),
        dropout=config.get("dropout", 0.1),
        num_regime_states=config.get("num_regime_states", 3),
        use_regime_module=config.get("use_regime_module", True),
        use_regime_attention=config.get("use_regime_attention", True),
        use_aux=config.get("use_aux", False),
        use_regime_vsn=config.get("use_regime_vsn", False),
    )
    module = RegimeAwareTFT(cfg)
    loss = CrisisAwareLoss(
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 0.3),
        pos_weight=config.get("pos_weight", 10.0),
        loss_type=config.get("loss_type", "bce"),
        focal_gamma=config.get("focal_gamma", 2.0),
        lambda_aux=config.get("lambda_aux", 0.5),
    )
    return TorchCrisisModel(module, loss, name="ra-tft",
                            predict_from=config.get("predict_from", "clf"))


@register_model("lstm")
def _make_lstm(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    from models.baselines.lstm_baseline import LSTMBaseline
    from models.crisis_loss import CrisisAwareLoss

    module = LSTMBaseline(
        input_dim=feature_dims["num_historical_numeric"],
        hidden_dim=config.get("hidden_size", 64),
        num_layers=config.get("lstm_layers", 2),
        dropout=config.get("dropout", 0.1),
        task="B",
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        num_quantiles=1,
    )
    # LSTM has no regime head — gamma=0 so the loss reduces to BCE.
    loss = CrisisAwareLoss(gamma=0.0)
    return TorchCrisisModel(module, loss, name="lstm")


@register_model("xgboost")
def _make_xgb(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    from models.baselines.xgboost_baseline import XGBoostBaseline

    inner = XGBoostBaseline(
        task="A",  # always classification for EWS, despite this being Task-B-named
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 5),
        learning_rate=config.get("xgb_lr", 0.1),
    )
    return SklearnCrisisModel(inner, name="xgboost")


@register_model("logistic")
def _make_logistic(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    from models.baselines.logistic_baseline import LogisticBaseline

    inner = LogisticBaseline(
        task="A",
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
    )
    return SklearnCrisisModel(inner, name="logistic")


@register_model("histgb")
def _make_histgb(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    """Histogram gradient boosting (LightGBM-style, sklearn-native) baseline."""
    from models.baselines.histgb_baseline import HistGBBaseline

    inner = HistGBBaseline(
        task=config.get("task", "A"),
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        max_iter=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", None),
        learning_rate=config.get("xgb_lr", 0.05),
        random_state=config.get("seed", 0),
    )
    return SklearnCrisisModel(inner, name="histgb")


@register_model("plessis-rf")
def _make_plessis_rf(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    """Plessis (2025) random-forest baseline — published crisis-prediction benchmark."""
    from models.baselines.plessis_baseline import PlessisBaseline

    inner = PlessisBaseline(
        task=config.get("task", "A"),
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 8),
        min_samples_leaf=config.get("min_samples_leaf", 5),
        random_state=config.get("seed", 0),
    )
    return SklearnCrisisModel(inner, name="plessis-rf")


# ---------- Stage 1 strong baselines (literature SOTA) ----------

@register_model("itransformer")
def _make_itransformer(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    """iTransformer (Liu et al. ICLR 2024) — variates-as-tokens transformer.

    The regime head is absent here; gamma is forced to 0 so CrisisAwareLoss
    reduces to BCE on crisis labels.
    """
    from models.itransformer import ITransformer, ITransformerConfig
    from models.crisis_loss import CrisisAwareLoss

    cfg = ITransformerConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_future_numeric=feature_dims.get("num_future_numeric", 0),
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        d_model=config.get("d_model", config.get("hidden_size", 128)),
        n_heads=config.get("attention_heads", 4),
        e_layers=config.get("e_layers", config.get("lstm_layers", 2)),
        d_ff=config.get("d_ff", 256),
        dropout=config.get("dropout", 0.1),
        pool=config.get("pool", "mean"),
    )
    module = ITransformer(cfg)
    loss = CrisisAwareLoss(
        beta=config.get("beta", 1.0),
        gamma=0.0,
        pos_weight=config.get("pos_weight", 10.0),
    )
    return TorchCrisisModel(module, loss, name="itransformer")


# ---------- Stage 1 novel architectures ----------

@register_model("topo-ratft")
def _make_topo_ratft(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    """RA-TFT with persistence-image topology features appended to historical channels.

    Feature_dims is expected to already include the inflated historical channel
    count from TopologyAugmentedDataset.get_feature_dims(). The model is otherwise
    identical to ra-tft.
    """
    from models.configs import TFTConfig
    from models.ra_tft import RegimeAwareTFT
    from models.crisis_loss import CrisisAwareLoss

    cfg = TFTConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_static_numeric=feature_dims.get("num_static_numeric", 0),
        num_future_numeric=feature_dims["num_future_numeric"],
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        task_type="ews",
        num_outputs=1,
        state_size=config.get("state_size", 64),
        hidden_size=config.get("hidden_size", 64),
        attention_heads=config.get("attention_heads", 4),
        lstm_layers=config.get("lstm_layers", 2),
        dropout=config.get("dropout", 0.1),
        num_regime_states=3,
        use_regime_module=True,
        use_regime_attention=True,
    )
    module = RegimeAwareTFT(cfg)
    loss = CrisisAwareLoss(
        beta=config.get("beta", 1.0),
        gamma=config.get("gamma", 0.3),
        pos_weight=config.get("pos_weight", 10.0),
    )
    return TorchCrisisModel(module, loss, name="topo-ratft")
