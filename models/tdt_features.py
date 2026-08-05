"""TDT-features → XGBoost hybrid (Stage 1.K fallback).

If TDT's class_head proves too seed-sensitive at the full scale, this hybrid
treats the diffusion-trained backbone as a fixed feature extractor and lets
gradient boosting do the discrimination. The thesis claim becomes:

  *"Diffusion-pretrained variate-token representations contain crisis-relevant
   signal that a gradient-boosted classifier can exploit, outperforming both
   the supervised baselines on raw features and the end-to-end-trained TDT."*

Pipeline:
  1. Train a vanilla TDT (or load a checkpoint) — for the diffusion + class
     head joint loss to shape the backbone.
  2. At inference, run each window through the backbone at t=0, c=calm, and
     extract the mean-pooled (B, d_model) features.
  3. Feed those features into XGBoost (one model per decoder step, same
     as XGBoostBaseline). Train + predict like any sklearn baseline.

Implementation: a thin wrapper that holds (a) a frozen TDT module and (b)
an XGBoostBaseline. Registered as `tdt-xgb`.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import CrisisModel, FitResult, register_model
from models.tdt import TDT, TDTConfig
from models.tdt_adapter import TDTCrisisModel


class TDTFeatureExtractorXGB(CrisisModel):
    """TDT backbone (frozen after diffusion training) → XGBoost head."""

    name = "tdt-xgb"

    def __init__(self, tdt_module: TDT, xgb_kwargs: Dict[str, Any]):
        self.tdt = tdt_module
        self.xgb_kwargs = xgb_kwargs
        self._xgb = None  # built in fit() after we know feature dim

    # ------------- training -------------

    def fit(self, train_dataset, val_dataset, train_config: Dict[str, Any]) -> FitResult:
        """Two-phase fit: (1) diffusion+class joint training on TDT, (2) XGBoost on features."""
        device = train_config.get("device", "cpu")
        t0 = time.time()

        # Phase 1: train the TDT backbone with joint loss.
        # We piggyback on the existing TDTCrisisModel.fit so we get the same
        # balanced sampler + joint loss treatment.
        tdt_wrapper = TDTCrisisModel(
            self.tdt,
            lambda_aux=0.1,
            lambda_clf=1.0,
            lambda_diff=1.0,
            pos_weight=7.0,
            balanced_sampler=True,
        )
        tdt_wrapper.fit(train_dataset, val_dataset, train_config)

        # Phase 2: extract features, fit XGBoost on them.
        from models.baselines.xgboost_baseline import XGBoostBaseline
        X_train_feat, y_train = self._extract(train_dataset, device)
        # XGBoostBaseline expects (N, T, F) historical + (N, decoder_steps) labels.
        # We have flat features (N, d_model) — treat as a single timestep of d_model features.
        N, D = X_train_feat.shape
        X_train_3d = X_train_feat.reshape(N, 1, D)
        self._xgb = XGBoostBaseline(
            task="A",
            encoder_steps=1,
            decoder_steps=y_train.shape[1],
            **self.xgb_kwargs,
        )
        self._xgb.fit(X_train_3d, y_train)
        return FitResult(
            history={},
            best_val_loss=float("nan"),
            wall_time_s=time.time() - t0,
            final_epoch=1,
        )

    # ------------- feature extraction -------------

    @torch.no_grad()
    def _extract(self, dataset, device: str) -> tuple:
        """Run each window through TDT backbone, return (N, d_model) + (N, dec)."""
        self.tdt.eval()
        self.tdt.to(device)
        loader = DataLoader(
            dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn,
        )
        feats, labels = [], []
        for batch, targets in loader:
            x0 = batch["historical_ts_numeric"].to(device)
            B = x0.size(0)
            t = torch.zeros(B, device=device, dtype=torch.long)
            c = torch.zeros(B, device=device, dtype=torch.long)
            out = self.tdt.denoise(x0, t, c)
            feats.append(out["pooled"].cpu().numpy())
            labels.append(targets["ews_label"].numpy())
        return np.concatenate(feats), np.concatenate(labels)

    # ------------- inference -------------

    def predict_proba(self, dataset) -> np.ndarray:
        device = next(self.tdt.parameters()).device
        feats, _ = self._extract(dataset, str(device))
        N, D = feats.shape
        probs = self._xgb.predict_from_arrays(feats.reshape(N, 1, D))
        return probs.reshape(-1)

    # ------------- persistence -------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"tdt_state": self.tdt.state_dict()}, path + ".tdt.pt")
        with open(path + ".xgb.pkl", "wb") as f:
            pickle.dump(self._xgb, f)

    def load(self, path: str) -> None:
        blob = torch.load(path + ".tdt.pt", map_location="cpu")
        self.tdt.load_state_dict(blob["tdt_state"])
        with open(path + ".xgb.pkl", "rb") as f:
            self._xgb = pickle.load(f)

    @property
    def n_params(self) -> int:
        # TDT params + ~12K for XGB trees.
        return int(sum(p.numel() for p in self.tdt.parameters() if p.requires_grad)) + 12_600


@register_model("tdt-xgb")
def _make_tdt_xgb(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    cfg = TDTConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_future_numeric=feature_dims.get("num_future_numeric", 0),
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        d_model=config.get("d_model", 64),
        n_heads=config.get("attention_heads", 2),
        e_layers=config.get("e_layers", 4),
        d_ff=config.get("d_ff", 256),
        dropout=config.get("dropout", 0.05),
        diffusion_steps=config.get("diffusion_steps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
        cond_drop_prob=config.get("cond_drop_prob", 0.0),
        num_regime_states=config.get("num_regime_states", 3),
        lambda_aux=config.get("lambda_aux", 0.1),
        use_class_head=True,
        predict_via="class_head",
        condition_mode=config.get("condition_mode", "ews"),
    )
    tdt_module = TDT(cfg)
    return TDTFeatureExtractorXGB(
        tdt_module,
        xgb_kwargs={
            "n_estimators": config.get("n_estimators", 200),
            "max_depth": config.get("max_depth", 5),
        },
    )
