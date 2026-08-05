"""TDTCrisisModel — CrisisModel adapter for the Temporal Diffusion Transformer.

The standard TorchCrisisModel adapter assumes the inner module produces crisis
logits that are scored with BCE. TDT's training objective is the DDPM
noise-prediction loss, not BCE — so we replace the .fit() method with a
diffusion-aware training loop. Inference (`predict_proba`) calls
`TDT.crisis_score` and then applies Platt scaling fitted on the validation
split, returning calibrated probabilities aligned with the project's
evaluation harness.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.base import CrisisModel, FitResult, register_model
from models.tdt import TDT, TDTConfig


class TDTCrisisModel(CrisisModel):
    """Adapter that drives diffusion training and conditional-likelihood scoring.

    Stage-1.J upgrade: the model now optimises a *joint* objective
        L = L_diffusion + λ_aux · L_regime + λ_clf · L_BCE(class_head, ews)
    and the training loader uses a class-balanced WeightedRandomSampler so
    the crisis-conditional path gets gradient updates at the same rate as
    the calm path (instead of the natural 13% / 87%).
    """

    name = "tdt"

    def __init__(
        self,
        module: TDT,
        lambda_aux: float = 0.1,
        lambda_clf: float = 1.0,
        lambda_diff: float = 1.0,
        pos_weight: float = 7.0,
        balanced_sampler: bool = True,
    ):
        self.module = module
        self.lambda_aux = lambda_aux
        self.lambda_clf = lambda_clf
        self.lambda_diff = lambda_diff  # 0 = no diffusion regularizer (Stage-4 ablation)
        self.pos_weight = pos_weight
        self.balanced_sampler = balanced_sampler  # False = vanilla shuffled loader (Stage-4 ablation)
        # Filled in by fit(): Platt calibrator + min/max for normalisation.
        self._calibrator = None

    # ------------------ training ------------------

    def fit(self, train_dataset, val_dataset, train_config: Dict[str, Any]) -> FitResult:
        device = train_config.get("device", "cpu")
        epochs = int(train_config.get("epochs", 50))
        lr = float(train_config.get("lr", 1e-3))
        batch_size = int(train_config.get("batch_size", 64))
        grad_clip = float(train_config.get("grad_clip", 1.0))
        ckpt_dir = train_config.get("checkpoint_dir", "checkpoints")
        patience = int(train_config.get("early_stopping_patience", 8))
        os.makedirs(ckpt_dir, exist_ok=True)

        self.module.to(device)
        opt = Adam(self.module.parameters(), lr=lr)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

        # Class-balanced sampling for training so the crisis-conditional path
        # gets gradients as often as calm. With ~13% positives in Task B,
        # balanced sampling roughly 4x's the gradient signal on positives.
        if self.balanced_sampler:
            train_loader = self._make_balanced_loader(train_dataset, batch_size)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=train_dataset.collate_fn,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float("inf")
        bad = 0
        t0 = time.time()

        for epoch in range(epochs):
            self.module.train()
            n, total = 0, 0.0
            for batch, targets in train_loader:
                x0 = batch["historical_ts_numeric"].to(device)
                B = x0.size(0)
                c = self._extract_condition(targets, B, device)
                out = self.module.diffusion_step(x0, c)
                loss = self.lambda_diff * out["loss_simple"]

                # Auxiliary regime classification (cheap inductive bias).
                if self.lambda_aux > 0 and "regime_label" in targets:
                    rl = targets["regime_label"].to(device)
                    rl_window = rl[:, -1]
                    aux = F.cross_entropy(out["regime_logits"], rl_window)
                    loss = loss + self.lambda_aux * aux

                # Discriminative head — joint diffusion + classification.
                # We run a second forward pass on the *clean* input (t=0,
                # c=calm) so the class_head reads the same features the
                # backbone has learned during diffusion. The clean-input
                # forward is essentially `class_head_logit(x0)` but with
                # gradients enabled.
                if self.lambda_clf > 0 and self.module.class_head is not None:
                    t_clean = torch.zeros(B, device=device, dtype=torch.long)
                    c_neutral = torch.zeros(B, device=device, dtype=torch.long)
                    clean_out = self.module.denoise(x0, t_clean, c_neutral)
                    class_logit = clean_out["class_logit"]
                    y_window = self._window_label(targets, B, device)
                    pos_w = torch.tensor([self.pos_weight], device=device)
                    l_clf = F.binary_cross_entropy_with_logits(
                        class_logit, y_window, pos_weight=pos_w
                    )
                    loss = loss + self.lambda_clf * l_clf

                opt.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)
                opt.step()
                total += float(loss.item())
                n += 1
            train_loss = total / max(n, 1)

            val_loss = self._validate_diffusion(val_loader, device)
            sched.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(opt.param_groups[0]["lr"])
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {opt.param_groups[0]['lr']:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                bad = 0
                self._save_inner(os.path.join(ckpt_dir, "best_model.pt"))
            else:
                bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Fit Platt scaler on val so predict_proba returns calibrated probs.
        self._fit_calibrator(val_dataset, device, batch_size)

        return FitResult(
            history=history,
            best_val_loss=best_val,
            wall_time_s=time.time() - t0,
            final_epoch=len(history["val_loss"]),
        )

    # ------------------ helpers ------------------

    def _make_balanced_loader(self, train_dataset, batch_size: int) -> DataLoader:
        """Return a DataLoader whose sampler oversamples positive windows.

        For each window we materialise the binary "any-positive-in-decoder"
        label and weight inversely by class frequency, so each batch is
        roughly 50/50 by construction.
        """
        n = len(train_dataset)
        # Pull window-level labels via the dataset's known per-window indices.
        # We rely on CrisisDataset's exposed arrays for speed.
        if hasattr(train_dataset, "ews_labels") and hasattr(train_dataset, "valid_indices"):
            ews = train_dataset.ews_labels
            enc = train_dataset.encoder_steps
            dec = train_dataset.decoder_steps
            window_pos = []
            for start in train_dataset.valid_indices:
                end = start + enc + dec
                window_pos.append(int(ews[start + enc : end].max() > 0.5))
            window_pos = torch.tensor(window_pos, dtype=torch.long)
        else:
            # Wrapped dataset (e.g. TopologyAugmentedDataset) — fall back to
            # materialising labels via __getitem__. Slower but safe.
            window_pos = torch.zeros(n, dtype=torch.long)
            for i in range(n):
                _, t = train_dataset[i]
                window_pos[i] = int(t["ews_label"].max() > 0.5)
        n_pos = int(window_pos.sum().item())
        n_neg = int(n - n_pos)
        if n_pos == 0 or n_neg == 0:
            # Degenerate — fall back to shuffled loader.
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn)
        w_pos = 0.5 / max(n_pos, 1)
        w_neg = 0.5 / max(n_neg, 1)
        weights = torch.where(window_pos.bool(),
                              torch.tensor(w_pos), torch.tensor(w_neg))
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights.double(), num_samples=n, replacement=True)
        return DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=train_dataset.collate_fn,
        )

    @staticmethod
    def _window_label(targets, B: int, device: str) -> torch.Tensor:
        """Per-window binary positive flag from decoder-step ews_label."""
        if "ews_label" not in targets:
            return torch.zeros(B, device=device)
        ews = targets["ews_label"].to(device)
        return (ews.max(dim=1).values > 0.5).float()

    def _extract_condition(
        self, targets: Dict[str, torch.Tensor], B: int, device: str
    ) -> torch.Tensor:
        """Choose the per-window conditioning label.

        With `condition_mode='ews'` (default), conditioning is the binary
        max(ews_label) over the decoder window, mapped to {calm=0, crisis=K-1}.
        That trains p(x | future_crisis=0) and p(x | future_crisis=1), so the
        inference-time ratio `log p(x|crisis) - log p(x|calm)` is *exactly* the
        log-odds for "crisis in the next decoder_steps days" — the prediction
        target.

        With `condition_mode='regime'`, conditioning is the regime label at
        the encoder/decoder boundary (the Stage-1 MVP framing). Kept for the
        ablation comparison the thesis will reference.

        Falls back to all-zeros (= calm) if neither label is present.
        """
        mode = self.module.config.condition_mode
        K = self.module.config.num_regime_states
        if mode == "ews":
            if "ews_label" not in targets:
                return torch.zeros(B, dtype=torch.long, device=device)
            ews = targets["ews_label"].to(device)  # (B, decoder_steps)
            # Window-level positive: any positive ews step in the decoder window.
            binary = (ews.max(dim=1).values > 0.5).long()
            # Map 0 -> calm (id=0); 1 -> crisis (id=K-1). With K=3 we skip the
            # 'stress' intermediate id during conditioning — that slot just
            # doesn't receive gradient updates, which is fine.
            return binary * (K - 1)
        # condition_mode == "regime"
        if "regime_label" not in targets:
            return torch.zeros(B, dtype=torch.long, device=device)
        rl = targets["regime_label"].to(device)  # (B, enc + dec)
        idx = rl.size(1) // 2 - 1
        return rl[:, idx]

    @torch.no_grad()
    def _validate_diffusion(self, val_loader: DataLoader, device: str) -> float:
        self.module.eval()
        n, total = 0, 0.0
        for batch, targets in val_loader:
            x0 = batch["historical_ts_numeric"].to(device)
            c = self._extract_condition(targets, x0.size(0), device)
            out = self.module.diffusion_step(x0, c)
            total += float(out["loss_simple"].item())
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _raw_scores(self, dataset, device: str, batch_size: int = 32) -> np.ndarray:
        """Return per-window raw scores. Source picked by config.predict_via.

        "class_head"       — class_head logit (discriminative). Fast.
        "likelihood_ratio" — crisis_score (VLB or MSE-ratio). Generative.
        """
        self.module.eval()
        self.module.to(device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn)
        out = []
        for batch, _ in loader:
            x0 = batch["historical_ts_numeric"].to(device)
            if (self.module.config.predict_via == "class_head"
                    and self.module.class_head is not None):
                s = self.module.class_head_logit(x0).cpu().numpy()
            else:
                s = self.module.crisis_score(x0).cpu().numpy()
            out.append(s)
        return np.concatenate(out)

    @torch.no_grad()
    def _fit_calibrator(self, val_dataset, device: str, batch_size: int) -> None:
        """Fit Platt scaling: raw TDT scores -> calibrated crisis probabilities.

        We use the dataset's ews_label (binary per timestep) — averaged over the
        decoder window to get one positive-rate per window — as the target.
        """
        from training.evaluation import PostHocCalibrator

        raw = self._raw_scores(val_dataset, device, batch_size)
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=val_dataset.collate_fn)
        labels = []
        for _, targets in loader:
            y = targets["ews_label"].numpy()
            # Any positive in the decoder window -> window-level positive.
            labels.append((y.max(axis=1) > 0.5).astype(np.float64))
        y_true = np.concatenate(labels)

        # Normalise raw scores into [0, 1] via min-max for Platt input. Platt's
        # logistic regression handles any monotonic input, but a bounded range
        # keeps the fit numerically clean.
        if raw.max() > raw.min():
            normed = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            normed = np.full_like(raw, 0.5)

        cal = PostHocCalibrator(method="platt")
        if len(np.unique(y_true)) >= 2:
            cal.fit(normed, y_true)
            self._calibrator = (cal, raw.min(), raw.max())
        else:
            # Degenerate: only one class — keep identity mapping.
            self._calibrator = None

    # ------------------ inference ------------------

    @torch.no_grad()
    def predict_proba(self, dataset) -> np.ndarray:
        device = next(self.module.parameters()).device
        raw = self._raw_scores(dataset, str(device))
        decoder_steps = self.module.config.decoder_steps
        if self._calibrator is None:
            # Unfit calibrator -> sigmoid as a default monotonic mapping.
            probs = 1.0 / (1.0 + np.exp(-raw))
        else:
            cal, lo, hi = self._calibrator
            normed = (raw - lo) / max(hi - lo, 1e-12)
            normed = np.clip(normed, 0.0, 1.0)
            probs = cal.transform(normed)
        # Broadcast per-window probability across decoder steps so the result
        # aligns with the standard (n_windows * decoder_steps,) evaluation shape.
        return np.repeat(probs, decoder_steps)

    # ------------------ save/load ------------------

    def _save_inner(self, path: str) -> None:
        torch.save({
            "state_dict": self.module.state_dict(),
            "calibrator": self._calibrator,
        }, path)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._save_inner(path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.module.load_state_dict(ckpt["state_dict"])
        self._calibrator = ckpt.get("calibrator")

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters() if p.requires_grad))


# ------------------------- Registration -------------------------

@register_model("tdt")
def _make_tdt(config: Dict[str, Any], feature_dims: Dict[str, int]) -> CrisisModel:
    cfg = TDTConfig(
        num_historical_numeric=feature_dims["num_historical_numeric"],
        num_future_numeric=feature_dims.get("num_future_numeric", 0),
        encoder_steps=config.get("encoder_steps", 252),
        decoder_steps=config.get("decoder_steps", 63),
        d_model=config.get("d_model", config.get("hidden_size", 128)),
        n_heads=config.get("attention_heads", 4),
        e_layers=config.get("e_layers", config.get("lstm_layers", 4)),
        d_ff=config.get("d_ff", 256),
        dropout=config.get("dropout", 0.1),
        diffusion_steps=config.get("diffusion_steps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
        cond_drop_prob=config.get("cond_drop_prob", 0.1),
        num_regime_states=config.get("num_regime_states", 3),
        lambda_aux=config.get("lambda_aux", 0.1),
        use_class_head=config.get("use_class_head", True),
        predict_via=config.get("predict_via", "class_head"),
        condition_mode=config.get("condition_mode", "ews"),
    )
    module = TDT(cfg)
    return TDTCrisisModel(
        module,
        lambda_aux=cfg.lambda_aux,
        lambda_clf=config.get("lambda_clf", 1.0),
        lambda_diff=config.get("lambda_diff", 1.0),
        pos_weight=config.get("pos_weight", 7.0),
        balanced_sampler=config.get("balanced_sampler", True),
    )
