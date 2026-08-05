"""Temporal Diffusion Transformer (TDT) — the main novel architecture.

Thesis contribution
-------------------
Crisis prediction is hard because real crises are rare (~50 events globally in
130 years of Laeven-Valencia data), heavy-tailed, and regime-structured.
Standard discriminative models trained from scratch on the crisis label
data-starve. TDT inverts the problem:

  1. Train a *generative* model of macro-financial trajectories conditioned on
     the regime (calm / stress / crisis) using a DDPM-style diffusion process.
  2. At inference, score a new window by the conditional likelihood ratio
     log p(x | c=crisis) − log p(x | c=calm). The ratio is a continuous,
     calibratable crisis score that exploits the *abundant* normal data to
     learn what "calm" looks like, rather than struggling to discriminate
     from the handful of crisis events.

Architecture
------------
  - Variate-token embedding (iTransformer-style): each macro variable becomes
    a token whose feature is its T-step history. Attention operates across
    variables, learning inter-feature structure that XGBoost's tree splits
    capture but standard temporal-attention transformers miss.
  - AdaLN-Zero modulation (from DiT, Peebles & Xie 2023): the diffusion
    timestep `t` and the conditioning `c` are projected to per-layer scale +
    shift + residual-gate parameters, applied at LayerNorm. Cleaner than
    concat-and-hope.
  - Classifier-free guidance: during training, the condition is randomly
    dropped to a learned "unconditional" embedding 10% of the time. At
    inference, this enables guided sampling and — critical for our use case —
    the conditional-likelihood-ratio score described above.

Training
--------
  Standard DDPM (Ho et al. 2020) with the simplified noise-prediction loss
  L_simple = E[ || eps − eps_theta(x_t, t, c) ||^2 ].
  Optional auxiliary loss: regime classification head on the pooled token
  representation, weighted by `lambda_aux` (small; serves as inductive bias).

Inference for crisis prediction
-------------------------------
  For each window, we approximate the conditional log-likelihood by the
  expected reconstruction MSE under random noise levels (a tractable proxy
  for the VLB). The crisis score is the MSE ratio (or its log) between the
  two conditionings. A monotonic calibration (Platt) fit on validation maps
  the score to a probability. This is the MVP scheme; we can swap in a
  proper probability-flow ODE likelihood in a later iteration if calibration
  isn't tight enough.

Outputs follow the project convention:
  forward(batch) → {"predicted": (B, decoder_steps, 1)}
where "predicted" is a logit derived from the conditional likelihood-ratio
score. `CrisisAwareLoss` consumes it the same as any other model — we
override `fit` on the wrapper to drive diffusion training instead of BCE,
keeping the wrapper interface honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Config -----------------------------

@dataclass
class TDTConfig:
    num_historical_numeric: int
    num_future_numeric: int = 0
    encoder_steps: int = 252
    decoder_steps: int = 63

    # Architecture
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # Diffusion
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"     # "linear" or "cosine"
    cond_drop_prob: float = 0.1       # classifier-free guidance dropout

    # Conditioning
    num_regime_states: int = 3
    use_topology_input: bool = False  # filled by TopologyAugmentedDataset
    num_topology_dims: int = 0

    # Auxiliary
    lambda_aux: float = 0.1           # weight on regime classification aux loss

    # Inference scoring
    #   "vlb"        — variational lower-bound weighted by Ho-2020 per-timestep
    #                  weights. Theoretically grounded: each timestep contributes
    #                  to the log-likelihood proportionally to its KL term.
    #                  Recommended default.
    #   "mse_ratio"  — Stage-1 MVP: uniformly-averaged MSE difference. Cheap
    #                  but ignores how much each timestep contributes to the
    #                  actual log-likelihood. Kept for ablation.
    score_method: str = "vlb"
    # Timesteps used at inference. For score_method=='vlb' we use a stratified
    # linspace covering the full noise schedule (low-noise timesteps carry
    # more signal). For "mse_ratio" we use the original hardcoded list.
    score_timesteps: Tuple[int, ...] = field(default_factory=lambda: (100, 250, 500, 750))
    score_num_timesteps: int = 16     # used when score_method=='vlb'
    score_repeats: int = 4            # number of noise draws to average per timestep

    # Causal-DAG attention prior (Option A integration).
    # If `use_causal_mask` is True, the model expects a (K, F, F) adjacency
    # tensor passed in at construction time. The adjacency for the active
    # regime of each batch element is added to attention logits in every block.
    use_causal_mask: bool = False
    causal_mask_scale: float = 1.0    # multiplier on the mask before adding to logits

    # Conditioning signal source — what does the classifier-free guidance condition on?
    #   "ews"    — max(ews_label) over the decoder window. Crisis-aligned: trains
    #              p(x | future_crisis=0) and p(x | future_crisis=1). At inference
    #              the conditional-likelihood ratio directly answers "will a crisis
    #              happen in the next decoder_steps?". Default — matches what we
    #              actually predict.
    #   "regime" — regime_label at encoder/decoder boundary. Earlier framing:
    #              p(x | current_regime). The ratio answers "is the present a
    #              crisis-regime window?", which is a proxy at best. Kept as an
    #              ablation knob and for backwards compatibility.
    condition_mode: str = "ews"

    # Discriminative head (Stage 1.J).
    # `use_class_head=True` (default) adds a binary classification head that
    # reads from the same final-layer variate-token features as the noise-
    # prediction head. The model now optimises a joint diffusion + BCE loss
    # and predict_proba returns the head's sigmoid by default (much faster
    # than the conditional-likelihood ratio, and discriminatively trained).
    use_class_head: bool = True
    # Inference output: "class_head" (sigmoid of class_head logit, fast +
    # discriminatively trained) or "likelihood_ratio" (the VLB / MSE-ratio
    # scoring kept for the ablation comparison in the thesis).
    predict_via: str = "class_head"


# ---------------------- Diffusion schedule ------------------------

def _cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from Nichol & Dhariwal (2021).

    Returns alpha_bar[0..T] of shape (T+1,) where alpha_bar[0]=1.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    return (f / f[0]).to(torch.float32)


def _make_schedule(T: int, kind: str) -> Dict[str, torch.Tensor]:
    if kind == "cosine":
        alpha_bar = _cosine_alpha_bar(T)
    elif kind == "linear":
        betas = torch.linspace(1e-4, 0.02, T + 1, dtype=torch.float32)
        alpha = 1 - betas
        alpha_bar = torch.cumprod(alpha, dim=0)
    else:
        raise ValueError(f"unknown beta_schedule '{kind}'")
    # Clamp to numerical safety.
    alpha_bar = alpha_bar.clamp(min=1e-6, max=1.0)
    # Derived quantities used by both training and VLB scoring.
    # alpha_t = alpha_bar_t / alpha_bar_{t-1}; beta_t = 1 - alpha_t.
    alpha = alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-12)
    alpha = torch.cat([torch.tensor([1.0]), alpha], dim=0)  # pad t=0
    beta = (1.0 - alpha).clamp(min=1e-12)
    # Posterior variance sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    one_minus_ab = (1.0 - alpha_bar).clamp(min=1e-12)
    one_minus_ab_prev = torch.cat([torch.tensor([0.0]), 1.0 - alpha_bar[:-1]], dim=0)
    sigma2 = (beta * one_minus_ab_prev / one_minus_ab).clamp(min=1e-12)
    # Ho-2020 simplified VLB weight per timestep:
    #   w_t = beta_t^2 / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))
    # This is the coefficient on ||eps - eps_theta||^2 in the per-timestep KL.
    vlb_weight = beta.pow(2) / (2.0 * sigma2 * alpha.clamp(min=1e-12) * one_minus_ab)
    # Clamp absurdly large weights (happens at extreme t for numerically-rough
    # schedules); doesn't change the qualitative weighting.
    vlb_weight = vlb_weight.clamp(max=1e6)
    return {
        "alpha_bar": alpha_bar,                    # (T+1,)
        "sqrt_alpha_bar": alpha_bar.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
        "vlb_weight": vlb_weight,                  # (T+1,)
    }


# ----------------------- Embeddings -------------------------------

class _SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding of the diffusion timestep, followed by an MLP."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        self.dim = dim
        h = hidden or dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.out_dim = h

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)  # (B, h)


class _ConditionEmbed(nn.Module):
    """Embed (regime, optional aux) into a fixed-width vector.

    A learned "unconditional" embedding is appended at index num_regime_states;
    `forward(c, mask)` swaps to it whenever `mask[i] == True` for the i-th batch
    element — this is how classifier-free guidance is implemented.
    """

    def __init__(self, num_regime_states: int, out_dim: int):
        super().__init__()
        self.num_regime_states = num_regime_states
        self.embed = nn.Embedding(num_regime_states + 1, out_dim)
        self.uncond_idx = num_regime_states

    def forward(self, c: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            c = torch.where(mask, torch.full_like(c, self.uncond_idx), c)
        return self.embed(c)


# ---------------------- AdaLN-Zero block --------------------------

class _AdaLNZeroBlock(nn.Module):
    """Transformer encoder block with adaptive LayerNorm-Zero modulation (DiT).

    The conditioning vector produces (scale, shift, gate) for each of the two
    LayerNorms, plus a residual gate. At init the gate is zero so the block
    starts as identity — critical for stable diffusion training.

    Optional per-sample attention bias (Option A): when `attn_bias` of shape
    (B, F, F) is passed to forward, it's repeated per-head and added to
    attention logits before softmax. This is how the causal-DAG mask makes
    it into the transformer without losing per-batch fidelity.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, cond_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        # 6 = (scale, shift, gate) for each of (attn, ffn)
        self.cond_proj = nn.Linear(cond_dim, 6 * d_model)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    @staticmethod
    def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); scale, shift: (B, D)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _expand_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """Expand (B, F, F) -> (B*num_heads, F, F) for nn.MultiheadAttention."""
        B, F1, F2 = bias.shape
        return bias.unsqueeze(1).expand(B, self.n_heads, F1, F2).reshape(B * self.n_heads, F1, F2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = self.cond_proj(cond)  # (B, 6*D)
        scale1, shift1, gate1, scale2, shift2, gate2 = params.chunk(6, dim=-1)
        h = self._modulate(self.ln1(x), scale1, shift1)
        attn_mask = None if attn_bias is None else self._expand_bias(attn_bias)
        a, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + gate1.unsqueeze(1) * self.drop(a)
        h = self._modulate(self.ln2(x), scale2, shift2)
        x = x + gate2.unsqueeze(1) * self.drop(self.ffn(h))
        return x


# --------------------------- TDT model ----------------------------

class TDT(nn.Module):
    """Temporal Diffusion Transformer.

    forward(batch) is in *inference* mode: it samples noise levels, predicts
    eps under both crisis and calm conditioning, computes the per-window MSE
    ratio, and returns a logit suitable for binary-cross-entropy loss against
    crisis labels. The diffusion training step is in `diffusion_step` —
    `TDTModel.fit` (in registry.py) calls that directly.
    """

    def __init__(self, config: TDTConfig):
        super().__init__()
        self.config = config

        # Embedding: each variate's T-step history -> d_model token
        self.variate_proj = nn.Linear(config.encoder_steps, config.d_model)

        # Timestep + condition embeddings
        self.t_embed = _SinusoidalTimeEmbed(config.d_model)
        self.c_embed = _ConditionEmbed(config.num_regime_states, self.t_embed.out_dim)

        # Stacked AdaLN-Zero blocks
        self.blocks = nn.ModuleList([
            _AdaLNZeroBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                cond_dim=self.t_embed.out_dim,
            )
            for _ in range(config.e_layers)
        ])

        # Output: predict noise of shape (B, T, F). We project each variate
        # token back to T values, then transpose.
        self.final_ln = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.encoder_steps)

        # Optional auxiliary regime classifier on the pooled token.
        self.aux_head = nn.Linear(config.d_model, config.num_regime_states)

        # Stage-1.J discriminative head — single logit per window. Reads from
        # the mean-pooled final-layer variate tokens. The eps-prediction head
        # writes to the same features, so backprop through L_clf shapes the
        # backbone to be discriminatively useful too.
        if config.use_class_head:
            self.class_head = nn.Linear(config.d_model, 1)
        else:
            self.class_head = None

        # Buffers for the noise schedule.
        sched = _make_schedule(config.diffusion_steps, config.beta_schedule)
        for k, v in sched.items():
            self.register_buffer(k, v, persistent=False)

        # Pre-resolve score timesteps to the valid range. If the user supplied
        # values that exceed `diffusion_steps`, scale them proportionally so a
        # quick smoke run with diffusion_steps=50 still works.
        valid = []
        for ts in config.score_timesteps:
            if ts <= 0:
                continue
            if ts <= config.diffusion_steps:
                valid.append(int(ts))
            else:
                valid.append(max(1, int(ts * config.diffusion_steps / 1000)))
        if not valid:
            valid = [max(1, config.diffusion_steps // 2)]
        self._score_timesteps = tuple(sorted(set(valid)))

        # Causal-DAG attention prior buffer (Option A). Filled in by
        # `set_causal_adjacency()` after construction; until then the mask is
        # disabled even if `config.use_causal_mask = True`.
        self.register_buffer(
            "_causal_adjacency",
            torch.zeros(config.num_regime_states, config.num_historical_numeric,
                        config.num_historical_numeric),
            persistent=False,
        )
        self._causal_loaded = False

    def set_causal_adjacency(self, adjacency: torch.Tensor) -> None:
        """Install a (K, F, F) regime-conditioned adjacency as attention prior.

        Must be called before training if `config.use_causal_mask=True`.
        Adjacency is expected to be normalised (e.g. row-wise) by the caller.
        """
        if not self.config.use_causal_mask:
            raise RuntimeError("set_causal_adjacency called but config.use_causal_mask is False")
        K, F1, F2 = adjacency.shape
        if K != self.config.num_regime_states:
            raise ValueError(f"adjacency K={K} mismatches num_regime_states={self.config.num_regime_states}")
        if F1 != self.config.num_historical_numeric or F2 != self.config.num_historical_numeric:
            raise ValueError(f"adjacency F=({F1},{F2}) mismatches num_historical_numeric={self.config.num_historical_numeric}")
        self._causal_adjacency = adjacency.to(self._causal_adjacency.device, dtype=torch.float32)
        self._causal_loaded = True

    def _attn_bias_for_condition(self, c: torch.Tensor) -> Optional[torch.Tensor]:
        """Pick adjacency[c] per batch element. Returns None when disabled."""
        if not (self.config.use_causal_mask and self._causal_loaded):
            return None
        # adjacency: (K, F, F); c: (B,) -> (B, F, F)
        bias = self._causal_adjacency.index_select(0, c.clamp_min(0))
        return bias * self.config.causal_mask_scale

    # ------------------ core denoiser ------------------

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Predict noise eps from (x_t, t, c).

        x_t: (B, T, F) noised window
        t  : (B,) long, in [0, diffusion_steps]
        c  : (B,) long regime id in [0, num_regime_states)
        cond_mask: (B, 1) bool; True positions get the unconditional embedding.
        Returns: {"eps_pred": (B, T, F), "regime_logits": (B, num_regime_states)}
        """
        # Variate-token embedding: (B, T, F) -> (B, F, d_model)
        h = self.variate_proj(x_t.transpose(1, 2))

        # Conditioning vector (scalar per batch element)
        if cond_mask is None:
            cond_mask_flat = None
        else:
            cond_mask_flat = cond_mask.squeeze(-1).bool()
        cond = self.t_embed(t.float()) + self.c_embed(c, cond_mask_flat)

        # Causal-DAG attention bias keyed by the *true* regime label `c`
        # (not the cond-masked one — even when CFG drops conditioning, the
        # structural prior still reflects the regime the window is actually in).
        attn_bias = self._attn_bias_for_condition(c)

        for block in self.blocks:
            h = block(h, cond, attn_bias=attn_bias)

        h = self.final_ln(h)                       # (B, F, d_model)
        eps_pred = self.out_proj(h).transpose(1, 2)  # (B, T, F)
        pooled = h.mean(dim=1)                       # (B, d_model)
        regime_logits = self.aux_head(pooled)
        out = {"eps_pred": eps_pred, "regime_logits": regime_logits, "pooled": pooled}
        if self.class_head is not None:
            out["class_logit"] = self.class_head(pooled).squeeze(-1)  # (B,)
        return out

    # ------------------ diffusion training step ------------------

    def diffusion_step(self, x_0: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One training-step forward.

        x_0: (B, T, F) clean window
        c  : (B,) long regime id

        Returns dict with:
          "loss_simple"  : DDPM simple noise-prediction loss
          "regime_logits": optional aux head output
          "eps_pred"     : predicted noise (for debugging)
        """
        B = x_0.size(0)
        device = x_0.device
        t = torch.randint(1, self.config.diffusion_steps + 1, (B,), device=device)

        eps = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_omab * eps

        # Classifier-free guidance: randomly drop the condition to "unconditional"
        cond_mask = torch.rand(B, 1, device=device) < self.config.cond_drop_prob
        out = self.denoise(x_t, t, c, cond_mask)

        loss_simple = F.mse_loss(out["eps_pred"], eps)
        return {
            "loss_simple": loss_simple,
            "regime_logits": out["regime_logits"],
            "eps_pred": out["eps_pred"],
        }

    # ------------------ inference: crisis score ------------------

    def _stratified_timesteps(self) -> torch.Tensor:
        """Stratified linspace over [1, diffusion_steps]. Used by VLB scoring
        so we cover the full noise schedule instead of four hardcoded points."""
        n = max(2, int(self.config.score_num_timesteps))
        T = self.config.diffusion_steps
        # Avoid t=0 (no noise) and ensure t<=T.
        timesteps = torch.linspace(1, T, n).round().long().clamp(min=1, max=T)
        # Deduplicate after rounding (cheap; ordering preserved).
        seen, uniq = set(), []
        for t in timesteps.tolist():
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return torch.tensor(uniq, dtype=torch.long)

    @torch.no_grad()
    def _conditional_neg_log_lik(
        self, x_0: torch.Tensor, c_id: int, score_timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Return a VLB-weighted approximate -log p(x_0 | c). Shape: (B,).

        This is the sum over selected timesteps of weight(t) * E_eps[||eps - eps_theta||^2],
        which is proportional to the per-timestep KL term in the diffusion ELBO.
        The constant log p(x_T) drops out of the ratio score so we ignore it.
        """
        B = x_0.size(0)
        device = x_0.device
        total = torch.zeros(B, device=device)
        for t_int in score_timesteps.tolist():
            for _ in range(self.config.score_repeats):
                t = torch.full((B,), int(t_int), device=device, dtype=torch.long)
                eps = torch.randn_like(x_0)
                sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
                sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
                x_t = sqrt_ab * x_0 + sqrt_omab * eps
                c = torch.full((B,), int(c_id), device=device, dtype=torch.long)
                eps_pred = self.denoise(x_t, t, c)["eps_pred"]
                mse = (eps - eps_pred).pow(2).mean(dim=(1, 2))
                w = self.vlb_weight[t_int]
                total = total + w * mse
        return total / self.config.score_repeats

    @torch.no_grad()
    def crisis_score(self, x_0: torch.Tensor, calm_id: int = 0, crisis_id: int = 2) -> torch.Tensor:
        """Per-window log-likelihood-ratio score: log p(x | crisis) − log p(x | calm).

        Two methods (selected by `config.score_method`):
          "vlb"       — proper VLB-weighted contribution per timestep with
                        stratified sampling across the full noise schedule.
                        Theoretically grounded; recommended default.
          "mse_ratio" — uniformly-weighted MSE difference at a small fixed set
                        of timesteps. The Stage-1 MVP, kept for ablation.

        A positive score means crisis conditioning explains the window better
        (lower noise reconstruction error → higher conditional likelihood).
        Caller calibrates via Platt scaling on validation labels.
        """
        if self.config.score_method == "vlb":
            ts = self._stratified_timesteps()
            neg_ll_calm = self._conditional_neg_log_lik(x_0, calm_id, ts)
            neg_ll_crisis = self._conditional_neg_log_lik(x_0, crisis_id, ts)
            # log p(x|crisis) - log p(x|calm) = -neg_ll_crisis - (-neg_ll_calm)
            return neg_ll_calm - neg_ll_crisis
        elif self.config.score_method == "mse_ratio":
            B = x_0.size(0)
            device = x_0.device
            scores = torch.zeros(B, device=device)
            for t_int in self._score_timesteps:
                for _ in range(self.config.score_repeats):
                    t = torch.full((B,), int(t_int), device=device, dtype=torch.long)
                    eps = torch.randn_like(x_0)
                    sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
                    sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
                    x_t = sqrt_ab * x_0 + sqrt_omab * eps
                    c_calm = torch.full((B,), calm_id, device=device, dtype=torch.long)
                    c_crisis = torch.full((B,), crisis_id, device=device, dtype=torch.long)
                    eps_calm = self.denoise(x_t, t, c_calm)["eps_pred"]
                    eps_crisis = self.denoise(x_t, t, c_crisis)["eps_pred"]
                    mse_calm = (eps - eps_calm).pow(2).mean(dim=(1, 2))
                    mse_crisis = (eps - eps_crisis).pow(2).mean(dim=(1, 2))
                    scores = scores + (mse_calm - mse_crisis)
            n = len(self._score_timesteps) * self.config.score_repeats
            return scores / n
        else:
            raise ValueError(f"unknown score_method '{self.config.score_method}'")

    # ------------------ project-protocol forward ------------------

    @torch.no_grad()
    def class_head_logit(self, x_0: torch.Tensor) -> torch.Tensor:
        """Run the discriminative class head on a clean (un-noised) window.

        The class head reads from the same backbone as the eps-prediction head.
        We pass `t=0` (which means alpha_bar=1, no noise added) and use the
        true regime label only via the `c` index — but since `cond_proj` was
        zero-initialised at the start of training and the AdaLN-Zero gates
        learn from gradient signal during fit(), the backbone has had to
        produce useful features anyway. Returns (B,) crisis logits.
        """
        if self.class_head is None:
            raise RuntimeError("class_head is disabled; set use_class_head=True")
        B = x_0.size(0)
        device = x_0.device
        t = torch.zeros(B, device=device, dtype=torch.long)
        # We don't have access to the true label at inference, so we pass
        # `c=calm` to get a deterministic "what would the model produce in the
        # calm-conditioned state for a clean input" representation. The class
        # head reads the features and decides whether they look like a crisis
        # window regardless of conditioning — this is the discriminative path.
        c = torch.zeros(B, device=device, dtype=torch.long)
        return self.denoise(x_0, t, c)["class_logit"]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inference: returns {"predicted": (B, decoder_steps, 1)} of crisis logits.

        Path selected by `config.predict_via`:
          "class_head"        — sigmoid(self.class_head(features)). Discrim
                                trained. Fast (one forward pass per window).
                                Default in Stage-1.J and beyond.
          "likelihood_ratio"  — log p(x|crisis) − log p(x|calm) via VLB / MSE
                                scoring. The original generative-classifier
                                framing. Kept for the thesis ablation.

        Either way, the same scalar score is broadcast across decoder_steps —
        TDT treats each window as a single "is the next window pre-crisis"
        event, matching how the eval harness flattens labels.
        """
        x = batch["historical_ts_numeric"]      # (B, T, F)
        if self.config.predict_via == "class_head" and self.class_head is not None:
            logit = self.class_head_logit(x)    # (B,)
        else:
            logit = self.crisis_score(x)        # (B,)
        logit = logit.unsqueeze(-1).unsqueeze(-1)
        return {"predicted": logit.expand(-1, self.config.decoder_steps, 1)}
