"""iTransformer for crisis prediction (Liu et al., ICLR 2024).

The "i" in iTransformer is "inverted": instead of treating each timestep as a
token (standard transformer), each *variate* (feature column) is a token. The
T-dimensional time series of one variate is projected to d_model, and attention
operates across variates. This gives the model an explicit view of inter-
variate dependencies that standard channel-mixing transformers learn only
implicitly via dot-product attention over (timestep × feature) pairs.

For crisis prediction (binary classification per decoder step) the original
forecasting head is replaced with a small MLP that maps the pooled variate
representation to a `decoder_steps` logit vector.

Citation:
  Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
  iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
  ICLR 2024. arXiv:2310.06625
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ITransformerConfig:
    num_historical_numeric: int
    num_future_numeric: int = 0
    encoder_steps: int = 252
    decoder_steps: int = 63
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    pool: str = "mean"           # "mean" or "cls"


class _VariateEmbedding(nn.Module):
    """Project each variate's T-dim time series to d_model.

    Input:  (B, T, F)
    Output: (B, F, d_model) — F variate tokens of width d_model
    """

    def __init__(self, encoder_steps: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(encoder_steps, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  ->  (B, F, T)  ->  (B, F, d_model)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return self.drop(x)


class _EncoderLayer(nn.Module):
    """One transformer encoder block with pre-LayerNorm.

    Attention is across variates (sequence axis = F), so the model learns
    inter-variate dependencies directly. There is no position encoding because
    variates are inherently unordered.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h))
        return x


class ITransformer(nn.Module):
    """iTransformer adapted to crisis-prediction binary classification.

    The forward returns the project-wide convention:
      {"predicted": (B, decoder_steps, 1)} — logits over the decoder horizon.
    No regime head, so the auxiliary regime loss in CrisisAwareLoss must be
    disabled (gamma=0) when training this model.
    """

    def __init__(self, config: ITransformerConfig):
        super().__init__()
        self.config = config
        self.embed = _VariateEmbedding(config.encoder_steps, config.d_model, config.dropout)
        self.layers = nn.ModuleList([
            _EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.e_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.decoder_steps),
        )
        if config.pool not in ("mean", "cls"):
            raise ValueError(f"unknown pool '{config.pool}'")
        if config.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["historical_ts_numeric"]  # (B, T, F)
        tokens = self.embed(x)              # (B, F, d_model)

        if self.config.pool == "cls":
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        if self.config.pool == "cls":
            pooled = tokens[:, 0]
        else:
            pooled = tokens.mean(dim=1)     # (B, d_model)

        logits = self.head(pooled)          # (B, decoder_steps)
        return {"predicted": logits.unsqueeze(-1)}  # (B, decoder_steps, 1)
