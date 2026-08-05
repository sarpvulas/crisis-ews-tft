import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention with shared V. Paper Eq. 7-9.

    Key difference from standard MHA: V is shared across heads and attention
    weights are averaged (not concatenated), making them directly interpretable.
    Supports optional additive attn_bias for regime conditioning.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Per-head Q, K projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # Shared V projection (single head dim, not embed_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        # Output projection
        self.out_proj = nn.Linear(self.head_dim, embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, tgt_len, _ = q.shape
        src_len = k.shape[1]

        # Project Q, K to per-head space
        q = self.q_proj(q).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K shape: (batch, heads, seq, head_dim)

        # Project V (shared across heads)
        v = self.v_proj(v)  # (batch, src_len, head_dim)

        # Attention scores per head
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, tgt, src)

        # Apply regime bias if provided
        if attn_bias is not None:
            # attn_bias: (batch, tgt_len, src_len) -> broadcast across heads
            attn_logits = attn_logits + attn_bias.unsqueeze(1)

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (batch, heads, tgt, src)

        # Average attention weights across heads (interpretable)
        avg_attn = attn_weights.mean(dim=1)  # (batch, tgt, src)

        # Apply averaged attention to shared V
        output = torch.matmul(avg_attn, v)  # (batch, tgt, head_dim)
        output = self.out_proj(output)  # (batch, tgt, embed_dim)

        return output, avg_attn
