import torch
import torch.nn as nn
from models.components.glu import GatedLinearUnit


class GateAddNorm(nn.Module):
    """Gating + residual add + LayerNorm. Paper Eq. 2 outer part."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.glu = GatedLinearUnit(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.layer_norm(self.glu(x) + residual)
