import torch
import torch.nn as nn
from models.components.gate_add_norm import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(a + GLU(eta1)). Paper Eqs. 2-4."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # W1
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # W3
        self.gate_add_norm = GateAddNorm(output_dim, dropout=dropout)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        return self.gate_add_norm(hidden, residual)
