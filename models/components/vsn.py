import torch
import torch.nn as nn
from models.components.grn import GatedResidualNetwork
from typing import Tuple, Optional


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network. Paper Section 4.2."""

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_inputs = num_inputs

        # GRN for computing variable weights from flattened input
        self.flattened_grn = GatedResidualNetwork(
            input_dim=num_inputs * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.softmax = nn.Softmax(dim=-1)

        # Per-variable GRNs
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                dropout=dropout,
            )
            for _ in range(num_inputs)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (..., num_inputs * input_dim)
        # Compute sparse weights
        weights = self.softmax(self.flattened_grn(x, context=context))  # (..., num_inputs)

        # Split input and process each variable
        var_outputs = []
        for i in range(self.num_inputs):
            var_input = x[..., i * self.input_dim:(i + 1) * self.input_dim]
            var_outputs.append(self.single_variable_grns[i](var_input))

        # Stack: (..., num_inputs, input_dim)
        var_outputs = torch.stack(var_outputs, dim=-2)

        # Weighted sum: (..., input_dim)
        weighted = (weights.unsqueeze(-1) * var_outputs).sum(dim=-2)

        return weighted, weights
