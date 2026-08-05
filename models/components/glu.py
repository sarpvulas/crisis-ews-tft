import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU(x) = sigmoid(W1*x + b1) * (W2*x + b2). Paper Eq. 1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(x)) * self.fc2(x)
