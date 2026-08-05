import torch
import torch.nn as nn
from typing import List, Optional


class InputChannelEmbedding(nn.Module):
    """Embeds numeric (via Linear) and categorical (via Embedding) features to state_size each."""

    def __init__(
        self,
        state_size: int,
        num_numeric: int,
        num_categorical: int,
        categorical_cardinalities: List[int],
    ):
        super().__init__()
        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical

        # Each numeric feature gets its own projection
        self.numeric_projections = nn.ModuleList([
            nn.Linear(1, state_size) for _ in range(num_numeric)
        ])

        # Each categorical feature gets its own embedding
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(card, state_size) for card in categorical_cardinalities
        ])

    def forward(
        self,
        x_numeric: Optional[torch.Tensor] = None,
        x_categorical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = []

        if x_numeric is not None and self.num_numeric > 0:
            for i in range(self.num_numeric):
                # x_numeric[..., i:i+1] keeps the last dim
                projected = self.numeric_projections[i](x_numeric[..., i:i+1])
                embeddings.append(projected)

        if x_categorical is not None and self.num_categorical > 0:
            for i in range(self.num_categorical):
                embedded = self.categorical_embeddings[i](x_categorical[..., i])
                embeddings.append(embedded)

        if not embeddings:
            # Return zero-size tensor preserving batch dims
            ref = x_numeric if x_numeric is not None and x_numeric.numel() > 0 else x_categorical
            if ref is not None and ref.numel() > 0:
                batch_shape = ref.shape[:-1]
                return torch.zeros(*batch_shape, 0, device=ref.device)
            return torch.empty(0)

        return torch.cat(embeddings, dim=-1)
