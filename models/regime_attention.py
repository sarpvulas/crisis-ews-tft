import torch
import torch.nn as nn


class RegimeAttentionBias(nn.Module):
    """Projects regime probabilities to attention bias. Paper Novel Component 2."""

    def __init__(self, num_states: int, state_size: int, num_heads: int, seq_len: int):
        super().__init__()
        self.regime_embedding = nn.Linear(num_states, state_size)
        self.bias_proj = nn.Linear(state_size, seq_len)

    def forward(self, regime_probs: torch.Tensor, total_seq_len: int) -> torch.Tensor:
        """
        Args:
            regime_probs: (batch, decoder_steps, num_states)
            total_seq_len: encoder_steps + decoder_steps
        Returns:
            bias: (batch, decoder_steps, total_seq_len) -- additive attention bias
        """
        embedded = self.regime_embedding(regime_probs)  # (batch, dec, state_size)
        bias = self.bias_proj(embedded)  # (batch, dec, seq_len_proj)

        # If projection dim doesn't match total_seq_len, interpolate
        if bias.shape[-1] != total_seq_len:
            bias = torch.nn.functional.interpolate(
                bias.unsqueeze(1), size=total_seq_len, mode="linear", align_corners=False
            ).squeeze(1)

        return bias
