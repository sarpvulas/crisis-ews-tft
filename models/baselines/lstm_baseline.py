"""Simple LSTM baseline for crisis prediction."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class LSTMBaseline(nn.Module):
    """2-layer LSTM baseline matching TFT input interface.

    For Task A: outputs sigmoid probabilities per decoder step.
    For Task B: outputs quantile predictions per decoder step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        task: str = "A",
        encoder_steps: int = 60,
        decoder_steps: int = 12,
        num_quantiles: int = 1,
    ):
        super().__init__()
        self.task = task
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles if task == "B" else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, self.num_quantiles)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass using the same batch dict interface as TFT."""
        x = batch["historical_ts_numeric"]  # (B, encoder_steps, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state repeated for each decoder step
        last_hidden = lstm_out[:, -1:, :]  # (B, 1, hidden_dim)
        decoder_input = last_hidden.expand(-1, self.decoder_steps, -1)

        predicted = self.output_layer(decoder_input)  # (B, decoder_steps, num_quantiles)

        if self.task == "A":
            predicted = torch.sigmoid(predicted)

        return {"predicted": predicted}

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict probabilities/values from batch dict.

        Returns:
            Task A: (batch, decoder_steps) probabilities
            Task B: (batch, decoder_steps, num_quantiles)
        """
        self.eval()
        outputs = self.forward(batch)
        preds = outputs["predicted"]

        if self.task == "A":
            return preds.squeeze(-1)
        return preds
