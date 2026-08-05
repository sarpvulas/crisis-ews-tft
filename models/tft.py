import torch
import torch.nn as nn
from typing import Dict
from models.configs import TFTConfig
from models.components.embeddings import InputChannelEmbedding
from models.components.vsn import VariableSelectionNetwork
from models.components.grn import GatedResidualNetwork
from models.components.gate_add_norm import GateAddNorm
from models.components.attention import InterpretableMultiHeadAttention


class TemporalFusionTransformer(nn.Module):
    """Vanilla TFT implementation following Lim et al. (2021)."""

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        s = config.state_size

        # === Input Embeddings ===
        self.static_embedding = InputChannelEmbedding(
            s, config.num_static_numeric, config.num_static_categorical,
            config.static_categorical_cardinalities,
        )
        self.historical_embedding = InputChannelEmbedding(
            s, config.num_historical_numeric, config.num_historical_categorical,
            config.historical_categorical_cardinalities,
        )
        self.future_embedding = InputChannelEmbedding(
            s, config.num_future_numeric, config.num_future_categorical,
            config.future_categorical_cardinalities,
        )

        # === Variable Selection ===
        self.static_vsn = VariableSelectionNetwork(
            s, config.total_static_inputs, s, config.dropout,
        )
        self.historical_vsn = VariableSelectionNetwork(
            s, config.total_historical_inputs, s, config.dropout, context_dim=s,
        )
        self.future_vsn = VariableSelectionNetwork(
            s, config.total_future_inputs, s, config.dropout, context_dim=s,
        )

        # === Static Encoders (4 context vectors) ===
        self.static_encoder_selection = GatedResidualNetwork(s, s, s, dropout=config.dropout)
        self.static_encoder_enrichment = GatedResidualNetwork(s, s, s, dropout=config.dropout)
        self.static_encoder_seq_hidden = GatedResidualNetwork(s, s, s, dropout=config.dropout)
        self.static_encoder_seq_cell = GatedResidualNetwork(s, s, s, dropout=config.dropout)

        # === LSTM Encoder-Decoder ===
        self.encoder_lstm = nn.LSTM(s, s, config.lstm_layers, batch_first=True, dropout=config.dropout)
        self.decoder_lstm = nn.LSTM(s, s, config.lstm_layers, batch_first=True, dropout=config.dropout)
        self.lstm_gate = GateAddNorm(s, dropout=config.dropout)

        # === Static Enrichment ===
        self.static_enrichment_grn = GatedResidualNetwork(s, s, s, context_dim=s, dropout=config.dropout)

        # === Attention ===
        self.attention = InterpretableMultiHeadAttention(s, config.attention_heads)
        self.attention_gate = GateAddNorm(s, dropout=config.dropout)

        # === Position-wise Feed-Forward ===
        self.poswise_grn = GatedResidualNetwork(s, s, s, dropout=config.dropout)
        self.poswise_gate = GateAddNorm(s, dropout=config.dropout)

        # === Output ===
        self.output_layer = nn.Linear(s, config.num_outputs)

        # === Optional novel TFT-on-top add-ons ===
        # Auxiliary forward-drawdown regression head (dense multi-task supervision).
        self.aux_head = nn.Linear(s, 1) if getattr(config, "use_aux", False) else None
        # Regime-conditioned VSN: a learned summary of the encoder window injected
        # into the (otherwise ~zero, no-static) variable-selection context. The
        # historical embedding is (batch, time, n_vars * s), so project from that.
        self.vsn_regime = (
            nn.Linear(config.total_historical_inputs * s, s)
            if getattr(config, "use_regime_vsn", False) else None
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Embed inputs ---
        static_rep = self.static_embedding(
            x_numeric=batch.get("static_feats_numeric"),
            x_categorical=batch.get("static_feats_categorical"),
        )
        hist_rep = self.historical_embedding(
            x_numeric=batch.get("historical_ts_numeric"),
            x_categorical=batch.get("historical_ts_categorical"),
        )
        future_rep = self.future_embedding(
            x_numeric=batch.get("future_ts_numeric"),
            x_categorical=batch.get("future_ts_categorical"),
        )

        num_samples = hist_rep.shape[0]
        num_encoder_steps = hist_rep.shape[1]
        num_decoder_steps = future_rep.shape[1]
        s = self.config.state_size
        device = hist_rep.device

        # --- Static Variable Selection ---
        has_static = self.config.total_static_inputs > 0
        if has_static:
            selected_static, static_weights = self.static_vsn(static_rep)
            c_selection = self.static_encoder_selection(selected_static)
            c_enrichment = self.static_encoder_enrichment(selected_static)
            c_seq_hidden = self.static_encoder_seq_hidden(selected_static)
            c_seq_cell = self.static_encoder_seq_cell(selected_static)
        else:
            static_weights = torch.zeros(num_samples, 0, device=device)
            c_selection = torch.zeros(num_samples, s, device=device)
            c_enrichment = torch.zeros(num_samples, s, device=device)
            c_seq_hidden = torch.zeros(num_samples, s, device=device)
            c_seq_cell = torch.zeros(num_samples, s, device=device)

        # --- Regime-conditioned VSN context (optional) ---
        if self.vsn_regime is not None:
            c_selection = c_selection + self.vsn_regime(hist_rep.mean(dim=1))

        # --- Temporal Variable Selection ---
        c_sel_hist = c_selection.unsqueeze(1).expand(-1, num_encoder_steps, -1)
        selected_hist, hist_weights = self.historical_vsn(hist_rep, context=c_sel_hist)

        c_sel_future = c_selection.unsqueeze(1).expand(-1, num_decoder_steps, -1)
        selected_future, future_weights = self.future_vsn(future_rep, context=c_sel_future)

        # --- LSTM ---
        init_hidden = c_seq_hidden.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
        init_cell = c_seq_cell.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)

        encoder_out, (hidden, cell) = self.encoder_lstm(selected_hist, (init_hidden, init_cell))
        decoder_out, _ = self.decoder_lstm(selected_future, (hidden, cell))

        lstm_out = torch.cat([encoder_out, decoder_out], dim=1)
        temporal_input = torch.cat([selected_hist, selected_future], dim=1)
        gated_lstm = self.lstm_gate(lstm_out, temporal_input)

        # --- Static Enrichment ---
        c_enrich = c_enrichment.unsqueeze(1).expand(-1, gated_lstm.shape[1], -1)
        enriched = self.static_enrichment_grn(gated_lstm, context=c_enrich)

        # --- Self-Attention ---
        # Build causal mask: decoder can attend to all encoder + causal decoder
        total_steps = num_encoder_steps + num_decoder_steps
        mask = torch.ones(num_decoder_steps, total_steps, device=enriched.device).bool()
        # Causal mask for decoder-to-decoder attention
        for i in range(num_decoder_steps):
            future_decoder_pos = num_encoder_steps + i + 1
            mask[i, future_decoder_pos:] = False

        # Only attend from decoder positions
        decoder_enriched = enriched[:, num_encoder_steps:, :]
        attn_out, attn_scores = self.attention(
            q=decoder_enriched,
            k=enriched,
            v=enriched,
            mask=mask,
        )
        gated_attn = self.attention_gate(attn_out, decoder_enriched)

        # --- Position-wise Feed-Forward ---
        poswise_out = self.poswise_grn(gated_attn)
        # Skip connection to LSTM output (decoder portion only)
        gated_poswise = self.poswise_gate(poswise_out, gated_lstm[:, num_encoder_steps:, :])

        # --- Output ---
        predicted = self.output_layer(gated_poswise)

        result = {
            "predicted": predicted,
            "attention_scores": attn_scores,
            "static_weights": static_weights,
            "historical_weights": hist_weights,
            "future_weights": future_weights,
            "gated_lstm_output": gated_lstm,
        }
        if self.aux_head is not None:
            result["aux_pred"] = self.aux_head(gated_poswise)
        return result
