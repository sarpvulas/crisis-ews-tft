import torch
import torch.nn as nn
from typing import Dict
from models.tft import TemporalFusionTransformer
from models.regime_module import RegimeDetectionModule
from models.regime_attention import RegimeAttentionBias
from models.configs import TFTConfig


class RegimeAwareTFT(TemporalFusionTransformer):
    """Regime-Aware TFT: extends vanilla TFT with regime detection and conditioned attention."""

    def __init__(self, config: TFTConfig):
        super().__init__(config)

        if config.use_regime_module:
            self.regime_module = RegimeDetectionModule(
                input_dim=config.state_size,
                num_states=config.num_regime_states,
                hidden_dim=config.state_size,
            )
        else:
            self.regime_module = None

        if config.use_regime_attention:
            total_steps = config.encoder_steps + config.decoder_steps
            self.regime_bias = RegimeAttentionBias(
                num_states=config.num_regime_states,
                state_size=config.state_size,
                num_heads=config.attention_heads,
                seq_len=total_steps,
            )
        else:
            self.regime_bias = None

        self.regime_heads = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Embed inputs (same as vanilla TFT) ---
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
        total_steps = num_encoder_steps + num_decoder_steps
        s = self.config.state_size
        device = hist_rep.device

        # --- Variable Selection ---
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

        # Regime-conditioned VSN context (optional)
        if getattr(self, "vsn_regime", None) is not None:
            c_selection = c_selection + self.vsn_regime(hist_rep.mean(dim=1))

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

        # === REGIME MODULE (Novel Component 1) ===
        regime_probs = None
        transition_matrix = None
        if self.regime_module is not None:
            regime_probs, transition_matrix = self.regime_module(gated_lstm)

        # --- Static Enrichment ---
        c_enrich = c_enrichment.unsqueeze(1).expand(-1, gated_lstm.shape[1], -1)
        enriched = self.static_enrichment_grn(gated_lstm, context=c_enrich)

        # --- Self-Attention with Regime Bias (Novel Component 2) ---
        mask = torch.ones(num_decoder_steps, total_steps, device=enriched.device).bool()
        for i in range(num_decoder_steps):
            mask[i, num_encoder_steps + i + 1:] = False

        decoder_enriched = enriched[:, num_encoder_steps:, :]

        attn_bias = None
        if self.regime_bias is not None and regime_probs is not None:
            decoder_regime = regime_probs[:, num_encoder_steps:, :]
            attn_bias = self.regime_bias(decoder_regime, total_seq_len=total_steps)

        attn_out, attn_scores = self.attention(
            q=decoder_enriched, k=enriched, v=enriched,
            mask=mask, attn_bias=attn_bias,
        )
        gated_attn = self.attention_gate(attn_out, decoder_enriched)

        # --- Position-wise FF + Output ---
        poswise_out = self.poswise_grn(gated_attn)
        gated_poswise = self.poswise_gate(poswise_out, gated_lstm[:, num_encoder_steps:, :])

        predicted = self.output_layer(gated_poswise)

        result = {
            "predicted": predicted,
            "attention_scores": attn_scores,
            "static_weights": static_weights,
            "historical_weights": hist_weights,
            "future_weights": future_weights,
            "gated_lstm_output": gated_lstm,
        }

        if regime_probs is not None:
            result["regime_probs"] = regime_probs
            result["transition_matrix"] = transition_matrix

        if getattr(self, "aux_head", None) is not None:
            result["aux_pred"] = self.aux_head(gated_poswise)

        return result
