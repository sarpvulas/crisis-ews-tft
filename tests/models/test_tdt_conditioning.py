"""Tests for TDT conditioning extraction (Stage 1.I).

The conditioning signal selects which `p(x | c)` distribution each training
window contributes to. After Stage 1.I, the default is `condition_mode='ews'`
— the binary max(ews_label) over the decoder window — which aligns training
with the test target.
"""

import torch
import pytest

from models.tdt import TDT, TDTConfig
from models.tdt_adapter import TDTCrisisModel


def _model(**cfg_kw):
    base = dict(
        num_historical_numeric=4, encoder_steps=12, decoder_steps=6,
        d_model=16, n_heads=2, e_layers=1, d_ff=32,
        diffusion_steps=40, num_regime_states=3,
    )
    base.update(cfg_kw)
    cfg = TDTConfig(**base)
    return TDTCrisisModel(TDT(cfg))


class TestEwsConditioning:

    def test_default_mode_is_ews(self):
        m = _model()
        assert m.module.config.condition_mode == "ews"

    def test_all_negative_window_maps_to_calm_id(self):
        m = _model()
        B, D = 4, m.module.config.decoder_steps
        targets = {"ews_label": torch.zeros(B, D)}
        c = m._extract_condition(targets, B, "cpu")
        assert c.shape == (B,)
        # calm id is 0
        assert torch.all(c == 0)

    def test_any_positive_window_maps_to_crisis_id(self):
        m = _model()
        K = m.module.config.num_regime_states
        B, D = 4, m.module.config.decoder_steps
        ews = torch.zeros(B, D)
        ews[:, 3] = 1.0  # one positive timestep per window
        c = m._extract_condition({"ews_label": ews}, B, "cpu")
        # crisis id is K-1
        assert torch.all(c == K - 1)

    def test_mixed_windows_partition_correctly(self):
        m = _model()
        K = m.module.config.num_regime_states
        B, D = 6, m.module.config.decoder_steps
        ews = torch.zeros(B, D)
        ews[[1, 3, 5], 2] = 1.0  # rows 1, 3, 5 are positive
        c = m._extract_condition({"ews_label": ews}, B, "cpu")
        assert c.tolist() == [0, K - 1, 0, K - 1, 0, K - 1]


class TestRegimeConditioning:

    def test_regime_mode_uses_regime_label(self):
        m = _model(condition_mode="regime")
        B = 3
        E = m.module.config.encoder_steps
        D = m.module.config.decoder_steps
        regime = torch.zeros(B, E + D, dtype=torch.long)
        regime[0, (E + D) // 2 - 1] = 2
        regime[1, (E + D) // 2 - 1] = 1
        c = m._extract_condition({"regime_label": regime}, B, "cpu")
        # Picks the value at the encoder/decoder boundary index.
        assert c[0].item() == 2
        assert c[1].item() == 1
        assert c[2].item() == 0


class TestFallback:

    def test_missing_label_returns_zeros(self):
        m = _model()
        c = m._extract_condition({}, 5, "cpu")
        assert c.shape == (5,)
        assert torch.all(c == 0)

    def test_regime_mode_with_no_regime_label_returns_zeros(self):
        m = _model(condition_mode="regime")
        c = m._extract_condition({}, 3, "cpu")
        assert torch.all(c == 0)
