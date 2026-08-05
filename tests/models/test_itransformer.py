"""Tests for the iTransformer port (models/itransformer.py)."""

import torch
import pytest

from models.itransformer import ITransformer, ITransformerConfig


@pytest.fixture
def small_config():
    return ITransformerConfig(
        num_historical_numeric=8,
        num_future_numeric=2,
        encoder_steps=24,
        decoder_steps=6,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        dropout=0.0,
    )


class TestITransformer:

    def test_forward_shape_mean_pool(self, small_config):
        model = ITransformer(small_config)
        batch = {"historical_ts_numeric": torch.randn(3, small_config.encoder_steps, small_config.num_historical_numeric)}
        out = model(batch)
        assert out["predicted"].shape == (3, small_config.decoder_steps, 1)

    def test_forward_shape_cls_pool(self):
        cfg = ITransformerConfig(
            num_historical_numeric=8, encoder_steps=24, decoder_steps=6,
            d_model=32, n_heads=4, e_layers=2, d_ff=64, dropout=0.0, pool="cls",
        )
        model = ITransformer(cfg)
        batch = {"historical_ts_numeric": torch.randn(2, cfg.encoder_steps, cfg.num_historical_numeric)}
        out = model(batch)
        assert out["predicted"].shape == (2, cfg.decoder_steps, 1)

    def test_invalid_pool_raises(self):
        with pytest.raises(ValueError):
            ITransformerConfig(num_historical_numeric=8, pool="banana")
            ITransformer(ITransformerConfig(num_historical_numeric=8, pool="banana"))

    def test_gradients_flow(self, small_config):
        model = ITransformer(small_config)
        batch = {"historical_ts_numeric": torch.randn(2, small_config.encoder_steps, small_config.num_historical_numeric, requires_grad=True)}
        out = model(batch)
        loss = out["predicted"].mean()
        loss.backward()
        # Every parameter should have a gradient.
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} got no grad"

    def test_attention_is_over_variates_not_time(self, small_config):
        """A sanity check: doubling the variate count should change output
        shape's last dim of the embedded representation, not the time
        dimension. Catches accidental axis swaps in _VariateEmbedding."""
        model = ITransformer(small_config)
        batch = {"historical_ts_numeric": torch.randn(1, small_config.encoder_steps, small_config.num_historical_numeric)}
        # Walk through embed manually to assert shape semantics.
        h = model.embed(batch["historical_ts_numeric"])
        assert h.shape == (1, small_config.num_historical_numeric, small_config.d_model)
