import torch
from models.ra_tft import RegimeAwareTFT
from models.configs import TFTConfig

def make_ra_config():
    return TFTConfig(
        num_historical_numeric=2,
        num_static_numeric=1,
        num_static_categorical=1,
        static_categorical_cardinalities=[3],
        num_future_numeric=1,
        state_size=32,
        hidden_size=32,
        attention_heads=4,
        lstm_layers=2,
        dropout=0.1,
        encoder_steps=20,
        decoder_steps=5,
        task_type="classification",
        num_outputs=1,
        use_regime_module=True,
        use_regime_attention=True,
        num_regime_states=3,
    )

def test_ra_tft_forward():
    cfg = make_ra_config()
    model = RegimeAwareTFT(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    assert output["predicted"].shape == (4, 5, 1)
    assert "regime_probs" in output
    assert output["regime_probs"].shape == (4, 25, 3)  # encoder + decoder steps
    assert "transition_matrix" in output

def test_ra_tft_ablation_no_regime():
    """With use_regime_module=False, should behave like vanilla TFT."""
    cfg = make_ra_config()
    cfg.use_regime_module = False
    cfg.use_regime_attention = False
    model = RegimeAwareTFT(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    assert output["predicted"].shape == (4, 5, 1)
    assert "regime_probs" not in output

def test_ra_tft_gradient_flow():
    cfg = make_ra_config()
    model = RegimeAwareTFT(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    loss = output["predicted"].sum() + output["regime_probs"].sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
