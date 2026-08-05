import torch
from models.tft import TemporalFusionTransformer
from models.configs import TFTConfig

def make_test_config():
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
    )

def test_tft_forward_classification():
    cfg = make_test_config()
    model = TemporalFusionTransformer(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    assert "predicted" in output
    assert output["predicted"].shape == (4, 5, 1)
    assert "attention_scores" in output
    assert "static_weights" in output

def test_tft_forward_regression():
    cfg = make_test_config()
    cfg.task_type = "regression"
    cfg.quantiles = [0.1, 0.5, 0.9]
    cfg.num_outputs = 3
    model = TemporalFusionTransformer(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    assert output["predicted"].shape == (4, 5, 3)

def test_tft_gradient_flow():
    """Ensure gradients flow through all parameters."""
    cfg = make_test_config()
    model = TemporalFusionTransformer(cfg)
    batch = {
        "static_feats_numeric": torch.randn(4, 1),
        "static_feats_categorical": torch.randint(0, 3, (4, 1)),
        "historical_ts_numeric": torch.randn(4, 20, 2),
        "future_ts_numeric": torch.randn(4, 5, 1),
    }
    output = model(batch)
    loss = output["predicted"].sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
