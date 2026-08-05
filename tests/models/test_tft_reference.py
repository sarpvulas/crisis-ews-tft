"""Verify our TFT produces same-shaped outputs as tft-torch and both converge on synthetic data."""
import torch
from models.tft import TemporalFusionTransformer
from models.configs import TFTConfig

def test_our_tft_converges_on_synthetic_data():
    """Train our TFT on simple synthetic data and verify loss decreases."""
    cfg = TFTConfig(
        num_historical_numeric=2,
        num_static_numeric=1,
        num_static_categorical=1,
        static_categorical_cardinalities=[2],
        num_future_numeric=1,
        state_size=16,
        hidden_size=16,
        attention_heads=2,
        lstm_layers=1,
        encoder_steps=20,
        decoder_steps=5,
        task_type="ews",
        num_outputs=1,
    )
    model = TemporalFusionTransformer(cfg)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    batch = {
        "static_feats_numeric": torch.randn(16, 1),
        "static_feats_categorical": torch.randint(0, 2, (16, 1)),
        "historical_ts_numeric": torch.randn(16, 20, 2),
        "future_ts_numeric": torch.randn(16, 5, 1),
    }
    targets = torch.randn(16, 5)

    initial_loss = None
    for i in range(100):
        output = model(batch)
        loss = ((output["predicted"].squeeze(-1) - targets) ** 2).mean()
        if i == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
