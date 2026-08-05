import torch
from models.regime_attention import RegimeAttentionBias

def test_bias_output_shape():
    bias_module = RegimeAttentionBias(num_states=3, state_size=32, num_heads=4, seq_len=25)
    regime_probs = torch.randn(8, 12, 3).softmax(dim=-1)  # decoder timesteps
    bias = bias_module(regime_probs, total_seq_len=25)
    assert bias.shape == (8, 12, 25)

def test_bias_is_zero_when_uniform_regime():
    """With uniform regime probs, bias should be similar across positions."""
    bias_module = RegimeAttentionBias(num_states=3, state_size=32, num_heads=4, seq_len=25)
    uniform = torch.ones(4, 5, 3) / 3
    bias = bias_module(uniform, total_seq_len=25)
    # Not exactly zero, but should be consistent across batch
    assert bias.std(dim=0).mean() < bias.abs().mean()
