import torch
from models.regime_module import RegimeDetectionModule

def test_regime_output_shape():
    mod = RegimeDetectionModule(input_dim=32, num_states=3, hidden_dim=64)
    h = torch.randn(8, 25, 32)  # LSTM hidden states
    probs, transition_matrix = mod(h)
    assert probs.shape == (8, 25, 3)
    assert transition_matrix.shape == (3, 3)

def test_regime_probs_sum_to_one():
    mod = RegimeDetectionModule(input_dim=32, num_states=3, hidden_dim=64)
    h = torch.randn(4, 10, 32)
    probs, _ = mod(h)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_transition_matrix_rows_sum_to_one():
    mod = RegimeDetectionModule(input_dim=32, num_states=3, hidden_dim=64)
    h = torch.randn(4, 10, 32)
    _, trans = mod(h)
    row_sums = trans.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_regime_initial_self_transition_high():
    """Transition matrix should be initialized with high diagonal (regime persistence)."""
    mod = RegimeDetectionModule(input_dim=32, num_states=3, hidden_dim=64)
    h = torch.randn(1, 5, 32)
    _, trans = mod(h)
    for i in range(3):
        assert trans[i, i] > 0.8, f"Self-transition for state {i} too low: {trans[i,i]}"

def test_regime_gradient_flow():
    mod = RegimeDetectionModule(input_dim=32, num_states=3, hidden_dim=64)
    h = torch.randn(4, 10, 32, requires_grad=True)
    probs, _ = mod(h)
    loss = probs.sum()
    loss.backward()
    assert h.grad is not None
