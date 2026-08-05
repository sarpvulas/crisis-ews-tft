import torch
from models.components.glu import GatedLinearUnit
from models.components.gate_add_norm import GateAddNorm
from models.components.grn import GatedResidualNetwork
from models.components.embeddings import InputChannelEmbedding
from models.components.vsn import VariableSelectionNetwork
from models.components.attention import InterpretableMultiHeadAttention

def test_glu_output_shape():
    glu = GatedLinearUnit(input_dim=32)
    x = torch.randn(8, 10, 32)
    out = glu(x)
    assert out.shape == (8, 10, 32)

def test_glu_preserves_dimension():
    glu = GatedLinearUnit(input_dim=64)
    x = torch.randn(4, 64)
    assert glu(x).shape == x.shape

def test_gate_add_norm_output_shape():
    gan = GateAddNorm(input_dim=32, dropout=0.1)
    x = torch.randn(8, 10, 32)
    residual = torch.randn(8, 10, 32)
    out = gan(x, residual)
    assert out.shape == (8, 10, 32)

def test_gate_add_norm_residual_connection():
    """With zero input, output should be close to LayerNorm(residual)."""
    gan = GateAddNorm(input_dim=32, dropout=0.0)
    # Set GLU weights to produce zero output
    with torch.no_grad():
        for p in gan.glu.parameters():
            p.zero_()
    x = torch.zeros(4, 32)
    residual = torch.ones(4, 32)
    out = gan(x, residual)
    # Should be LayerNorm(0 + residual) = LayerNorm(ones)
    expected = torch.nn.LayerNorm(32)(residual)
    assert torch.allclose(out, expected, atol=1e-5)

def test_grn_output_shape():
    grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32, dropout=0.1)
    x = torch.randn(8, 10, 32)
    out = grn(x)
    assert out.shape == (8, 10, 32)

def test_grn_with_context():
    grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32, context_dim=16, dropout=0.1)
    x = torch.randn(8, 10, 32)
    c = torch.randn(8, 10, 16)
    out = grn(x, context=c)
    assert out.shape == (8, 10, 32)

def test_grn_skip_projection():
    """When input_dim != output_dim, GRN should project the skip connection."""
    grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=16, dropout=0.1)
    x = torch.randn(8, 32)
    out = grn(x)
    assert out.shape == (8, 16)

def test_numeric_embedding():
    emb = InputChannelEmbedding(state_size=32, num_numeric=3, num_categorical=0, categorical_cardinalities=[])
    x_num = torch.randn(8, 10, 3)
    out = emb(x_numeric=x_num)
    assert out.shape == (8, 10, 3 * 32)

def test_categorical_embedding():
    emb = InputChannelEmbedding(state_size=32, num_numeric=0, num_categorical=2, categorical_cardinalities=[5, 10])
    x_cat = torch.randint(0, 5, (8, 10, 2))
    out = emb(x_categorical=x_cat)
    assert out.shape == (8, 10, 2 * 32)

def test_mixed_embedding():
    emb = InputChannelEmbedding(state_size=32, num_numeric=2, num_categorical=1, categorical_cardinalities=[5])
    x_num = torch.randn(8, 10, 2)
    x_cat = torch.randint(0, 5, (8, 10, 1))
    out = emb(x_numeric=x_num, x_categorical=x_cat)
    assert out.shape == (8, 10, 3 * 32)  # 2 numeric + 1 categorical

def test_static_embedding():
    """Static features have no time dimension."""
    emb = InputChannelEmbedding(state_size=32, num_numeric=1, num_categorical=1, categorical_cardinalities=[3])
    x_num = torch.randn(8, 1)
    x_cat = torch.randint(0, 3, (8, 1))
    out = emb(x_numeric=x_num, x_categorical=x_cat)
    assert out.shape == (8, 2 * 32)

def test_vsn_output_shape():
    vsn = VariableSelectionNetwork(
        input_dim=32, num_inputs=4, hidden_dim=64, dropout=0.1
    )
    # Flattened input: (batch, time, num_inputs * input_dim)
    x = torch.randn(8, 10, 4 * 32)
    out, weights = vsn(x)
    assert out.shape == (8, 10, 32)
    assert weights.shape == (8, 10, 4)

def test_vsn_weights_sum_to_one():
    vsn = VariableSelectionNetwork(input_dim=32, num_inputs=3, hidden_dim=64, dropout=0.1)
    x = torch.randn(8, 10, 3 * 32)
    _, weights = vsn(x)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_vsn_with_context():
    vsn = VariableSelectionNetwork(
        input_dim=32, num_inputs=4, hidden_dim=64, dropout=0.1, context_dim=32
    )
    x = torch.randn(8, 10, 4 * 32)
    c = torch.randn(8, 10, 32)
    out, weights = vsn(x, context=c)
    assert out.shape == (8, 10, 32)

def test_attention_output_shape():
    attn = InterpretableMultiHeadAttention(embed_dim=32, num_heads=4)
    q = torch.randn(8, 12, 32)  # decoder steps
    k = torch.randn(8, 60, 32)  # encoder + decoder steps
    v = torch.randn(8, 60, 32)
    out, scores = attn(q, k, v)
    assert out.shape == (8, 12, 32)
    assert scores.shape == (8, 12, 60)  # averaged across heads

def test_attention_with_mask():
    attn = InterpretableMultiHeadAttention(embed_dim=32, num_heads=4)
    q = torch.randn(2, 5, 32)
    k = torch.randn(2, 10, 32)
    v = torch.randn(2, 10, 32)
    mask = torch.ones(5, 10).bool()
    mask[:, 8:] = False  # Mask out last 2 positions
    out, scores = attn(q, k, v, mask=mask)
    # Masked positions should have ~0 attention
    assert scores[:, :, 8:].max() < 0.01

def test_attention_scores_sum_to_one():
    attn = InterpretableMultiHeadAttention(embed_dim=32, num_heads=4)
    q = torch.randn(4, 5, 32)
    k = torch.randn(4, 10, 32)
    v = torch.randn(4, 10, 32)
    _, scores = attn(q, k, v)
    sums = scores.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_attention_with_regime_bias():
    """Attention should accept an optional additive bias."""
    attn = InterpretableMultiHeadAttention(embed_dim=32, num_heads=4)
    q = torch.randn(4, 5, 32)
    k = torch.randn(4, 10, 32)
    v = torch.randn(4, 10, 32)
    bias = torch.randn(4, 5, 10)  # additive bias
    out_no_bias, scores_no_bias = attn(q, k, v)
    out_bias, scores_bias = attn(q, k, v, attn_bias=bias)
    # Outputs should differ when bias is applied
    assert not torch.allclose(scores_no_bias, scores_bias, atol=1e-3)
