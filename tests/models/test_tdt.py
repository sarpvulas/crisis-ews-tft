"""Tests for the Temporal Diffusion Transformer (models/tdt.py).

These cover:
  - shape contracts (forward, denoise, crisis_score)
  - the noise schedule's mathematical properties
  - classifier-free guidance routing (unconditional embedding swap)
  - the score_timesteps clipping fix
  - gradients flow through every parameter
"""

import torch
import pytest

from models.tdt import (
    TDT,
    TDTConfig,
    _cosine_alpha_bar,
    _make_schedule,
)


@pytest.fixture
def small_config():
    return TDTConfig(
        num_historical_numeric=6,
        encoder_steps=24,
        decoder_steps=4,
        d_model=24,
        n_heads=4,
        e_layers=2,
        d_ff=48,
        diffusion_steps=80,
        num_regime_states=3,
        dropout=0.0,
    )


class TestNoiseSchedule:

    def test_cosine_starts_at_one(self):
        ab = _cosine_alpha_bar(100)
        assert ab[0].item() == pytest.approx(1.0, abs=1e-6)

    def test_cosine_is_monotone_decreasing(self):
        ab = _cosine_alpha_bar(200)
        diffs = ab[1:] - ab[:-1]
        # Allow tiny float noise — strict negative on average.
        assert (diffs <= 1e-7).all()

    def test_linear_schedule_shape(self):
        sched = _make_schedule(50, "linear")
        for key in ("alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"):
            assert sched[key].shape == (51,)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            _make_schedule(10, "wat")


class TestForwardShapes:

    def test_diffusion_step(self, small_config):
        model = TDT(small_config)
        x0 = torch.randn(3, small_config.encoder_steps, small_config.num_historical_numeric)
        c = torch.tensor([0, 1, 2])
        out = model.diffusion_step(x0, c)
        assert out["loss_simple"].ndim == 0
        assert out["eps_pred"].shape == x0.shape
        assert out["regime_logits"].shape == (3, small_config.num_regime_states)

    def test_forward_returns_decoder_step_logits(self, small_config):
        model = TDT(small_config)
        batch = {"historical_ts_numeric": torch.randn(2, small_config.encoder_steps, small_config.num_historical_numeric)}
        out = model(batch)
        assert out["predicted"].shape == (2, small_config.decoder_steps, 1)

    def test_crisis_score_returns_one_per_window(self, small_config):
        model = TDT(small_config)
        x = torch.randn(4, small_config.encoder_steps, small_config.num_historical_numeric)
        scores = model.crisis_score(x)
        assert scores.shape == (4,)


class TestClassifierFreeGuidance:

    def test_cond_proj_is_zero_initialised(self, small_config):
        """AdaLN-Zero design: every block's cond_proj starts at zero so the
        block is identity at init. This is what keeps diffusion training
        stable. Verify the invariant — if it ever breaks, training will
        diverge unpredictably."""
        model = TDT(small_config)
        for block in model.blocks:
            assert torch.all(block.cond_proj.weight == 0)
            assert torch.all(block.cond_proj.bias == 0)

    def test_unconditional_embedding_row_differs_from_conditional(self, small_config):
        """The unconditional row of the condition embedding must be a
        separate parameter from any conditional row, otherwise CFG is a
        no-op once cond_proj learns nonzero weights."""
        model = TDT(small_config)
        c_table = model.c_embed.embed.weight
        # Rows 0..(K-1) are the conditional regimes; row K is unconditional.
        K = small_config.num_regime_states
        for i in range(K):
            assert not torch.allclose(c_table[i], c_table[K])

    def test_after_training_cond_changes_eps(self, small_config):
        """Once cond_proj learns nonzero weights via a single optimiser step,
        swapping conditional <-> unconditional must change the eps prediction.
        Catches a bug where the unconditional path silently shares weights."""
        torch.manual_seed(0)
        model = TDT(small_config)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x0 = torch.randn(4, small_config.encoder_steps, small_config.num_historical_numeric)
        c = torch.tensor([0, 1, 2, 2])
        for _ in range(3):
            opt.zero_grad()
            loss = model.diffusion_step(x0, c)["loss_simple"]
            loss.backward()
            opt.step()
        # Now compute eps with and without conditioning.
        eps = torch.randn_like(x0)
        t = torch.full((4,), 10, dtype=torch.long)
        sqrt_ab = model.sqrt_alpha_bar[t].view(4, 1, 1)
        sqrt_om = model.sqrt_one_minus_alpha_bar[t].view(4, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_om * eps
        cond_out = model.denoise(x_t, t, c, cond_mask=torch.zeros(4, 1, dtype=torch.bool))
        uncond_out = model.denoise(x_t, t, c, cond_mask=torch.ones(4, 1, dtype=torch.bool))
        assert not torch.allclose(cond_out["eps_pred"], uncond_out["eps_pred"])


class TestScoreTimestepClipping:
    """Bug fix regression: if diffusion_steps < max(score_timesteps), the model
    used to index out of range. Now they're clipped proportionally."""

    def test_short_diffusion_does_not_crash(self):
        cfg = TDTConfig(
            num_historical_numeric=4, encoder_steps=12, decoder_steps=2,
            d_model=16, n_heads=2, e_layers=1, d_ff=32,
            diffusion_steps=20,
            score_timesteps=(100, 500, 999),  # all > 20
        )
        model = TDT(cfg)
        x = torch.randn(2, cfg.encoder_steps, cfg.num_historical_numeric)
        # Must not raise IndexError.
        scores = model.crisis_score(x)
        assert scores.shape == (2,)


class TestVLBScoring:
    """Stage 1.H — variational lower-bound scoring upgrade."""

    def _config(self, **kw):
        base = dict(
            num_historical_numeric=5, encoder_steps=12, decoder_steps=3,
            d_model=16, n_heads=2, e_layers=1, d_ff=32,
            diffusion_steps=40, num_regime_states=3, dropout=0.0,
            score_num_timesteps=4, score_repeats=2,
        )
        base.update(kw)
        return TDTConfig(**base)

    def test_vlb_default_method(self):
        cfg = self._config()
        assert cfg.score_method == "vlb"

    def test_vlb_weight_buffer_shape(self):
        cfg = self._config()
        model = TDT(cfg)
        assert hasattr(model, "vlb_weight")
        assert model.vlb_weight.shape == (cfg.diffusion_steps + 1,)
        # First entry corresponds to t=0 (no noise added); the rest should be positive.
        assert (model.vlb_weight[1:] > 0).all()

    def test_stratified_timesteps_span_schedule(self):
        cfg = self._config(score_num_timesteps=8)
        model = TDT(cfg)
        ts = model._stratified_timesteps()
        assert ts.min().item() >= 1
        assert ts.max().item() <= cfg.diffusion_steps
        assert len(ts) <= 8
        # Stratified linspace should cover both low and high noise levels.
        assert ts.min().item() < cfg.diffusion_steps // 4
        assert ts.max().item() > 3 * cfg.diffusion_steps // 4

    def test_vlb_score_shape(self):
        cfg = self._config()
        model = TDT(cfg)
        x = torch.randn(4, cfg.encoder_steps, cfg.num_historical_numeric)
        scores = model.crisis_score(x)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_mse_ratio_path_still_works(self):
        """Old MSE-ratio scoring kept for ablation; must still return finite scores."""
        cfg = self._config(score_method="mse_ratio")
        model = TDT(cfg)
        x = torch.randn(2, cfg.encoder_steps, cfg.num_historical_numeric)
        scores = model.crisis_score(x)
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()

    def test_unknown_score_method_raises(self):
        cfg = self._config(score_method="banana")
        model = TDT(cfg)
        x = torch.randn(1, cfg.encoder_steps, cfg.num_historical_numeric)
        with pytest.raises(ValueError, match="unknown score_method"):
            model.crisis_score(x)

    def test_after_training_vlb_score_distinguishes_conditions(self):
        """The whole point of VLB scoring: after training, the score should
        differ between two conditioned predictions. Verifies the new code path
        isn't accidentally identity-mapping."""
        torch.manual_seed(0)
        cfg = self._config()
        model = TDT(cfg)
        x0 = torch.randn(4, cfg.encoder_steps, cfg.num_historical_numeric)
        # Use synthetic labels that have structure: positives are first half.
        c = torch.tensor([0, 0, 2, 2])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(8):
            opt.zero_grad()
            model.diffusion_step(x0, c)["loss_simple"].backward()
            opt.step()
        # Score should not collapse to all-same values.
        scores = model.crisis_score(x0)
        assert scores.std() > 1e-6, f"VLB scores collapsed: {scores}"


class TestCausalMask:
    """Option A — causal-DAG attention bias integration."""

    def _config(self, **kw):
        base = dict(
            num_historical_numeric=6, encoder_steps=24, decoder_steps=4,
            d_model=24, n_heads=2, e_layers=2, d_ff=48,
            diffusion_steps=80, num_regime_states=3, dropout=0.0,
        )
        base.update(kw)
        return TDTConfig(**base)

    def test_mask_disabled_by_default(self):
        cfg = self._config()
        model = TDT(cfg)
        # No call to set_causal_adjacency -> bias is None.
        c = torch.tensor([0, 1])
        bias = model._attn_bias_for_condition(c)
        assert bias is None

    def test_set_causal_adjacency_validates_shape(self):
        cfg = self._config(use_causal_mask=True)
        model = TDT(cfg)
        bad = torch.zeros(2, cfg.num_historical_numeric, cfg.num_historical_numeric)
        with pytest.raises(ValueError):
            model.set_causal_adjacency(bad)
        bad_F = torch.zeros(cfg.num_regime_states, cfg.num_historical_numeric + 1,
                            cfg.num_historical_numeric)
        with pytest.raises(ValueError):
            model.set_causal_adjacency(bad_F)

    def test_set_causal_adjacency_requires_flag(self):
        cfg = self._config(use_causal_mask=False)
        model = TDT(cfg)
        good = torch.zeros(cfg.num_regime_states, cfg.num_historical_numeric,
                           cfg.num_historical_numeric)
        with pytest.raises(RuntimeError):
            model.set_causal_adjacency(good)

    def test_mask_changes_eps_after_training(self):
        """At init the AdaLN-Zero gates are zero so the attention path
        contributes nothing — mask vs no-mask are identical by design. After
        a few training steps the gates open and the mask must measurably
        change eps_pred. This is the property we actually care about."""
        torch.manual_seed(0)
        cfg = self._config(use_causal_mask=True, causal_mask_scale=2.0)
        masked = TDT(cfg)
        adj = torch.randn(cfg.num_regime_states, cfg.num_historical_numeric, cfg.num_historical_numeric)
        masked.set_causal_adjacency(adj)
        # Train a few steps so gates are nonzero.
        opt = torch.optim.Adam(masked.parameters(), lr=1e-2)
        x0 = torch.randn(4, cfg.encoder_steps, cfg.num_historical_numeric)
        c = torch.tensor([0, 1, 2, 2])
        for _ in range(4):
            opt.zero_grad()
            masked.diffusion_step(x0, c)["loss_simple"].backward()
            opt.step()
        # Now compute eps with the mask, then turn the mask off and recompute.
        x_t = torch.randn(4, cfg.encoder_steps, cfg.num_historical_numeric)
        t = torch.full((4,), 10, dtype=torch.long)
        with_mask = masked.denoise(x_t, t, c)["eps_pred"]
        masked._causal_loaded = False
        without_mask = masked.denoise(x_t, t, c)["eps_pred"]
        assert not torch.allclose(with_mask, without_mask, atol=1e-6), (
            "Causal mask had no effect on eps prediction after training"
        )

    def test_diffusion_step_works_with_mask(self):
        cfg = self._config(use_causal_mask=True)
        model = TDT(cfg)
        adj = torch.randn(cfg.num_regime_states, cfg.num_historical_numeric, cfg.num_historical_numeric) * 0.1
        model.set_causal_adjacency(adj)
        x0 = torch.randn(3, cfg.encoder_steps, cfg.num_historical_numeric)
        c = torch.tensor([0, 1, 2])
        out = model.diffusion_step(x0, c)
        assert torch.isfinite(out["loss_simple"])
        assert out["loss_simple"].requires_grad


class TestGradients:

    def test_every_trainable_param_gets_a_grad(self, small_config):
        """Joint diffusion + class-head loss must produce a grad for every
        trainable parameter. Stage 1.J added a discriminative class_head that
        is *not* in the diffusion graph — without the BCE term it would never
        get a gradient. This test pins the joint training contract."""
        model = TDT(small_config)
        x0 = torch.randn(2, small_config.encoder_steps, small_config.num_historical_numeric, requires_grad=False)
        c = torch.tensor([0, 2])
        # Diffusion path (trains eps_pred, regime_logits, and the backbone).
        out_diff = model.diffusion_step(x0, c)
        # Classification path (trains class_head + backbone). We exercise it
        # directly via class_head_logit on a clean input; the joint loss
        # adds these together exactly as TDTCrisisModel.fit does.
        if model.class_head is not None:
            class_logit = model.denoise(x0, torch.zeros(2, dtype=torch.long), c)["class_logit"]
            y = torch.tensor([0.0, 1.0])
            l_clf = torch.nn.functional.binary_cross_entropy_with_logits(class_logit, y)
        else:
            l_clf = 0.0
        loss = out_diff["loss_simple"] + 0.1 * out_diff["regime_logits"].mean() + 0.5 * l_clf
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == [], f"params without grad: {missing}"
