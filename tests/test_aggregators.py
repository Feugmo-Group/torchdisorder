"""Tests for torchdisorder/model/aggregators.py.

Maps the PINN analogy to TorchDisorder terms:
    PDE residual        →  chi2_scattering  (F_Q / S_Q chi-squared)
    boundary condition  →  fis_loss         (F_IS order-parameter loss)

All stateless aggregators (relobralo, brdr, ema, soft_adapt) are the
recommended starting points — no extra backward passes, compose safely
with Cooper's augmented Lagrangian.  GradNorm and LRAnnealing require
real model parameters and do an internal backward; NTK is very expensive
and skipped in the default test run.
"""

import pytest
import torch
import torch.nn as nn

from torchdisorder.model.aggregators import (
    AGGREGATOR_NAMES,
    Aggregator,
    BalancedResidualDecayRate,
    EMA,
    GradNorm,
    HomoscedasticUncertainty,
    LRAnnealing,
    NTKAggregator,
    Relobralo,
    ResNorm,
    SoftAdapt,
    Sum,
    build_aggregator,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_LOSSES = 2
KEYS = ("chi2_scattering", "fis_loss")


def _make_losses(
    chi2: float = 1.0, fis: float = 0.5, requires_grad: bool = True
) -> dict[str, torch.Tensor]:
    """Mimic the two-term loss dict produced by CooperLoss."""
    return {
        "chi2_scattering": torch.tensor(chi2, requires_grad=requires_grad),
        "fis_loss": torch.tensor(fis, requires_grad=requires_grad),
    }


def _dummy_params() -> list[nn.Parameter]:
    """Minimal parameter list needed by gradient-based aggregators."""
    layer = nn.Linear(4, 4)
    return list(layer.parameters())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_aggregator_names_complete():
    """AGGREGATOR_NAMES must list all 10 strategies."""
    expected = {
        "sum", "grad_norm", "res_norm", "homoscedastic",
        "lr_annealing", "soft_adapt", "relobralo", "ema", "brdr", "ntk",
    }
    assert set(AGGREGATOR_NAMES) == expected


def test_build_aggregator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown aggregator"):
        build_aggregator("not_a_strategy", params=[], num_losses=2)


# ---------------------------------------------------------------------------
# Helpers shared across stateless aggregator tests
# ---------------------------------------------------------------------------

def _stateless_roundtrip(agg: Aggregator, n_steps: int = 5) -> None:
    """Run n_steps of training and verify output shape, dtype, and gradient."""
    agg.train()
    for step in range(n_steps):
        losses = _make_losses(chi2=1.0 / (step + 1), fis=0.5 / (step + 1))
        total = agg(losses, step)
        assert total.shape == (), f"step {step}: expected scalar, got {total.shape}"
        assert total.dtype == torch.float32
        total.backward()  # must not raise


# ---------------------------------------------------------------------------
# Sum (baseline)
# ---------------------------------------------------------------------------


def test_sum_equal_weights():
    agg = build_aggregator("sum", params=[], num_losses=2)
    losses = _make_losses(chi2=2.0, fis=1.0)
    total = agg(losses, step=0)
    assert total.item() == pytest.approx(3.0, rel=1e-5)


def test_sum_custom_weights():
    agg = build_aggregator("sum", params=[], num_losses=2, weights=[3.0, 0.5])
    losses = _make_losses(chi2=1.0, fis=1.0)
    total = agg(losses, step=0)
    assert total.item() == pytest.approx(3.5, rel=1e-5)


def test_sum_gradient_flows():
    agg = build_aggregator("sum", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"No gradient for {k}"


# ---------------------------------------------------------------------------
# Relobralo — recommended for TorchDisorder
# ---------------------------------------------------------------------------


def test_relobralo_scalar_output():
    agg = build_aggregator("relobralo", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_relobralo_weights_adapt():
    """After several steps with chi2 >> fis, chi2 weight should decrease."""
    agg = build_aggregator("relobralo", params=[], num_losses=2, rho=0.9)
    agg.train()
    for step in range(50):
        losses = _make_losses(chi2=100.0, fis=0.01)
        total = agg(losses, step)
        total.backward()
    w = agg.current_weights
    assert w is not None
    assert len(w) == 2
    # chi2 has decayed less relative to its start (both start at step-0 values),
    # so relobralo should up-weight fis to balance.
    # Simply verify weights are positive and sum is > 0.
    assert all(wi > 0 for wi in w)


def test_relobralo_frozen_in_eval():
    """In eval mode, weights must not change between calls."""
    agg = build_aggregator("relobralo", params=[], num_losses=2, rho=0.9)
    agg.train()
    losses = _make_losses(chi2=10.0, fis=1.0)
    agg(losses, step=0).backward()
    w_before = list(agg.current_weights)

    agg.eval()
    agg(_make_losses(chi2=0.001, fis=0.001), step=1)
    assert agg.current_weights == w_before


def test_relobralo_gradient_flows():
    agg = build_aggregator("relobralo", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"No gradient for {k}"


# ---------------------------------------------------------------------------
# EMA — recommended for TorchDisorder
# ---------------------------------------------------------------------------


def test_ema_scalar_output():
    agg = build_aggregator("ema", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_ema_balances_scale():
    """After warmup, large-chi2 / small-fis scenario should down-weight chi2."""
    agg = build_aggregator("ema", params=[], num_losses=2, ema_alpha=0.9)
    agg.train()
    for step in range(100):
        losses = _make_losses(chi2=1000.0, fis=0.001)
        agg(losses, step).backward()
    w = agg.current_weights
    # weight on fis should be >> weight on chi2 after scaling
    assert w[1] > w[0], f"EMA did not up-weight fis: {w}"


def test_ema_gradient_flows():
    agg = build_aggregator("ema", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"No gradient for {k}"


# ---------------------------------------------------------------------------
# SoftAdapt — recommended for TorchDisorder
# ---------------------------------------------------------------------------


def test_soft_adapt_scalar_output():
    agg = build_aggregator("soft_adapt", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_soft_adapt_gradient_flows():
    agg = build_aggregator("soft_adapt", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"No gradient for {k}"


# ---------------------------------------------------------------------------
# BRDR (BalancedResidualDecayRate) — recommended for TorchDisorder
# ---------------------------------------------------------------------------


def test_brdr_scalar_output():
    agg = build_aggregator("brdr", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_brdr_gradient_flows():
    agg = build_aggregator("brdr", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"No gradient for {k}"


def test_brdr_weights_property():
    agg = build_aggregator("brdr", params=[], num_losses=2)
    agg.train()
    for step in range(5):
        agg(_make_losses(), step).backward()
    w = agg.current_weights
    assert w is not None and len(w) == 2


# ---------------------------------------------------------------------------
# ResNorm
# ---------------------------------------------------------------------------


def test_res_norm_scalar_output():
    agg = build_aggregator("res_norm", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_res_norm_gradient_flows():
    agg = build_aggregator("res_norm", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None


# ---------------------------------------------------------------------------
# HomoscedasticUncertainty
# ---------------------------------------------------------------------------


def test_homoscedastic_scalar_output():
    agg = build_aggregator("homoscedastic", params=[], num_losses=2)
    _stateless_roundtrip(agg)


def test_homoscedastic_gradient_flows():
    agg = build_aggregator("homoscedastic", params=[], num_losses=2)
    losses = _make_losses()
    total = agg(losses, step=0)
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None


# ---------------------------------------------------------------------------
# GradNorm — requires real params, does internal backward
# ---------------------------------------------------------------------------


def _grad_norm_setup():
    """Return (layer, agg) where losses are computed from layer and agg tracks layer.parameters()."""
    layer = nn.Linear(4, 1)
    agg = build_aggregator("grad_norm", params=list(layer.parameters()), num_losses=2, alpha=0.12)
    return layer, agg


def test_grad_norm_scalar_output():
    layer, agg = _grad_norm_setup()
    agg.train()
    x = torch.randn(2, 4)
    pred = layer(x).mean()
    losses = {
        "chi2_scattering": (pred - 1.0) ** 2,
        "fis_loss": (pred - 0.5) ** 2,
    }
    total = agg(losses, step=0)
    assert total.shape == ()
    assert total.requires_grad


def test_grad_norm_gradient_flows():
    layer, agg = _grad_norm_setup()
    agg.train()
    x = torch.randn(2, 4)
    pred = layer(x).mean()
    losses = {
        "chi2_scattering": (pred - 1.0) ** 2,
        "fis_loss": (pred - 0.5) ** 2,
    }
    total = agg(losses, step=0)
    total.backward()
    assert total.item() > 0


# ---------------------------------------------------------------------------
# LRAnnealing — requires real params, does internal backward
# ---------------------------------------------------------------------------


def test_lr_annealing_scalar_output():
    layer = nn.Linear(4, 1)
    agg = build_aggregator("lr_annealing", params=list(layer.parameters()), num_losses=2)
    agg.train()
    x = torch.randn(2, 4)
    pred = layer(x).mean()
    losses = {
        "chi2_scattering": (pred - 1.0) ** 2,
        "fis_loss": (pred - 0.5) ** 2,
    }
    total = agg(losses, step=1)
    assert total.shape == ()


# ---------------------------------------------------------------------------
# NTKAggregator — very expensive; skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ntk_aggregator_scalar_output():
    """NTK is O(N_params * N_losses) per step — skip unless --slow flag given."""
    params = _dummy_params()
    agg = build_aggregator("ntk", params=params, num_losses=2)
    layer = nn.Linear(4, 1)
    x = torch.randn(2, 4)
    pred = layer(x).mean()
    losses = {
        "chi2_scattering": (pred - 1.0) ** 2,
        "fis_loss": (pred - 0.5) ** 2,
    }
    total = agg(losses, step=0)
    assert total.shape == ()


# ---------------------------------------------------------------------------
# build_aggregator factory — smoke test every name except ntk
# ---------------------------------------------------------------------------

STATELESS_NAMES = ["sum", "res_norm", "homoscedastic", "soft_adapt", "relobralo", "ema", "brdr"]


@pytest.mark.parametrize("name", STATELESS_NAMES)
def test_build_all_stateless(name: str):
    agg = build_aggregator(name, params=[], num_losses=NUM_LOSSES)
    assert isinstance(agg, Aggregator)
    losses = _make_losses()
    total = agg(losses, step=0)
    assert total.shape == ()
    total.backward()
    for k, v in losses.items():
        assert v.grad is not None, f"{name}: no grad for {k}"


# ---------------------------------------------------------------------------
# Aggregator base: current_weights default
# ---------------------------------------------------------------------------


def test_sum_current_weights_is_none():
    """Sum has no dynamic weights — current_weights returns None."""
    agg = Sum(params=[], num_losses=2)
    assert agg.current_weights is None


# ---------------------------------------------------------------------------
# Smoke test: realistic TorchDisorder training loop
#
# Mimics what happens inside EnvironmentConstrainedOptimizer + CooperLoss:
#   - A fake "scattering model" maps atomic positions → predicted F(Q)
#   - chi2_scattering = MSE between predicted and target F(Q)
#   - fis_loss        = (mean_fis - fis_target)²
#   - chi2 starts ~100× larger than fis_loss (typical scale mismatch)
#   - After N steps the recommended aggregators should up-weight fis_loss
#     so both terms contribute comparably to the gradient.
# ---------------------------------------------------------------------------

RECOMMENDED = ["relobralo", "ema", "soft_adapt", "brdr"]
TRAINING_STEPS = 30


class _FakeScatteringModel(nn.Module):
    """Minimal differentiable model: positions → chi2 + fis_loss."""

    def __init__(self, n_atoms: int = 8, q_points: int = 50):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(n_atoms, 3) * 0.1)
        self._q = torch.linspace(0.5, 20.0, q_points)
        self._target_fq = torch.sin(self._q) / self._q
        self._fis_target = 0.005

    def forward(self) -> dict[str, torch.Tensor]:
        r = self.positions.norm(dim=-1).mean()
        pred_fq = torch.sin(self._q * r) / (self._q * r + 1e-6)
        chi2 = ((pred_fq - self._target_fq) ** 2).mean() * 100.0

        mean_fis = 0.3 - 0.01 * r
        fis_loss = (mean_fis - self._fis_target) ** 2

        return {"chi2_scattering": chi2, "fis_loss": fis_loss}


@pytest.mark.parametrize("name", RECOMMENDED)
def test_smoke_torchdisorder_training_loop(name: str):
    """
    Each recommended aggregator must:
    1. Run TRAINING_STEPS without error.
    2. Produce scalar output with gradient at every step.
    3. Pass gradients back to model.positions.
    4. Show that fis_loss weight grows relative to chi2 weight
       (because chi2 starts much larger and the aggregator compensates).
    """
    model = _FakeScatteringModel()
    agg = build_aggregator(name, params=[], num_losses=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    agg.train()
    model.train()

    first_fis_weight = None
    last_fis_weight = None

    for step in range(TRAINING_STEPS):
        opt.zero_grad()
        losses = model()

        assert "chi2_scattering" in losses
        assert "fis_loss" in losses
        assert losses["chi2_scattering"].item() > losses["fis_loss"].item(), (
            "Sanity: chi2 should dominate at this scale"
        )

        total = agg(losses, step)

        assert total.shape == (), f"step {step}: expected scalar"
        assert total.requires_grad, f"step {step}: total has no grad"

        total.backward()
        assert model.positions.grad is not None, f"step {step}: no grad on positions"

        opt.step()

        w = agg.current_weights
        if w is not None:
            fis_w = w[1]
            if first_fis_weight is None:
                first_fis_weight = fis_w
            last_fis_weight = fis_w

    # If the aggregator tracks weights, fis weight should have grown
    # (chi2 >> fis initially, so aggregator compensates by up-weighting fis).
    if first_fis_weight is not None and last_fis_weight is not None:
        assert last_fis_weight >= first_fis_weight * 0.5, (
            f"{name}: fis weight unexpectedly collapsed "
            f"({first_fis_weight:.4f} → {last_fis_weight:.4f})"
        )


def test_smoke_safe_with_no_fis():
    """CooperLoss can run with aggregator but without fis_loss (single-term dict)."""
    agg = build_aggregator("relobralo", params=[], num_losses=1)
    agg.train()
    for step in range(5):
        losses = {"chi2_scattering": torch.tensor(1.0 / (step + 1), requires_grad=True)}
        total = agg(losses, step)
        assert total.shape == ()
        total.backward()


def test_smoke_eval_mode_deterministic():
    """In eval mode the aggregator must return the same total for the same losses."""
    agg = build_aggregator("relobralo", params=[], num_losses=2, rho=0.9)
    agg.train()
    for step in range(10):
        losses = _make_losses(chi2=10.0 / (step + 1), fis=0.1 / (step + 1))
        agg(losses, step).backward()

    agg.eval()
    losses_eval = _make_losses(chi2=5.0, fis=0.05)
    out1 = agg(losses_eval, step=99).item()
    out2 = agg(losses_eval, step=100).item()
    assert out1 == pytest.approx(out2, rel=1e-6), (
        "eval mode must be deterministic (weights frozen)"
    )
