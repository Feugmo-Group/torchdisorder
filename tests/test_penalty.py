"""Tests for the adaptive constraint penalty.

The penalty exists to stop the optimizer buying spectral agreement with local
geometry.  It can only do that if it actually moves, and if it can reach a
magnitude comparable to the objective -- both of which previously failed.
"""

import pytest
import torch

from torchdisorder.engine.optimizer import ScalarAdaptivePenalty


def _penalty(**kw):
    kw.setdefault("device", "cpu")
    return ScalarAdaptivePenalty(**kw)


def test_ratio_targeting_raises_penalty_toward_the_objective_scale():
    """With chi2 ~1e7 and tiny violations, the penalty must climb by orders of magnitude.

    The old defaults left the constraint term at ~1e-6 of chi2, so constraints were
    decorative.  Repeated updates should approach target_ratio * chi2 / violation.
    """
    p = _penalty(init=10.0, target_ratio=0.05, max_penalty=1e9)
    chi2, viol_sq = 2.2e7, 15.7      # magnitudes measured on a real SiO2 run
    for _ in range(200):
        p.update(viol_sq, chi2=chi2)

    expected = 0.05 * chi2 / viol_sq
    assert p.current_penalty == pytest.approx(expected, rel=0.05)
    assert p.current_penalty > 1e4, "penalty must reach the scale of the objective"


def test_ratio_targeting_lowers_an_overlarge_penalty():
    """Targeting is two-sided; an overshoot must come back down."""
    p = _penalty(init=1e8, target_ratio=0.05, max_penalty=1e9)
    chi2, viol_sq = 2.2e7, 15.7
    for _ in range(200):
        p.update(viol_sq, chi2=chi2)
    assert p.current_penalty == pytest.approx(0.05 * chi2 / viol_sq, rel=0.05)


def test_ratio_targeting_moves_gradually():
    """A single update must not jump orders of magnitude on a transient spike."""
    p = _penalty(init=10.0, target_ratio=0.05, adapt_rate=0.15, max_penalty=1e9)
    p.update(15.7, chi2=2.2e7)
    # desired is ~7e4; one step at adapt_rate 0.15 should be a modest multiple
    assert 10.0 < p.current_penalty < 100.0


def test_ceiling_warns_when_it_binds():
    """A max_penalty that silently caps the penalty is how constraints went inert."""
    p = _penalty(init=10.0, target_ratio=0.05, max_penalty=1e3)
    with pytest.warns(UserWarning, match="pinned at max_penalty"):
        for _ in range(200):
            p.update(15.7, chi2=2.2e7)
    assert p.current_penalty == pytest.approx(1e3)


def test_zero_violation_does_not_divide_by_zero():
    p = _penalty(init=10.0, target_ratio=0.05)
    p.update(0.0, chi2=2.2e7)
    assert p.current_penalty == pytest.approx(10.0)


def test_without_target_ratio_the_improvement_rule_still_applies():
    """Legacy behaviour must survive for anyone who sets target_ratio: null."""
    p = _penalty(init=10.0, target_ratio=None, patience=2, growth_rate=2.0,
                 max_penalty=1e6)
    for _ in range(10):          # never improving -> penalty grows
        p.update(5.0)
    assert p.current_penalty > 10.0

    q = _penalty(init=100.0, target_ratio=None, decay_rate=0.5, min_penalty=1.0)
    for v in (100.0, 50.0, 25.0, 12.0):   # steadily improving -> penalty decays
        q.update(v)
    assert q.current_penalty < 100.0


def test_call_returns_current_value_as_tensor():
    p = _penalty(init=10.0, target_ratio=0.05, max_penalty=1e9)
    for _ in range(50):
        p.update(15.7, chi2=2.2e7)
    out = p(None)
    assert torch.is_tensor(out)
    assert float(out) == pytest.approx(p.current_penalty)


# ---------------------------------------------------------------------------
# Aggregator-driven balancing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["relobralo", "soft_adapt", "ema", "res_norm"])
def test_aggregator_modes_drive_the_penalty_to_the_objective_scale(name):
    """Any usable aggregator must lift the penalty off its init value.

    The failure being guarded is a penalty frozen at init=10 while chi2 is 1e7 --
    the state every run was in before update() was called at all.
    """
    p = _penalty(init=10.0, aggregator=name, max_penalty=1e9)
    assert p._aggregator is not None, f"{name} should have been constructed"
    for _ in range(100):
        p.update(15.7, chi2=2.2e7)
    assert p.current_penalty > 1e3, (
        f"{name}: penalty stayed at {p.current_penalty:.3g}, constraints would be inert"
    )


def test_gradient_based_aggregator_falls_back_rather_than_crashing():
    """grad_norm needs a graph; it gets detached floats, so it must degrade safely."""
    p = _penalty(init=10.0, aggregator="grad_norm", target_ratio=0.05, max_penalty=1e9)
    for _ in range(100):
        p.update(15.7, chi2=2.2e7)
    # Falls back to the fixed-ratio rule and still reaches a useful magnitude.
    assert p.current_penalty > 1e3


def test_aggregator_takes_precedence_over_target_ratio():
    p = _penalty(init=10.0, aggregator="ema", target_ratio=0.05, max_penalty=1e9)
    assert p._aggregator is not None
    p.update(15.7, chi2=2.2e7)
    assert p._agg_step == 1, "the aggregator should have been consulted"


def test_aggregator_ignores_zero_violation():
    p = _penalty(init=10.0, aggregator="relobralo", max_penalty=1e9)
    p.update(0.0, chi2=2.2e7)
    assert p.current_penalty == pytest.approx(10.0)
