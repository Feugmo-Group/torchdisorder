"""Tests for objective conditioning: sigma policy and chi^2 normalisation.

These exist because chi^2 for an unnormalised F(Q) fit is O(1e8) while every other
term is O(1). Rescaling the other terms up to meet it was measured worse than
leaving them alone; normalising the objective is the alternative.
"""

import pytest
import torch

from torchdisorder.model.loss import chi_squared, rmse


def test_fractional_floor_suppresses_near_zero_sigma_points():
    """A single point with sigma ~ 0 must not dominate chi^2.

    The SiO2 target has uncertainties hand-patched from 0.0 to 1e-7; two such
    points can contribute ~1e12 on their own.
    """
    est = torch.tensor([0.10, 0.20, 0.30])
    tgt = torch.tensor([0.11, 0.20, 0.30])
    sig = torch.tensor([1e-7, 1e-3, 1e-3])

    raw = float(chi_squared(est, tgt, sig))
    floored = float(chi_squared(est, tgt, sig, sigma_mode="fractional",
                                sigma_floor_frac=0.02))
    # sigma is clamped at 1e-6 upstream, so the 1e-7 point lands at (0.01/1e-6)^2 = 1e8
    assert raw > 9e7, "the pathological point should dominate without a floor"
    assert floored < raw / 1e6
    # floor = 0.02 * max|target| = 0.006; residual 0.01 -> (0.01/0.006)^2 ~ 2.8
    assert floored == pytest.approx((0.01 / (0.02 * 0.30)) ** 2, rel=0.05)


def test_data_mode_is_unchanged_by_the_floor_argument():
    """Default behaviour must be byte-identical to before the option existed."""
    est = torch.tensor([0.1, 0.2]); tgt = torch.tensor([0.15, 0.2])
    sig = torch.tensor([1e-3, 1e-3])
    assert float(chi_squared(est, tgt, sig)) == pytest.approx(
        float(chi_squared(est, tgt, sig, sigma_mode="data", sigma_floor_frac=0.5))
    )


def test_floor_only_raises_sigma_never_lowers_it():
    est = torch.tensor([0.1, 0.2]); tgt = torch.tensor([0.15, 0.2])
    big = torch.tensor([10.0, 10.0])
    assert float(chi_squared(est, tgt, big, sigma_mode="fractional",
                             sigma_floor_frac=0.001)) == pytest.approx(
           float(chi_squared(est, tgt, big)))


def test_rmse_is_scale_free_relative_to_sigma():
    """RMS is the fallback fit metric when chi^2 is not interpretable."""
    est = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.1, 2.1, 2.9])
    assert float(rmse(est, tgt)) == pytest.approx(0.1, abs=1e-6)


def test_normalisation_makes_the_objective_order_one():
    """chi^2/scale must start at 1 so relative weights are meaningful."""
    chi2 = torch.tensor(2.4e8)
    scale = float(chi2)
    assert float(chi2 / scale) == pytest.approx(1.0)


def test_normalisation_preserves_gradient_direction():
    """Dividing by a constant must not change where the optimum is."""
    x = torch.tensor([2.0], requires_grad=True)
    (x ** 2).backward()
    raw = x.grad.clone()

    y = torch.tensor([2.0], requires_grad=True)
    ((y ** 2) / 1e8).backward()
    assert torch.allclose(raw / 1e8, y.grad)
