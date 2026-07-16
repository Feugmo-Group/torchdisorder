"""Smoke tests for PyTorchOrderParameters.

Tests run on CPU with small synthetic structures so they need no GPU,
no MACE, and no experimental data files.
"""

import math
import pytest
import torch

# torch_sim has a known torchvision/torch version conflict in some envs.
# All tests in this module are skipped when either package is not importable.
_SKIP_REASON = None
try:
    import torch_sim as ts
    from torchdisorder.engine.order_params import PyTorchOrderParameters
except Exception as _e:
    _SKIP_REASON = str(_e)
    ts = None  # type: ignore
    PyTorchOrderParameters = None  # type: ignore

pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=f"torch_sim or order_params unavailable: {_SKIP_REASON}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(positions: torch.Tensor, cell_len: float = 10.0):
    """Wrap positions in a minimal SimState with a cubic cell."""
    n = positions.shape[0]
    cell = torch.eye(3) * cell_len
    atomic_numbers = torch.ones(n, dtype=torch.long)  # all H for simplicity
    return ts.SimState(
        positions=positions,
        cell=cell.unsqueeze(0),
        atomic_numbers=atomic_numbers,
        pbc=torch.tensor([True, True, True]),
        masses=atomic_numbers.float(),
    )


def _op_calc(cutoff: float = 4.0, max_neighbors: int = 16) -> PyTorchOrderParameters:
    return PyTorchOrderParameters(cutoff=cutoff, device="cpu", max_neighbors=max_neighbors)


# ---------------------------------------------------------------------------
# Coordination number
# ---------------------------------------------------------------------------

def test_cn_single_neighbor():
    """One central atom with one neighbour → CN = 1."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["cn"])
    assert result["cn"].shape == (1,)
    assert result["cn"][0].item() == pytest.approx(1.0)


def test_cn_tetrahedron():
    """Central atom at origin with 4 neighbours → CN = 4."""
    d = 2.0
    neighbours = torch.tensor([
        [d, 0.0, 0.0], [-d, 0.0, 0.0],
        [0.0, d, 0.0], [0.0, -d, 0.0],
    ])
    positions = torch.cat([torch.zeros(1, 3), neighbours])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["cn"])
    assert result["cn"][0].item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Tetrahedral order parameter
# ---------------------------------------------------------------------------

def test_tet_perfect_tetrahedron():
    """Perfect tetrahedral geometry → tet ≈ 1.0."""
    d = 1.6
    a = d / math.sqrt(3)
    neighbours = torch.tensor([
        [ a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [ a, -a, -a],
    ])
    positions = torch.cat([torch.zeros(1, 3), neighbours])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["tet"])
    assert result["tet"][0].item() > 0.8


def test_tet_linear():
    """Two collinear neighbours → tet << 1."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["tet"])
    assert result["tet"][0].item() < 0.3


# ---------------------------------------------------------------------------
# Distortion index (di)
# ---------------------------------------------------------------------------

def test_di_equal_bonds():
    """Equal bond lengths → di = 0."""
    d = 2.0
    neighbours = torch.tensor([
        [d, 0.0, 0.0], [-d, 0.0, 0.0],
        [0.0, d, 0.0], [0.0, -d, 0.0],
    ])
    positions = torch.cat([torch.zeros(1, 3), neighbours])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["di"])
    assert result["di"].shape == (1,)
    assert result["di"][0].item() == pytest.approx(0.0, abs=1e-5)


def test_di_unequal_bonds():
    """Unequal bond lengths → di > 0."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],   # centre
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],   # different distance
        [0.0, 2.0, 0.0],
    ])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["di"])
    assert result["di"][0].item() > 1e-3


def test_di_single_neighbor_is_zero():
    """Fewer than 2 neighbours → di clamped to 0 (std undefined)."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["di"])
    assert result["di"][0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SUPPORTED_TYPES completeness
# ---------------------------------------------------------------------------

def test_supported_types_no_tri_plan():
    """tri_plan/sq_plan/tri_pyr are not in SUPPORTED_TYPES (they returned zeros silently)."""
    unsupported = {"tri_plan", "sq_plan", "tri_pyr"}
    assert not unsupported.intersection(PyTorchOrderParameters.SUPPORTED_TYPES)


def test_di_in_supported_types():
    """di must appear in all three SUPPORTED_TYPES lists."""
    from torchdisorder.engine.order_params import TorchSimOrderParameters
    assert "di" in PyTorchOrderParameters.SUPPORTED_TYPES
    assert "di" in TorchSimOrderParameters.SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_tet_gradient_flows():
    """Gradient of tet w.r.t. positions must be non-None and finite."""
    d = 1.6
    a = d / math.sqrt(3)
    raw = torch.tensor([
        [0.0, 0.0, 0.0],
        [ a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [ a, -a, -a],
    ], requires_grad=False)
    positions = raw.clone().detach().requires_grad_(True)
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["tet"])
    result["tet"].sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_di_gradient_flows():
    """Gradient of di w.r.t. positions must be non-None and finite."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], requires_grad=True)
    state = _make_state(positions)
    calc = _op_calc(cutoff=3.0)
    idx = torch.tensor([0])
    result = calc(state, idx, ["di"])
    result["di"].sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()
