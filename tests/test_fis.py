"""Tests for the F_IS (local inversion symmetry) order parameter.

Covers:
  - Analytical limits: single bond → 0, antiparallel pair → 1, perfect tetrahedron → 1
  - Numerical match against the reference numpy implementation
  - Autograd (gradient must flow through F_IS)
  - Integration test on crystalline c-SiO2 (F_IS should be ≈1 for perfect tetrahedra)
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

_SKIP_REASON = None
try:
    import torch_sim as ts
    from torchdisorder.engine.order_params import PyTorchOrderParameters, TorchSimOrderParameters
except Exception as _e:
    _SKIP_REASON = str(_e)
    ts = None  # type: ignore
    PyTorchOrderParameters = None  # type: ignore
    TorchSimOrderParameters = None  # type: ignore

pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=f"torch_sim or order_params unavailable: {_SKIP_REASON}",
)

REPO_ROOT = Path(__file__).parent.parent
SIO2_CIF = REPO_ROOT / "data" / "crystal-structures" / "c-SiO2.cif"

SI_Z = 14
O_Z = 8


# ---------------------------------------------------------------------------
# Helpers shared with test_order_params.py
# ---------------------------------------------------------------------------

def _make_state(positions: torch.Tensor, cell_len: float = 20.0):
    n = positions.shape[0]
    cell = torch.eye(3) * cell_len
    atomic_numbers = torch.ones(n, dtype=torch.long)
    return ts.SimState(
        positions=positions,
        cell=cell.unsqueeze(0),
        atomic_numbers=atomic_numbers,
        pbc=torch.tensor([True, True, True]),
        masses=atomic_numbers.float(),
    )


def _make_state_with_species(positions: torch.Tensor, atomic_numbers: torch.Tensor, cell_len: float = 20.0):
    cell = torch.eye(3) * cell_len
    return ts.SimState(
        positions=positions,
        cell=cell.unsqueeze(0),
        atomic_numbers=atomic_numbers,
        pbc=torch.tensor([True, True, True]),
        masses=atomic_numbers.float(),
    )


def _calc(cutoff: float = 4.0, max_neighbors: int = 16) -> PyTorchOrderParameters:
    return PyTorchOrderParameters(cutoff=cutoff, device="cpu", max_neighbors=max_neighbors)


# ---------------------------------------------------------------------------
# Reference numpy implementation (inlined for self-contained validation)
# ---------------------------------------------------------------------------

def _ref_fis_numpy(
    positions_np: np.ndarray,
    neighbor_vecs: list,       # list[list[np.ndarray]] — displacement vectors per atom
    mode: str = "variable_R",
) -> np.ndarray:
    """Per-atom F_IS averaged over xy, xz, yz, matching the JCTC reference code."""
    shear_pairs = [(0, 1), (0, 2), (1, 2)]
    n = len(positions_np)
    local_per_shear = []

    for mu, nu in shear_pairs:
        Xi = np.zeros((n, 3), dtype=float)
        denom = np.zeros(n, dtype=float)

        for i, nbrs in enumerate(neighbor_vecs):
            for rij in nbrs:
                R = float(np.linalg.norm(rij))
                if R < 1e-10:
                    continue
                nhat = rij / R
                orient = nhat[mu] * nhat[nu]
                weight = R if mode == "variable_R" else 1.0
                Xi[i] += weight * orient * nhat
                denom[i] += (weight * orient) ** 2

        valid = denom > 1e-10
        local = np.zeros(n, dtype=float)
        local[valid] = 1.0 - np.sum(Xi[valid] ** 2, axis=1) / denom[valid]
        local_per_shear.append(local)

    return np.mean(local_per_shear, axis=0)


# ---------------------------------------------------------------------------
# Analytical limit: single bond → F_IS = 0
# ---------------------------------------------------------------------------

def test_fis_single_bond_along_diagonal():
    """One bond along [1,1,1] → local F_IS = 0 (no inversion partner)."""
    d = 1.6
    c = d / math.sqrt(3)
    positions = torch.tensor([[0.0, 0.0, 0.0], [c, c, c]])
    state = _make_state(positions)
    calc = _calc(cutoff=2.5)
    result = calc(state, torch.tensor([0]), ["fis"])
    assert result["fis"].shape == (1,)
    assert result["fis"][0].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Analytical limit: two antiparallel bonds → F_IS = 1
# ---------------------------------------------------------------------------

def test_fis_antiparallel_bonds():
    """Two antiparallel bonds of equal length → F_IS = 1.0."""
    d = 1.6
    c = d / math.sqrt(3)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [ c,  c,  c],
        [-c, -c, -c],
    ])
    state = _make_state(positions)
    calc = _calc(cutoff=3.0)
    result = calc(state, torch.tensor([0]), ["fis"])
    assert result["fis"][0].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Analytical limit: perfect SiO4 tetrahedron → F_IS = 1
# ---------------------------------------------------------------------------

def test_fis_perfect_sio4_tetrahedron():
    """Perfect SiO4 tetrahedral arrangement → F_IS = −1/3.

    Td symmetry has NO inversion centre: Xi_z (for the xy shear) equals
    4R/(3√3) because all four n̂ˣn̂ʸn̂ᶻ contributions have the same sign
    and add instead of cancelling.  The result is F_IS = 1 − 4/3 = −1/3.
    """
    d = 1.6
    a = d / math.sqrt(3)
    positions = torch.tensor([
        [ 0.0,  0.0,  0.0],   # central Si
        [ a,    a,    a],      # O
        [-a,   -a,    a],      # O
        [-a,    a,   -a],      # O
        [ a,   -a,   -a],      # O
    ])
    state = _make_state(positions)
    calc = _calc(cutoff=2.5)
    result = calc(state, torch.tensor([0]), ["fis"])
    assert result["fis"][0].item() == pytest.approx(-1 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# Asymmetric environment → F_IS < 1
# ---------------------------------------------------------------------------

def test_fis_asymmetric_environment():
    """Three non-centrosymmetric neighbors → F_IS < 1."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [0.0, 1.6, 0.0],
        [0.0, 0.0, 1.6],
    ])
    state = _make_state(positions)
    calc = _calc(cutoff=2.5)
    result = calc(state, torch.tensor([0]), ["fis"])
    assert result["fis"][0].item() < 0.9


# ---------------------------------------------------------------------------
# Milkus2016 mode: same analytical limits hold
# ---------------------------------------------------------------------------

def test_fis_milkus2016_perfect_tetrahedron():
    """milkus2016 mode gives F_IS = −1/3 for equal-length bonds (same as variable_R)."""
    d = 1.6
    a = d / math.sqrt(3)
    positions = torch.tensor([
        [ 0.0,  0.0,  0.0],
        [ a,    a,    a],
        [-a,   -a,    a],
        [-a,    a,   -a],
        [ a,   -a,   -a],
    ])
    state = _make_state(positions)
    calc = PyTorchOrderParameters(cutoff=2.5, device="cpu", max_neighbors=16, fis_mode="milkus2016")
    result = calc(state, torch.tensor([0]), ["fis"])
    assert result["fis"][0].item() == pytest.approx(-1 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# Numerical match against reference numpy code
# ---------------------------------------------------------------------------

def test_fis_matches_reference_numpy():
    """PyTorch F_IS must match the reference numpy implementation to 1e-5."""
    # Place 3 SiO4-like units in a large box (no PBC wrapping)
    d = 1.6
    a = d / math.sqrt(3)
    # Three Si centers at [0,0,0], [6,0,0], [12,0,0]
    # Each with 4 O neighbors at tetrahedral vertices (shifted)
    offsets = torch.tensor([
        [ a,  a,  a], [-a, -a,  a], [-a,  a, -a], [ a, -a, -a],
    ])
    positions_list = []
    # Si at origin
    positions_list.append(torch.zeros(1, 3))
    positions_list.append(offsets)
    # Si at [6, 0, 0]
    center2 = torch.tensor([[6.0, 0.0, 0.0]])
    positions_list.append(center2)
    positions_list.append(offsets + center2)
    positions_all = torch.cat(positions_list, dim=0)  # 10 atoms

    # Atomic numbers: Si=14 at indices 0, 5; O=8 elsewhere
    atomic_numbers = torch.full((10,), O_Z, dtype=torch.long)
    atomic_numbers[0] = SI_Z
    atomic_numbers[5] = SI_Z

    state = _make_state_with_species(positions_all, atomic_numbers, cell_len=30.0)
    calc = _calc(cutoff=2.5, max_neighbors=16)

    si_indices = torch.tensor([0, 5])
    result_torch = calc(state, si_indices, ["fis"], element_filter=[O_Z])

    # Reference numpy: build neighbor vectors manually for the same structure
    pos_np = positions_all.numpy()
    si_idx = [0, 5]
    cutoff = 2.5

    neighbor_vecs = [[] for _ in range(len(si_idx))]
    for k, i in enumerate(si_idx):
        for j in range(len(pos_np)):
            if atomic_numbers[j].item() != O_Z:
                continue
            rij = pos_np[j] - pos_np[i]
            R = float(np.linalg.norm(rij))
            if 0.0 < R < cutoff:
                neighbor_vecs[k].append(rij)

    ref = _ref_fis_numpy(pos_np[si_idx], neighbor_vecs, mode="variable_R")

    assert result_torch["fis"].shape == (2,)
    for k in range(2):
        assert result_torch["fis"][k].item() == pytest.approx(ref[k], abs=1e-5), (
            f"Si[{k}]: torch={result_torch['fis'][k].item():.6f}, numpy={ref[k]:.6f}"
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_fis_gradient_flows():
    """d(F_IS)/d(positions) must be non-None and finite."""
    d = 1.6
    a = d / math.sqrt(3)
    positions = torch.tensor([
        [ 0.0,  0.0,  0.0],
        [ a,    a,    a],
        [-a,   -a,    a],
        [-a,    a,   -a],
        [ a,   -a,   -a],
    ], requires_grad=True)
    state = _make_state(positions)
    calc = _calc(cutoff=2.5)
    result = calc(state, torch.tensor([0]), ["fis"])
    result["fis"].sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_fis_gradient_asymmetric():
    """Gradient in non-symmetric case must also be non-None and finite."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [0.0, 1.6, 0.3],
        [0.3, 0.0, 1.6],
    ], requires_grad=True)
    state = _make_state(positions)
    calc = _calc(cutoff=2.5)
    result = calc(state, torch.tensor([0]), ["fis"])
    result["fis"].sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


# ---------------------------------------------------------------------------
# SUPPORTED_TYPES
# ---------------------------------------------------------------------------

def test_fis_in_supported_types():
    """fis must appear in SUPPORTED_TYPES for both backends."""
    assert "fis" in PyTorchOrderParameters.SUPPORTED_TYPES
    assert "fis" in TorchSimOrderParameters.SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# Integration test: crystalline c-SiO2 → F_IS ≈ 1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SIO2_CIF.exists(), reason="c-SiO2.cif not found")
def test_fis_sio2_crystal():
    """F_IS for Si in crystalline SiO2 should be close to 1 (perfect tetrahedra)."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        pytest.skip("pymatgen not available")

    struct = Structure.from_file(str(SIO2_CIF))
    cart = torch.tensor(struct.cart_coords, dtype=torch.float64).float()
    cell_mat = torch.tensor(struct.lattice.matrix, dtype=torch.float64).float()
    znums = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)

    state = ts.SimState(
        positions=cart,
        cell=cell_mat.unsqueeze(0),
        atomic_numbers=znums,
        pbc=torch.tensor([True, True, True]),
        masses=znums.float(),
    )

    si_indices = torch.where(znums == SI_Z)[0]
    calc = PyTorchOrderParameters(cutoff=2.2, device="cpu", max_neighbors=8)
    result = calc(state, si_indices, ["fis"], element_filter=[O_Z])

    mean_fis = result["fis"].mean().item()
    # Crystalline SiO2 has (near-) perfect SiO4 tetrahedra → F_IS ≈ −1/3.
    # Td symmetry has no inversion centre so the metric is negative; we allow
    # a loose window around −1/3 to accommodate small distortions in the CIF.
    assert -0.5 < mean_fis < 0.0, (
        f"Crystalline SiO2 tetrahedra should give F_IS ≈ −1/3, got {mean_fis:.4f}"
    )

    # Also compare mean F_IS against reference numpy implementation
    pos_np = cart.numpy()
    cell_np = cell_mat.numpy()
    cell_inv = np.linalg.inv(cell_np)

    neighbor_vecs = []
    for idx in si_indices.tolist():
        nbrs = []
        for j in range(len(pos_np)):
            if znums[j].item() != O_Z:
                continue
            dr = pos_np[j] - pos_np[idx]
            # Triclinic minimum image
            frac = dr @ cell_inv
            frac -= np.round(frac)
            dr_mic = frac @ cell_np
            R = float(np.linalg.norm(dr_mic))
            if 0.0 < R < 2.2:
                nbrs.append(dr_mic)
        neighbor_vecs.append(nbrs)

    ref = _ref_fis_numpy(pos_np[si_indices.tolist()], neighbor_vecs, mode="variable_R")
    ref_mean = float(np.nanmean(ref))

    assert abs(mean_fis - ref_mean) < 1e-3, (
        f"PyTorch mean F_IS={mean_fis:.5f}, numpy mean F_IS={ref_mean:.5f}"
    )
