"""Tests for structural plausibility validation.

The point of these checks is to catch structures that fit a scattering target while
being physically impossible, so the tests are built around geometries whose verdict is
unambiguous by construction rather than around archived files.
"""

import numpy as np
import pytest
from ase import Atoms

from torchdisorder.common.validation import (
    ValidationReport,
    coordination_profile,
    validate_structure,
)


def _sio4_network(n: int = 2, spacing: float = 8.0, bond: float = 1.61) -> Atoms:
    """A grid of isolated, well-separated SiO4 tetrahedra.

    Every Si has exactly four O at ``bond``, and the tetrahedra are far enough apart
    that no Si sees a neighbouring unit's O inside the plateau window.  That makes
    <CN> = 4 exactly, by construction, with no reliance on a data file.
    """
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    corners = corners / np.sqrt(3) * bond

    si_pos, o_pos = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                centre = np.array([ix, iy, iz]) * spacing + spacing / 2
                si_pos.append(centre)
                o_pos.extend(centre + corners)

    positions = np.vstack([np.array(si_pos), np.array(o_pos)])
    return Atoms(
        f"Si{len(si_pos)}O{len(o_pos)}",
        positions=positions,
        cell=[n * spacing] * 3,
        pbc=True,
    )


def test_clean_structure_passes():
    """A well-separated structure reports no failures and is truthy."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[12, 12, 12], pbc=True)
    report = validate_structure(atoms)
    assert report
    assert report.n_overlaps == 0
    assert report.failures == []
    assert report.summary().startswith("PASS")


def test_overlapping_atoms_fail():
    """Two atoms at 0.3 A are flagged; this is the archived-run failure mode."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [0.3, 0, 0]], cell=[12, 12, 12], pbc=True)
    report = validate_structure(atoms)
    assert not report
    assert report.n_overlaps == 1
    assert report.min_distance == pytest.approx(0.3, abs=1e-6)
    assert report.worst_ratio < 1.0
    assert report.summary().startswith("FAIL")


def test_overlap_counted_once_per_pair():
    """neighbor_list yields (i,j) and (j,i); the report must not double-count."""
    atoms = Atoms("Si3", positions=[[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0]],
                  cell=[12, 12, 12], pbc=True)
    report = validate_structure(atoms)
    # pairs (0,1), (1,2) and (0,2) are all within the covalent floor
    assert report.n_overlaps == 3


def test_overlap_pairs_report_species_and_are_sorted():
    atoms = Atoms("SiO2", positions=[[0, 0, 0], [0.4, 0, 0], [0.2, 0, 0]],
                  cell=[12, 12, 12], pbc=True)
    report = validate_structure(atoms)
    assert not report
    assert report.overlap_pairs, "expected the worst offenders to be listed"
    distances = [d for _, _, d, _ in report.overlap_pairs]
    assert distances == sorted(distances)
    for si, sj, d, floor in report.overlap_pairs:
        assert {si, sj} <= {"Si", "O"}
        assert d < floor


def test_overlap_tolerance_is_respected():
    """A borderline contact flips verdict with the tolerance, proving it is used."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[12, 12, 12], pbc=True)
    assert validate_structure(atoms, overlap_tol=0.6)      # 1.5 A > 0.6 * 2.22
    assert not validate_structure(atoms, overlap_tol=0.9)  # 1.5 A < 0.9 * 2.22


def test_plateau_holds_for_a_real_network():
    """A tetrahedral network keeps <CN> = 4 across the cutoff window."""
    report = validate_structure(
        _sio4_network(), check_plateau=True, central="Si", neighbour="O", expected_cn=4.0
    )
    assert report, report.summary()
    assert report.plateau_ok
    assert report.plateau is not None


def test_wrong_expected_cn_fails_even_when_geometry_is_clean():
    """Coordination is checked independently of overlap."""
    report = validate_structure(
        _sio4_network(), check_plateau=True, central="Si", neighbour="O", expected_cn=6.0
    )
    assert not report
    assert report.n_overlaps == 0
    assert any("expected 6.00" in f for f in report.failures)


def test_missing_central_species_is_reported():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [5, 0, 0]], cell=[12, 12, 12], pbc=True)
    report = validate_structure(atoms, check_plateau=True, central="Si", neighbour="O")
    assert not report
    assert any("coordination profile" in f for f in report.failures)


def test_coordination_profile_filters_by_species():
    """Si-centred Si-O coordination must not pick up Si-Si or O-centred neighbours."""
    profile = coordination_profile(_sio4_network(), central="Si", neighbour="O",
                                   cutoffs=(2.0, 2.2))
    assert set(profile) == {2.0, 2.2}
    assert all(v == pytest.approx(4.0) for v in profile.values())


def test_report_is_falsey_only_on_failure():
    ok = ValidationReport(n_atoms=1, formula="Si", density=1.0, min_distance=2.0,
                          n_overlaps=0, worst_ratio=2.0)
    bad = ValidationReport(n_atoms=1, formula="Si", density=1.0, min_distance=0.1,
                           n_overlaps=1, worst_ratio=0.1, failures=["overlap"])
    assert ok and not bad


def test_accepts_a_file_path(tmp_path):
    """Paths are accepted so the checker can be pointed at archived runs."""
    from ase.io import write

    path = tmp_path / "s.cif"
    write(str(path), _sio4_network(), format="cif")
    assert validate_structure(str(path))


def test_plateau_tolerates_cutoffs_below_the_first_shell():
    """The lowest cutoffs sit below the first-shell peak; that is not a failure.

    A physical a-SiO2 network gives <CN> = 3.29 at 1.8 A and 3.99/4.00/4.00/4.00
    above it.  Judging the raw max-min spread over the whole window rejects that
    perfectly good structure, so the check must locate the flat run instead.
    """
    from torchdisorder.common.validation import validate_structure

    profile = {1.8: 3.293, 2.0: 3.990, 2.2: 4.001, 2.4: 4.001, 2.6: 4.002}
    import torchdisorder.common.validation as V

    real = V.coordination_profile
    V.coordination_profile = lambda *a, **k: profile
    try:
        report = validate_structure(
            _sio4_network(), check_plateau=True, central="Si", neighbour="O",
            expected_cn=4.0,
        )
    finally:
        V.coordination_profile = real

    assert report, report.summary()
    assert report.plateau_ok


def test_plateau_rejects_a_monotonic_climb():
    """A structure with no first shell climbs steadily and must still fail."""
    from torchdisorder.common.validation import validate_structure
    import torchdisorder.common.validation as V

    # The withdrawn a-SiO2: <CN> rises with cutoff, never flattening.
    profile = {1.8: 2.149, 2.0: 2.517, 2.2: 2.984, 2.4: 3.459, 2.6: 3.880}
    real = V.coordination_profile
    V.coordination_profile = lambda *a, **k: profile
    try:
        report = validate_structure(
            _sio4_network(), check_plateau=True, central="Si", neighbour="O",
            expected_cn=4.0,
        )
    finally:
        V.coordination_profile = real

    assert not report
    assert not report.plateau_ok
