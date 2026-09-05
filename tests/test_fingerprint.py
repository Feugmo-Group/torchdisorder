"""The atom-order fingerprint must catch the pairing bug it was written for.

Constraints are keyed by atom index, so a constraints file is only meaningful
against the exact ordering it was generated from. `c-SiO2.cif` is the concrete
case: pymatgen expands its symmetry operations and groups sites by label, ASE
reads the literal atom_site order, and 510 of 1125 sites end up different.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from torchdisorder.constraints.fingerprint import (
    atom_order_fingerprint,
    verify_atom_order,
)


CIF = "data/crystal-structures/c-SiO2.cif"


def _quartz_like(n_si: int = 8, n_o: int = 16, seed: int = 0) -> Atoms:
    """A cell with two elements and no special symmetry, for permutation tests."""
    rng = np.random.default_rng(seed)
    n = n_si + n_o
    return Atoms(
        symbols="Si" * n_si + "O" * n_o,
        scaled_positions=rng.random((n, 3)),
        cell=np.eye(3) * 12.0,
        pbc=True,
    )


def test_a_structure_matches_its_own_fingerprint():
    atoms = _quartz_like()
    assert verify_atom_order(atom_order_fingerprint(atoms), atoms) == []


def test_cif_round_trip_is_not_flagged(tmp_path: object):
    """Writing and re-reading perturbs coordinates at ~1e-6; that is not a reorder."""
    atoms = _quartz_like()
    fp = atom_order_fingerprint(atoms)
    path = tmp_path / "rt.cif"
    atoms.write(path)
    assert verify_atom_order(fp, read(path)) == []


def test_rescaling_the_cell_is_not_flagged():
    """The trainer rescales to a target density; fractional coords are invariant."""
    atoms = _quartz_like()
    fp = atom_order_fingerprint(atoms)
    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * 1.15, scale_atoms=True)
    assert verify_atom_order(fp, scaled) == []


def test_a_same_element_permutation_is_caught():
    """The gap the element guard cannot close: Si constraints on the wrong Si."""
    atoms = _quartz_like()
    fp = atom_order_fingerprint(atoms)

    shuffled = atoms.copy()
    pos = shuffled.get_scaled_positions()
    si = [i for i, s in enumerate(shuffled.get_chemical_symbols()) if s == "Si"]
    pos[si] = pos[list(reversed(si))]
    shuffled.set_scaled_positions(pos)

    # Every atom is still the element the constraints expect — the old guard passes.
    assert shuffled.get_chemical_symbols() == atoms.get_chemical_symbols()
    problems = verify_atom_order(fp, shuffled)
    assert problems, "a same-element permutation must not pass silently"
    assert "reordered" in problems[0]


def test_a_cross_element_reorder_is_caught():
    atoms = _quartz_like()
    fp = atom_order_fingerprint(atoms)
    reordered = atoms[::-1]
    problems = verify_atom_order(fp, reordered)
    assert any("element sequence differs" in m for m in problems)


def test_a_different_atom_count_is_reported_on_its_own():
    """8206 vs 8204 makes every index meaningless; say that and stop."""
    atoms = _quartz_like()
    fp = atom_order_fingerprint(atoms)
    problems = verify_atom_order(fp, atoms[:-2])
    assert len(problems) == 1
    assert "atom count differs" in problems[0]


def test_spot_checks_are_drawn_from_the_constrained_indices():
    atoms = _quartz_like(n_si=8, n_o=16)
    si = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Si"]
    fp = atom_order_fingerprint(atoms, si)
    sampled = {entry[0] for entry in fp["spot_checks"]}
    assert sampled <= set(si)
    assert all(entry[1] == "Si" for entry in fp["spot_checks"])


def test_a_missing_fingerprint_is_not_a_failure():
    """Constraints files written before this check must still load."""
    assert verify_atom_order({}, _quartz_like()) == []
    assert verify_atom_order(None, _quartz_like()) == []


@pytest.mark.skipif(not __import__("pathlib").Path(CIF).exists(),
                    reason="c-SiO2.cif not present")
def test_the_real_pymatgen_ase_disagreement_is_caught():
    """The bug in the wild: same file, same 1125 atoms, different indexing."""
    pymatgen = pytest.importorskip("pymatgen.core")
    struct = pymatgen.Structure.from_file(CIF)
    atoms = read(CIF)

    assert len(struct) == len(atoms), "this test needs the equal-count case"
    pmg_symbols = [s.specie.symbol for s in struct]
    ase_symbols = atoms.get_chemical_symbols()
    disagree = sum(a != b for a, b in zip(pmg_symbols, ase_symbols, strict=True))
    assert disagree > 0, "c-SiO2.cif no longer exhibits the reorder"

    # A generator fingerprints what pymatgen sees; the trainer loads what ASE sees.
    fp = atom_order_fingerprint(struct)
    assert verify_atom_order(fp, atoms), (
        "the readers disagree on ordering and the fingerprint must say so"
    )
