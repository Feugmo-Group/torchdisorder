"""Tests for the glass gate.

The cases that matter are the real structures in data/crystal-structures, since
the whole point of the module is to separate structures whose truth we already
know by other means.  Those live in ``test_known_structures`` and skip when the
data directory is absent.  The rest are built analytically so they pin the
normalisation and the fail-closed behaviour without any data dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from torchdisorder.common.glass_quality import (
    GLASS_SYSTEMS,
    assess_glass,
    forbidden_contacts,
    partial_rdf,
    sublattice_disorder,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "crystal-structures"


def random_cell(n=400, length=20.0, symbol="Si", seed=0):
    """An ideal gas: g(r) = 1 everywhere, the analytic reference for the norm."""
    rng = np.random.default_rng(seed)
    return Atoms(f"{symbol}{n}", positions=rng.random((n, 3)) * length,
                 cell=[length] * 3, pbc=True)


def test_partial_rdf_of_random_cell_is_unity():
    """Normalisation check: an uncorrelated cell must give g(r) = 1, not 2 or 0.5.

    This is the test that catches a factor-of-two error in the ordered-pair
    counting, which would otherwise shift every threshold in the module.
    """
    r, g = partial_rdf(random_cell(n=800, length=25.0), "Si", "Si", rmax=10.0)
    # Skip the smallest radii, where a handful of counts per bin makes the
    # estimate noisy for reasons that have nothing to do with normalisation.
    tail = g[r > 3.0]
    assert tail.mean() == pytest.approx(1.0, abs=0.05)
    assert tail.std() < 0.15


def test_random_cell_reads_as_disordered():
    """The ideal-gas limit is the most disordered thing there is."""
    rep = assess_glass(random_cell(n=800, length=25.0), sublattice="Si")
    assert rep, rep.summary()
    assert rep.disorder_ok


def test_crystal_reads_as_ordered():
    """A perfect crystal must fail the disorder test, at a comfortable margin."""
    # 6x6x6 diamond Si: big enough that r_long = 6 A is well inside L/2.
    crystal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((6, 6, 6))
    rep = assess_glass(crystal, sublattice="Si")
    assert not rep
    assert not rep.disorder_ok
    dis = sublattice_disorder(crystal, "Si", rmax=16.0)
    assert dis["long_std"] > 1.0
    assert dis["max_g"] > 10.0


def test_small_cell_fails_closed():
    """A cell too small to judge must FAIL, never pass.

    With a 6 A cell there is at most one histogram bin beyond r_long, whose std
    is exactly 0.0 and would sail under any ceiling.  Crystalline Li3PS4 is 32
    atoms in a 6 A cell, so this is the real configuration, not a contrivance.
    """
    tiny = bulk("Si", "diamond", a=5.43, cubic=True)
    rep = assess_glass(tiny, sublattice="Si")
    assert not rep
    assert not rep.disorder_ok
    assert any("supercell" in f for f in rep.failures)


def test_forbidden_contact_counts_pairs_once():
    """Two O2 molecules are two, not four -- neighbour lists double-count."""
    atoms = Atoms("O4", positions=[[0, 0, 0], [1.21, 0, 0],
                                   [0, 6, 0], [1.21, 6, 0]],
                  cell=[20, 20, 20], pbc=True)
    counts = forbidden_contacts(atoms, GLASS_SYSTEMS["SiO2"]["forbidden"])
    assert counts["O2 molecules"] == 2


def test_forbidden_unbonded_counts_atoms():
    """One sulfur in range of the P, one stranded -> exactly one offender."""
    atoms = Atoms("PSS", positions=[[0, 0, 0], [2.0, 0, 0], [9, 9, 9]],
                  cell=[20, 20, 20], pbc=True)
    counts = forbidden_contacts(atoms, GLASS_SYSTEMS["LiPS"]["forbidden"])
    assert counts["P-free sulfur"] == 1
    assert counts["P-P pairs"] == 0


def test_chemistry_failure_is_independent_of_disorder():
    """A perfectly disordered cell still fails if it carries forbidden species.

    This is the GeO2-at-2800-K case: flawless disorder, seven O2 molecules, and
    the reason chemistry and disorder are reported as separate booleans.
    """
    atoms = random_cell(n=600, length=25.0, symbol="Ge")
    # Glue an O2 dumbbell into a gap; the Ge sublattice is untouched.
    atoms += Atoms("O2", positions=[[12.4, 12.4, 12.4], [13.61, 12.4, 12.4]])
    rep = assess_glass(atoms, "GeO2")
    assert not rep
    assert rep.disorder_ok, "Ge sublattice was not modified"
    assert not rep.chemistry_ok
    assert rep.counts["O2 molecules"] == 1


def test_tolerated_allowance():
    """A real Li3PS4 glass contains P2S6, so its P-P bonds must be allowable.

    The phosphorus here sits on a 5 A grid rather than at random positions, and
    that matters: the P-P rule has a 2.8 A range, so a random P gas dense enough
    to give a usable g(r) generates hundreds of accidental "P-P bonds" and the
    count stops meaning anything.  Real Li3PS4 has a P number density of
    0.0063/A^3 -- a mean P-P separation of 5.4 A -- so close P pairs are genuinely
    rare and the rule is only meaningful at that dilution.  Only chemistry is
    asserted below; a grid is crystalline and fails the disorder test by design.
    """
    grid = np.stack(np.meshgrid(*[np.arange(6) * 5.0] * 3, indexing="ij"), -1)
    atoms = Atoms(f"P{6**3}", positions=grid.reshape(-1, 3),
                  cell=[30.0] * 3, pbc=True)
    assert forbidden_contacts(atoms, GLASS_SYSTEMS["LiPS"]["forbidden"])[
        "P-P pairs"] == 0, "grid spacing must exceed the 2.8 A P-P rule range"

    atoms += Atoms("P", positions=[[2.2, 0.0, 0.0]])  # one deliberate P-P bond
    strict = assess_glass(atoms, "LiPS")
    assert not strict.chemistry_ok
    assert strict.counts["P-P pairs"] == 1

    lenient = assess_glass(atoms, "LiPS", tolerated={"P-P pairs": 5})
    assert lenient.chemistry_ok, lenient.summary()


def test_unknown_system_is_rejected():
    with pytest.raises(ValueError, match="unknown system"):
        assess_glass(random_cell(), "NaCl")


@pytest.mark.parametrize(
    ("filename", "system", "expect_glass"),
    [
        ("c-SiO2.cif", "SiO2", False),
        ("sio2_from_crystal.cif", "SiO2", False),   # rattled crystal
        ("SiO2_mq.cif", "SiO2", False),             # partially melted
        ("SiO2_mq_hot.cif", "SiO2", False),         # disordered, 1 O2
        ("SiO2_mq_hot_mpa.cif", "SiO2", True),
        ("sio2_glass_gap.cif", "SiO2", True),       # published GAP model
        ("c-GeO2.cif", "GeO2", False),
        ("GeO2_mq.cif", "GeO2", False),             # disordered, 11 O2
        ("geo2_glass_nnp.cif", "GeO2", True),       # published NNP model
    ],
)
def test_known_structures(filename, system, expect_glass):
    """Every structure whose true phase we know independently."""
    from ase.io import read

    path = DATA / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    rep = assess_glass(read(str(path)), system)
    assert bool(rep) is expect_glass, rep.summary()


def test_published_models_agree_with_our_best():
    """Our best melt-quench structure must match a published model's disorder.

    If this drifts, the measure has become an artefact of our own pipeline rather
    than a property of glass.
    """
    from ase.io import read

    ours = DATA / "SiO2_mq_hot_mpa.cif"
    published = DATA / "sio2_glass_gap.cif"
    if not (ours.exists() and published.exists()):
        pytest.skip("reference structures not present")
    a = sublattice_disorder(read(str(ours)), "Si")["long_std"]
    b = sublattice_disorder(read(str(published)), "Si")["long_std"]
    assert a == pytest.approx(b, abs=0.05), f"ours {a:.3f} vs published {b:.3f}"
