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


def test_polysulfide_is_detected():
    """An S-S bond at 2.05 A is polysulfide and must be caught.

    From LiPS-25 lips70: 12 such bonds at 2.03-2.07 A. Non-bonded S...S contacts
    in an intact thiophosphate start above 3.2 A, so the two are far apart and the
    rule needs no tuning. Both sulfurs here are bonded to the P, so the older
    "P-free sulfur" rule sees nothing -- this rule is not redundant with it.
    """
    # Equilateral triangle of side 2.05: both sulfurs are bonded to the P, so the
    # older rule is satisfied, and yet they are bonded to each other.
    atoms = Atoms("PSS", positions=[[0, 0, 0],
                                    [2.05, 0, 0],
                                    [2.05 * 0.5, 2.05 * np.sqrt(3) / 2, 0]],
                  cell=[20, 20, 20], pbc=True)
    counts = forbidden_contacts(atoms, GLASS_SYSTEMS["LiPS"]["forbidden"])
    assert counts["S-S bonds (polysulfide)"] == 1
    assert counts["P-free sulfur"] == 0, "both S are within 2.8 A of the P"


def test_intact_thiophosphate_has_no_polysulfide():
    """A clean PS4 tetrahedron must not trip the S-S rule.

    Its S...S edges sit at ~3.3 A. If the cutoff were set much above 2.4 A this
    would fire on every good structure, so this test pins the upper bound.
    """
    # Ideal PS4: P at centre, 4 S at 2.05 A along tetrahedral directions.
    d = 2.05 / np.sqrt(3.0)
    corners = np.array([[d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]])
    atoms = Atoms("PS4", positions=np.vstack([[0, 0, 0], corners]),
                  cell=[20, 20, 20], pbc=True)
    edges = [np.linalg.norm(corners[m] - corners[n])
             for m in range(4) for n in range(m + 1, 4)]
    assert min(edges) > 3.2, f"PS4 S...S edge is {min(edges):.2f} A"
    counts = forbidden_contacts(atoms, GLASS_SYSTEMS["LiPS"]["forbidden"])
    assert counts["S-S bonds (polysulfide)"] == 0
    assert counts["P-free sulfur"] == 0
    assert counts["P-P pairs"] == 0


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


def test_noise_floor_falls_with_sublattice_size():
    """Counting noise must scale down as the sublattice grows.

    This is the whole justification for the noise-aware ceiling: the floor is a
    property of how many centres you have, not of the material.
    """
    from torchdisorder.common.glass_quality import noise_floor

    small = noise_floor(random_cell(n=100, length=25.0), "Si")
    large = noise_floor(random_cell(n=1600, length=25.0), "Si")
    assert small > large, f"{small:.3f} should exceed {large:.3f}"
    # Shot noise is 1/sqrt(pair count), and a same-species g(r) has pair count
    # going as n^2 -- both the number of centres and their density scale with n.
    # So the floor falls as 1/n, and 16x the atoms is ~16x quieter, not 4x.
    # Measured 0.289 -> 0.017, a factor of 17.
    assert 8.0 < small / large < 32.0, f"ratio {small / large:.2f} not ~16x"


def test_dilute_sublattice_is_not_called_ordered_for_being_noisy():
    """A dilute but genuinely random sublattice must pass on disorder.

    This is the lips70 case: 96 P gives a noise floor of ~0.31, so its std lands
    near 0.5 with no order present at all. Judged on the absolute ceiling alone it
    would be rejected; the noise term is what prevents that.

    40 atoms in a 26 A cell rather than lips70's exact density, because the test
    has to clear 0.5 on noise alone for the noise path to be taken at all -- at
    lips70's density a purely random cell sits at 0.29, under the ceiling, and
    passes without ever consulting the floor.
    """
    from torchdisorder.common.glass_quality import LONG_STD_CEILING, noise_floor

    dilute = random_cell(n=40, length=26.0, symbol="Si")
    floor = noise_floor(dilute, "Si")
    assert floor > 0.25, f"fixture must be noisy enough to matter, got {floor:.3f}"

    rep = assess_glass(dilute, sublattice="Si")
    assert rep.long_std > LONG_STD_CEILING, (
        f"fixture std {rep.long_std:.3f} must exceed the absolute ceiling, or "
        "this test passes without exercising the noise term at all")
    assert rep.disorder_ok, rep.summary()
    assert rep.ceiling > LONG_STD_CEILING, "noise term should have raised the ceiling"


def test_noise_term_does_not_rescue_a_crystal():
    """The dilute-sublattice allowance must not become a loophole for crystals.

    Li3PS4_gamma has essentially the same noise floor as lips70 (0.302 vs 0.309)
    and must still fail, by an order of magnitude.
    """
    crystal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((5, 5, 5))
    # Thin the sublattice to make it as noisy as a real thiophosphate.
    del crystal[[k for k in range(len(crystal)) if k % 8]]
    rep = assess_glass(crystal, sublattice="Si")
    assert not rep.disorder_ok, rep.summary()


def test_glass_coordination_is_composition_specific():
    """The expected CN is a glass number per composition, not the crystal's 4.

    a-Li3PS4 sits at 3.67 by the DFT 6:2:1 speciation, and the crystal target of
    4.00 +/- 0.15 rejected two real runs -- lips75 at 3.848, by 0.0025. Where no
    glass speciation is published the value must be None so the check is skipped
    rather than run against a guess.
    """
    assert GLASS_SYSTEMS["Li3PS4"]["expected_cn"] == pytest.approx(3.67, abs=0.01)
    assert GLASS_SYSTEMS["lips75"] is GLASS_SYSTEMS["Li3PS4"]
    assert GLASS_SYSTEMS["Li7P3S11"]["expected_cn"] is None
    assert GLASS_SYSTEMS["lips70"] is GLASS_SYSTEMS["Li7P3S11"]
    # The oxides keep the crystal value, because their tetrahedron does survive.
    assert GLASS_SYSTEMS["SiO2"]["expected_cn"] == pytest.approx(4.0)

    # Compositions must not share a dict, or setting one would set the others.
    assert GLASS_SYSTEMS["Li3PS4"] is not GLASS_SYSTEMS["Li7P3S11"]
    # ...but they do share the same chemistry rules.
    assert (GLASS_SYSTEMS["Li3PS4"]["forbidden"]
            is GLASS_SYSTEMS["Li7P3S11"]["forbidden"])


def test_glass_cn_accepts_a_defective_network_the_crystal_target_rejects():
    """3.85 is a glass; 4.00 +/- 0.15 calls it broken.

    Guards the specific regression: a P network with a realistic defect
    population must pass the glass criterion and fail the crystal one.
    """
    from torchdisorder.common.validation import validate_structure

    # 3 P per 4 with a full 4 S shell, 1 with 3 -> <CN> = 3.75, a glassy value.
    positions, symbols = [], []
    for k in range(4):
        centre = np.array([6.0 * k, 0.0, 0.0])
        positions.append(centre)
        symbols.append("P")
        n_s = 4 if k < 3 else 3
        d = 2.05 / np.sqrt(3.0)
        for sx, sy, sz in [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)][:n_s]:
            positions.append(centre + np.array([sx, sy, sz]))
            symbols.append("S")
    atoms = Atoms(symbols, positions=positions, cell=[30, 30, 30], pbc=True)

    crystal_target = validate_structure(atoms, check_plateau=True, central="P",
                                        neighbour="S", expected_cn=4.0,
                                        cn_tol=0.15, bond_cutoff=2.5)
    glass_target = validate_structure(atoms, check_plateau=True, central="P",
                                      neighbour="S", expected_cn=3.67,
                                      cn_tol=0.30, bond_cutoff=2.5)
    assert not crystal_target, "crystal target should reject a defective glass"
    assert glass_target, glass_target.summary()


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
