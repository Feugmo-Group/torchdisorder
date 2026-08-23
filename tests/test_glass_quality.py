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
    speciation,
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


def _tripod(origin, axis, bond=2.05):
    """Three ligand positions completing a tetrahedron whose fourth bond is `axis`.

    The remaining three bonds sit at the tetrahedral angle from `axis`, i.e. with
    component -1/3 along it.  Used to build speciation fixtures analytically so
    they do not depend on any data file.
    """
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    # any two unit vectors perpendicular to `axis`
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    out = []
    for theta in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        d = -axis / 3.0 + np.sqrt(8.0) / 3.0 * (np.cos(theta) * u + np.sin(theta) * v)
        out.append(np.asarray(origin, float) + bond * d)
    return out


def _speciation_fixture():
    """One PS4, one P2S7 and one P2S6 unit, far apart in a big box.

    Built by construction rather than taken from a file, so the expected answer
    is known exactly: 1 P in PS4, 2 in P2S7, 2 in P2S6.
    """
    sym, pos = [], []

    # PS4: P at centre, four S at tetrahedral corners.
    d = 2.05 / np.sqrt(3.0)
    sym.append("P")
    pos.append([0.0, 0.0, 0.0])
    for c in ([d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]):
        sym.append("S")
        pos.append(c)

    # P2S7: two P bridged by a shared S, each carrying three terminal S.
    # `axis` is the bond the P already has, so it points from the P *towards* its
    # partner -- hence -sign here, not +sign. Getting this backwards aims the
    # tripod at the partner and the unit classifies as "other".
    o = np.array([15.0, 0.0, 0.0])
    sym.append("S")
    pos.append(o)                       # bridging S
    for sign in (-1.0, 1.0):
        p = o + np.array([sign * 2.05, 0.0, 0.0])
        sym.append("P")
        pos.append(p)
        for t in _tripod(p, [-sign, 0.0, 0.0]):
            sym.append("S")
            pos.append(t)

    # P2S6: two P bonded directly, each carrying three terminal S.
    o = np.array([0.0, 15.0, 0.0])
    for sign in (-1.0, 1.0):
        p = o + np.array([0.0, sign * 1.1, 0.0])
        sym.append("P")
        pos.append(p)
        for t in _tripod(p, [0.0, -sign, 0.0]):
            sym.append("S")
            pos.append(t)

    atoms = Atoms(symbols=sym, positions=np.array(pos), cell=[30.0] * 3, pbc=True)

    # Self-check the fixture: a silently malformed unit would make the classifier
    # look wrong when it is the test that is broken -- which is what happened.
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, 3.0)
    z = np.array(atoms.get_chemical_symbols())
    ps = d[(z[i] == "P") & (z[j] == "S")]
    ss = d[(z[i] == "S") & (z[j] == "S")]
    assert ps.min() == pytest.approx(2.05, abs=1e-6), f"P-S is {ps.min():.3f} A"
    # Empty is the good case here -- well-separated units have no S-S pair inside
    # 3 A at all -- so guard the reduction rather than letting it raise.
    assert ss.size == 0 or ss.min() > 2.4, f"S-S contact at {ss.min():.3f} A"
    return atoms


def test_speciation_classifies_the_three_units():
    """The classifier must separate PS4, P2S7 and P2S6 by topology alone.

    P2S7 and PS4 have the SAME ligand count and are told apart only by whether a
    ligand bridges
    P2S6 is the one with a P-P bond.  Collapsing any pair of
    these is what made raw defect counts useless as a chemistry verdict.
    """
    got = speciation(_speciation_fixture())
    assert got["n_central"] == 5
    assert got["counts"] == {"PS4": 1, "P2S7": 2, "P2S6": 2, "other": 0}


def test_speciation_fractions_are_per_central_atom():
    """Fractions must be per P, not per unit -- an easy factor-of-two error.

    The fixture holds three *units* but five *phosphorus atoms*, so PS4 is 1/5
    and not 1/3.  The published 6:2:1 ratio counts units and must be converted
    before comparison
    this test pins which convention the code uses.
    """
    fr = got = speciation(_speciation_fixture())["fractions"]
    assert fr["PS4"] == pytest.approx(1 / 5)
    assert fr["P2S7"] == pytest.approx(2 / 5)
    assert fr["P2S6"] == pytest.approx(2 / 5)
    assert sum(got.values()) == pytest.approx(1.0)


def test_speciation_never_changes_the_verdict():
    """Speciation is a diagnostic.  It must not gate, in either direction.

    Published models of a-Li3PS4 disagree from 58% to 90% PS4 by phosphorus, so
    no threshold is defensible
    a structure at 90% is a real glass by g(r) and
    one at 98% is not.  Turning a soft expectation into a hard criterion is how
    <CN> passed three unmelted crystals.
    """
    rng = np.random.default_rng(0)
    # A disordered, chemically clean Li-P-S cell built from intact PS4 units:
    # speciation reads 100% PS4, far outside every published ratio.
    sym, pos = [], []
    d = 2.05 / np.sqrt(3.0)
    corners = np.array([[d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]])
    # 45 A rather than 34: the 6.5 A exclusion below is a hard-sphere condition,
    # so at high density it builds a correlation peak of its own and max g climbs
    # over the 8.0 ceiling -- the fixture would then fail the disorder gate for a
    # reason that has nothing to do with what is being tested.
    L = 45.0
    centres = rng.random((90, 3)) * L

    def far(c, k):
        # Minimum image: without it two units can sit adjacent across the
        # periodic boundary, their ligands cross-coordinate, and the fixture
        # silently stops being all-PS4.
        dv = c - k
        dv -= L * np.round(dv / L)
        return np.linalg.norm(dv) > 6.5

    keep = []
    for c in centres:
        if all(far(c, k) for k in keep):
            keep.append(c)
    for c in keep:
        sym.append("P")
        pos.append(c)
        for corner in corners:
            sym.append("S")
            pos.append(c + corner)
    atoms = Atoms(symbols=sym, positions=np.array(pos), cell=[L] * 3, pbc=True)

    rep = assess_glass(atoms, "LiPS")
    # 100% PS4 is outside every published a-Li3PS4 ratio (58-90%), so if
    # speciation gated at all, this structure would be rejected.
    assert rep.speciation["PS4"] == pytest.approx(1.0), "fixture should be all PS4"
    assert rep.disorder_ok and rep.chemistry_ok, rep.summary()
    assert rep, "anomalous speciation must not reject a structure that passes both gates"
    assert not any("specia" in f.lower() for f in rep.failures)


def test_speciation_matches_an_independently_labelled_structure():
    """Ground truth: a model whose speciation was assigned, not inferred.

    The PCCP class2 force-field a-Li3PS4 encodes its speciation in LAMMPS atom
    types, giving 647 PS4 / 242 P2S6 / 222 P2S7 phosphorus.  Its fixed bond
    topology makes it useless as evidence about what a real glass contains --
    the answer was an input -- but that is exactly what makes it a test fixture.
    Skips when the file is absent: its licence is unresolved, so it is not
    committed.  See the note in .gitignore for where to fetch it.
    """
    path = DATA / "ref_aLi3PS4_pccp.cif"
    if not path.exists():
        pytest.skip("PCCP reference structure not present")
    from ase.io import read

    got = speciation(read(str(path)))["counts"]
    assert got == {"PS4": 647, "P2S6": 242, "P2S7": 222, "other": 0}
