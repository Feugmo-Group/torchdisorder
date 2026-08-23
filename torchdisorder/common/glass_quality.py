"""Decide whether a melt-quench structure is actually a glass.

Why this module exists
----------------------
Coordination number cannot tell a glass from a crystal that never melted, and
:func:`~torchdisorder.common.validation.validate_structure` is blind to the
difference too.  On 2026-08-17 that single gap produced three false positives in
one day across two material systems: GeO2 runs at 2400, 2200 and 2000 K all
reported <CN> = 4.00 with zero coordination defects and "PASSED", and all three
were unmelted crystal.

The reason is structural.  If the crystal already contains the target motif --
isolated PS4 in Li3PS4, corner-sharing GeO4 in GeO2 -- then a structure that
never melted scores a perfect <CN> with no defects, which looks *better* than a
real glass.  Perfection is the warning sign rather than the goal: DFT BOMD of
Li3PS4 glass gives PS4 : P2S6 : P2S7 ~ 6 : 2 : 1, so "~100% PS4" means
under-melted.  The melt-quench script has always said as much in its own output
("a passing health check is NOT proof of a glass -- a hot crystal also passes"),
and a printed warning turned out to be no substitute for a check.

Two independent tests, both required
------------------------------------
``chemistry``
    Species that should not exist, counted explicitly.  A mean hides them: seven
    free O2 molecules in a 3000-atom GeO2 cell move <CN> by less than 0.01.  The
    rules are per-system because the forbidden species are -- molecular O2 at
    1.21 A in an oxide, P-P pairs and P-free sulfur in a thiophosphate, both of
    which are the signature of a universal potential reducing P(V) to P(IV).

``disorder``
    The central-atom sublattice g(r), judged *beyond 6 A*.  Long-range order is
    what melting destroys; first-shell peaks survive in both phases and so
    discriminate nothing.  Two numbers do the work -- the tallest peak anywhere
    in g(r), and the standard deviation of g(r) past ``r_long``.

The measured separation is an order of magnitude, so this is decisive rather
than a judgement call.  Every structure in data/crystal-structures, measured by
this module at its default settings:

    structure                  max g   std(r>6)   O2   verdict
    c-SiO2                     25.50      2.303    0   crystal
    sio2_from_crystal           13.60      1.709    0   crystal (a 0.3 A rattle)
    SiO2_mq                    10.96      0.456    0   partially melted
    SiO2_mq_hot                 4.63      0.134    1   glass, wrong chemistry
    SiO2_mq_hot_mpa             4.54      0.143    0   GLASS
    sio2_glass_gap (published)  4.61      0.134    0   GLASS
    c-GeO2                     26.30      2.152    0   crystal
    GeO2_mq                     5.46      0.135   11   glass, wrong chemistry
    geo2_glass_nnp (published)  5.12      0.130    0   GLASS

Both published reference models land within 0.01 of our own best melt-quench
structures on ``std``, which is the evidence that the measure is meaningful and
not an artefact of our pipeline.  The ``sio2_from_crystal`` row is worth reading
twice: rattling the crystal moves std from 2.303 only to 1.709, barely a tenth of
the way to the glass value of 0.134, while wrecking the coordination on the way.
Displacement degrades a crystal without disordering it, which is why rattling is
not one of the routes to an amorphous model.

Raw std is not comparable between chemistries
---------------------------------------------
The table above is all oxides, whose central sublattice runs to 375-1728 atoms.
A Li7P3S11 cell of 672 atoms contains only 96 P, and a g(r) built from 96 centres
is far noisier -- so a *perfectly disordered* thiophosphate scores a std that
would read as ordered on the oxide scale.  Measuring the counting-noise floor
directly, by randomising positions in the same cell (:func:`noise_floor`):

    structure                  N_a    std   noise   ratio   truth
    sio2_glass_gap            1728  0.134   0.039    3.47   glass
    SiO2_mq_hot_mpa            375  0.143   0.079    1.80   glass
    geo2_glass_nnp            1080  0.130   0.049    2.66   glass
    lips70 (LiPS-25)            96  0.504   0.309    1.63   disordered
    Li3PS4_gamma               108  3.930   0.285   13.79   crystal
    c-SiO2                     375  2.303   0.072   31.97   crystal
    c-GeO2                     375  2.152   0.078   27.55   crystal

lips70's std of 0.504 is almost entirely shot noise: its excess over the floor is
*lower than every published glass reference*.  A flat 0.5 ceiling would have
rejected a genuinely disordered structure for being dilute.

But the ratio alone will not do either, because a glass holds std ~ 0.13 while
noise falls as 1/sqrt(N_a) -- so the ratio grows with cell size, and the same
silica glass scores 1.80 at 375 Si and 3.47 at 1728 Si.  A large enough glass
would fail a fixed ratio test.

The criterion is therefore ``std < max(LONG_STD_CEILING, 2 x noise_floor)``:
the absolute term governs dense sublattices, the noise term rescues dilute ones,
and neither rescues a crystal, which sits 14-32x over its own floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GlassReport",
    "GLASS_SYSTEMS",
    "assess_glass",
    "partial_rdf",
    "sublattice_disorder",
    "forbidden_contacts",
    "noise_floor",
    "speciation",
]

# Placed in the observed gap between glass (4-5, 0.12-0.13) and crystal
# (12-13, 1.4-1.6).
#
# Both must pass, and each catches cases the other misses: a partially melted
# SiO2 run measures std = 0.456, sneaking under the std ceiling, and is caught
# only by max g = 10.96.  Conversely GeO2 at 2800 K is perfectly disordered on
# both counts and fails on chemistry alone.
#
# max_g depends on DEFAULT_DR: a crystal's peaks are near-delta, so halving the
# bin width nearly doubles their height (c-SiO2 measures 12.96 at dr = 0.1 and
# 25.50 at dr = 0.05).  The ceiling is calibrated at DEFAULT_DR -- change one and
# you must recalibrate the other.  std beyond r_long is far less bin-sensitive,
# which is a second reason to treat it as the primary discriminator.
MAX_G_CEILING = 8.0
LONG_STD_CEILING = 0.5
DEFAULT_DR = 0.05

# Below this radius, g(r) is dominated by the first and second coordination
# shells, which are equally sharp in a glass and a crystal.
DEFAULT_R_LONG = 6.0


@dataclass(frozen=True)
class ForbiddenRule:
    """A contact or an environment that should not occur in a good structure.

    Two shapes, distinguished by ``kind``:

    ``"contact"``
        Pairs of ``a``-``b`` closer than ``rmax``.  Detects molecular species --
        O2 at 1.21 A, S2 at 1.89 A -- and homonuclear bonds that signal a
        reduced oxidation state, such as P-P in a thiophosphate.
    ``"unbonded"``
        Atoms of ``a`` with no ``b`` neighbour within ``rmax``.  Detects a
        network former shedding its ligands: free S(2-) in Li-P-S.
    """

    kind: str
    a: str
    b: str
    rmax: float
    label: str


# Per-system glass-quality rules.  Note this is deliberately *not* the literature
# table in scripts/compare_to_literature.py -- that one holds published bond
# lengths and angles for scoring a refinement, this one holds the failure modes
# each chemistry actually exhibits under melt-quench.
GLASS_SYSTEMS: dict[str, dict] = {
    "SiO2": {
        "sublattice": "Si",
        "forbidden": [
            # Molecular O2 sits at 1.21 A; a peroxide O-O bridge at ~1.5 A is
            # also not part of a silica network.  1.35 A separates both from the
            # shortest physical O...O contact in a tetrahedron (2.63 A).
            ForbiddenRule("contact", "O", "O", 1.35, "O2 molecules"),
        ],
    },
    "GeO2": {
        "sublattice": "Ge",
        "forbidden": [
            ForbiddenRule("contact", "O", "O", 1.35, "O2 molecules"),
        ],
    },
    "LiPS": {
        "sublattice": "P",
        "forbidden": [
            # A P-P bond is the fingerprint of P(V) -> P(IV) reduction, which
            # every universal potential tried so far does to a Li-P-S melt.
            # Real P2S6(4-) does contain a P-P bond at ~2.2 A, so this counts
            # occurrences rather than forbidding them outright -- see the note
            # on `tolerated` below.
            ForbiddenRule("contact", "P", "P", 2.8, "P-P pairs"),
            # An S-S bond at ~2.05 A is polysulfide: the potential has oxidised
            # sulfide to S2(2-) or an S-S bridge. Added after LiPS-25 lips70,
            # where 12 S-S bonds at 2.03-2.07 A accounted for 16 of the 18 P-free
            # sulfurs -- so "P-free sulfur" was detecting the consequence while
            # naming the wrong cause. Non-bonded S...S contacts start above 3.2 A,
            # so 2.4 A separates a real bond cleanly.
            ForbiddenRule("contact", "S", "S", 2.4, "S-S bonds (polysulfide)"),
            # Sulfur with no phosphorus neighbour has left the network as free
            # S(2-). Kept alongside the S-S rule because the two failures are
            # different: sulfur can detach without pairing up.
            ForbiddenRule("unbonded", "S", "P", 2.8, "P-free sulfur"),
        ],
        # Reported beside the gates, never gated on -- see `speciation`. Present
        # because the counts above cannot tell a real P2S6(4-) unit from P(V)
        # reduction, and for Li-P-S that distinction is the whole question.
        "speciation": {"central": "P", "ligand": "S"},
    },
}

# Expected *glass* coordination of the central atom by its neighbour.
#
# This is not the crystal value, and the difference rejected two good runs. A
# crystal target of 4.00 +/- 0.15 failed lips75 at 3.848 and lips70 at 3.771 --
# the first by 0.0025 -- when both sit in the range a real glass should occupy.
#
# For a-Li3PS4, DFT BOMD gives PS4 : P2S6 : P2S7 ~ 6 : 2 : 1 by unit. Counting per
# phosphorus: PS4 contributes 6 P at CN 4, P2S6 4 P at CN 3 (three S and one P),
# P2S7 2 P at CN 4. So <CN> = (24 + 12 + 8) / 12 = 3.67, and a structure at 4.00
# would be the under-melted one.
#
# None means "no published glass speciation for this composition" -- the check is
# then skipped rather than being run against an invented number, and the glass
# gate carries the verdict. Do not fill these in without a source.
_GLASS_CN = {
    # a-SiO2 and a-GeO2 are ~99% 4-fold; the tetrahedron survives vitrification.
    "SiO2": (4.0, 0.15),
    "GeO2": (4.0, 0.15),
    # Li3PS4: from the 6:2:1 BOMD speciation above. The tolerance is wider than
    # the oxides' because the speciation ratio itself is approximate.
    "Li3PS4": (3.67, 0.30),
    # Li7P3S11 and Li4P2S7: the crystals are all-4-fold (PS4 + P2S7), but no glass
    # speciation is available, so how far below 4 a good glass sits is unknown.
    "Li7P3S11": (None, None),
    "Li4P2S7": (None, None),
    "LiPS": (None, None),
}

for _system, (_cn, _tol) in _GLASS_CN.items():
    if _system in GLASS_SYSTEMS:
        GLASS_SYSTEMS[_system]["expected_cn"] = _cn
        GLASS_SYSTEMS[_system]["cn_tol"] = _tol

# Composition-specific entries share the Li-P-S rules but carry their own
# expected coordination, so they cannot simply alias one dict.
for _name in ("Li3PS4", "Li7P3S11", "Li4P2S7"):
    GLASS_SYSTEMS[_name] = {
        "sublattice": GLASS_SYSTEMS["LiPS"]["sublattice"],
        "forbidden": GLASS_SYSTEMS["LiPS"]["forbidden"],
        "expected_cn": _GLASS_CN[_name][0],
        "cn_tol": _GLASS_CN[_name][1],
    }

# The labels the run scripts actually use, mapped to the composition they mean.
for _alias, _target in (("lips75", "Li3PS4"), ("lips70", "Li7P3S11"),
                        ("lips67", "Li4P2S7"), ("LiPS_75", "Li3PS4"),
                        ("lips", "LiPS")):
    GLASS_SYSTEMS[_alias] = GLASS_SYSTEMS[_target]


@dataclass
class GlassReport:
    """Outcome of :func:`assess_glass`.  Falsey when the structure is not a glass.

    ``chemistry_ok`` and ``disorder_ok`` are reported separately on purpose: the
    two failures have different remedies.  Bad chemistry means the potential is
    wrong for the system (use a system-specific model), while bad disorder means
    the melt was too cool or too short (raise the temperature or lengthen it).
    Collapsing them into one boolean loses the diagnosis.
    """

    system: str
    sublattice: str
    max_g: float
    long_std: float
    r_long: float
    noise: float = float("nan")
    """Counting-noise floor of ``long_std`` for this sublattice size; nan if not
    measured.  ``long_std / noise`` is the size-independent read on disorder --
    glasses observed at 1.6-3.5, crystals at 14-32."""
    ceiling: float = float("nan")
    """The limit ``long_std`` was actually judged against."""
    counts: dict = field(default_factory=dict)
    """Rule label -> number of offending atoms/pairs found."""
    speciation: dict = field(default_factory=dict)
    """Unit -> fraction of central atoms in it, from :func:`speciation`.  Empty
    when the system has no speciation rule.  **Diagnostic only** -- it never
    contributes to the verdict, because published models of the same material
    disagree too much to support a threshold.  Read it beside the disorder gate."""
    chemistry_ok: bool = True
    disorder_ok: bool = True
    range_ok: bool = True
    """False when the cell is too small to judge disorder at all.  Distinct from
    ``disorder_ok`` so callers can tell "this is ordered" from "this could not be
    measured" -- the remedies are a hotter melt and a bigger supercell."""
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        verdict = "GLASS" if self else "NOT A GLASS"
        # Do not quote a limit against a number that was never measurable, or the
        # header reads as though the structure was judged and failed on it.
        if not self.range_ok:
            std = f"std(r > {self.r_long:g} A) = not measurable in this cell"
        elif np.isnan(self.noise):
            std = f"std(r > {self.r_long:g} A) = {self.long_std:.3f} (limit {self.ceiling:.3f})"
        else:
            # Quote the excess over noise, not just the raw std: it is the number
            # that is comparable between a 1728-atom oxide sublattice and 96 P.
            std = (f"std(r > {self.r_long:g} A) = {self.long_std:.3f} "
                   f"= {self.long_std / self.noise:.2f}x noise floor "
                   f"{self.noise:.3f} (limit {self.ceiling:.3f})")
        lines = [
            f"{verdict}  [{self.system}]  "
            f"{self.sublattice}-{self.sublattice} sublattice: "
            f"max g = {self.max_g:.2f} (limit {MAX_G_CEILING}), {std}"
        ]
        if self.counts:
            found = ", ".join(f"{v} {k}" for k, v in self.counts.items())
            lines.append(f"  chemistry: {found}")
        if self.speciation:
            # Labelled "diagnostic" in the output itself, so nobody reads a
            # speciation that looks wrong as though the verdict rested on it.
            spec = ", ".join(f"{k} {100 * v:.0f}%" for k, v in self.speciation.items())
            lines.append(f"  speciation (diagnostic, not gated): {spec}")
        for f in self.failures:
            lines.append(f"  - {f}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def partial_rdf(atoms, a, b, rmax: float = 10.0, dr: float = DEFAULT_DR):
    """Partial pair distribution function g_ab(r), periodic images included.

    Returns ``(r, g)`` with ``r`` at bin centres.  ``g`` tends to 1 at large r
    for a homogeneous structure, which is what makes the spread past
    ``DEFAULT_R_LONG`` comparable between systems of different density.
    """
    from ase.data import atomic_numbers
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    za = atomic_numbers[a] if isinstance(a, str) else int(a)
    zb = atomic_numbers[b] if isinstance(b, str) else int(b)

    n_a = int((z == za).sum())
    n_b = int((z == zb).sum())
    if n_a == 0 or n_b == 0:
        raise ValueError(f"structure contains no {a}-{b} pairs to correlate")

    nbins = max(round(rmax / dr), 1)

    # Build the neighbour list over ONLY the two species involved, in the same
    # cell, rather than over everything and masking afterwards. The pair count
    # goes as (density x N), so restricting SiO2 to its 1/3 Si costs about a
    # ninth as much for an identical result -- 13 s to 1.5 s on 1125 atoms, which
    # is what makes the noise floor and the directory audit affordable.
    keep = (z == za) | (z == zb)
    sub = atoms[keep]
    zs = sub.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", sub, float(rmax))
    m = (zs[i] == za) & (zs[j] == zb)
    hist, edges = np.histogram(d[m], bins=nbins, range=(0.0, float(rmax)))
    r = 0.5 * (edges[:-1] + edges[1:])

    # neighbor_list yields ordered pairs (both (i,j) and (j,i)), so the ideal-gas
    # count for n_a centres each seeing density rho_b is n_a * rho_b * 4 pi r^2 dr
    # with no factor of two -- correct for a == b as well, since the self-pair is
    # excluded by construction.
    rho_b = n_b / atoms.get_volume()
    shell = 4.0 * np.pi * r**2 * (edges[1] - edges[0])
    return r, hist / (n_a * rho_b * shell)


def sublattice_disorder(atoms, species, rmax: float = 10.0,
                        r_long: float = DEFAULT_R_LONG, dr: float = DEFAULT_DR) -> dict:
    """Long-range order in the ``species``-``species`` sublattice.

    ``max_g`` is the tallest peak anywhere in g(r); ``long_std`` is the standard
    deviation of g(r) beyond ``r_long``, which is the discriminating number.
    """
    r, g = partial_rdf(atoms, species, species, rmax=rmax, dr=dr)
    long = g[r > r_long]
    return {
        "r": r,
        "g": g,
        "max_g": float(g.max()),
        "long_std": float(long.std()) if long.size else float("nan"),
        "n_long_bins": int(long.size),
    }


def noise_floor(atoms, species, rmax: float = 10.0, r_long: float = DEFAULT_R_LONG,
                dr: float = DEFAULT_DR, seed: int = 0) -> float:
    """The std of g(r > r_long) attributable to counting noise alone.

    Measured by randomising every position while keeping the cell, the
    composition and the number of central atoms -- so the result is an ideal gas
    with this structure's sublattice size, whose g(r) is 1 everywhere apart from
    shot noise.  Anything above this floor is genuine structure.

    This matters because the raw std is not comparable between chemistries. A
    Li7P3S11 cell of 672 atoms holds only 96 P, and its P-P g(r) therefore has a
    noise floor of ~0.31, against ~0.04-0.08 for an oxide sublattice of 375-1728
    atoms. Judged on raw std against a single ceiling, a perfectly disordered
    thiophosphate reads as ordered purely because it is dilute.
    """
    rng = np.random.default_rng(seed)
    shuffled = atoms.copy()
    shuffled.set_scaled_positions(rng.random((len(atoms), 3)))
    return sublattice_disorder(shuffled, species, rmax=rmax, r_long=r_long,
                               dr=dr)["long_std"]


def forbidden_contacts(atoms, rules) -> dict:
    """Count occurrences of each :class:`ForbiddenRule`.

    Counts are of distinct pairs (for ``"contact"``) or of atoms (for
    ``"unbonded"``), never means -- a handful of bad species in a large cell is
    invisible in any average but still disqualifies the structure.
    """
    from ase.data import atomic_numbers
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    counts: dict[str, int] = {}
    for rule in rules:
        za = atomic_numbers[rule.a]
        zb = atomic_numbers[rule.b]
        if not (z == za).any():
            continue
        i, j, _d = neighbor_list("ijd", atoms, float(rule.rmax))
        m = (z[i] == za) & (z[j] == zb)
        if rule.kind == "contact":
            if za == zb:
                # Each pair appears twice; count i < j once.
                n = int((m & (i < j)).sum())
            else:
                n = int(m.sum())
        elif rule.kind == "unbonded":
            bonded = np.zeros(len(atoms), bool)
            bonded[i[m]] = True
            n = int((~bonded & (z == za)).sum())
        else:
            raise ValueError(f"unknown rule kind: {rule.kind!r}")
        counts[rule.label] = n
    return counts


def speciation(atoms, central: str = "P", ligand: str = "S",
               bond_cutoff: float = 2.5, homo_cutoff: float = 2.8) -> dict:
    """Classify every ``central`` atom by the structural unit it belongs to.

    Returns ``{"counts": {unit: n}, "fractions": {unit: f}, "n_central": int}``
    where the units are ``PS4``, ``P2S7``, ``P2S6`` and ``other``, and fractions
    are **per central atom** (not per unit -- see the note below, it is an easy
    factor-of-two trap).

    Why this exists
    ---------------
    It replaces reading raw defect counts as a chemistry verdict, which is wrong
    in two ways at once. Counts are *extensive*: the same a-Li3PS4 model shows 2
    P-P pairs in a 62-P cell and **121** in a 1111-P cell, so any absolute
    threshold is a statement about cell size rather than chemistry. And a count
    *conflates opposite phenomena*: the P-P bond in a genuine P2S6(4-) unit and
    the P-P bond left by a potential reducing P(V) to P(IV) are the same number
    and opposite meanings. Speciation fractions are intensive, are what 31P NMR
    measures, and separate the two cases by topology rather than by tally.

    Classification is by local topology alone, which is what makes it applicable
    to a structure that arrives as bare coordinates:

    ==========  =========================================================
    ``PS4``     4 ligands, none bridging, no homonuclear neighbour
    ``P2S7``    4 ligands, exactly 1 bridging, no homonuclear neighbour
    ``P2S6``    3 ligands, none bridging, exactly 1 homonuclear neighbour
    ``other``   anything else -- under-coordinated, over-bridged, chained
    ==========  =========================================================

    A ligand is *bridging* when it touches two or more central atoms.

    Validation
    ----------
    Checked atom-by-atom against a structure where the answer is known
    independently: the PCCP class2 force-field a-Li3PS4 model encodes its
    speciation in LAMMPS atom types, and this classifier reproduces all
    **1111 / 1111** P assignments (647 PS4, 242 P2S6, 222 P2S7) from geometry
    alone. That model's fixed bond topology makes it useless as *evidence* about
    what speciation a real glass has -- the answer was an input -- but it is
    exactly that fixed topology which makes it a ground-truth test fixture.

    Do not gate on this
    -------------------
    Published models of the *same* material disagree far too much to support a
    threshold: by P atom, a-Li3PS4 reads 58% PS4 (PCCP), 76% (Staacke, 500 K) and
    90% (Staacke, as-quenched), against ~50% implied by the 6:2:1 literature
    ratio. A structure at 90% PS4 is a genuine glass by g(r) and one at 98% is
    not, so speciation cannot separate them on its own. Report it beside the
    disorder gate; let a human read the two together.

    Note on 6:2:1
    -------------
    The published PS4 : P2S6 : P2S7 ratio counts *units*. Per phosphorus it is 6
    x 1P : 2 x 2P : 1 x 2P = 6 : 4 : 2 of 12 P, i.e. **50% / 33% / 17%** -- not
    67 / 22 / 11. Compare like with like.
    """
    from ase.neighborlist import neighbor_list

    z = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    units = ("PS4", "P2S7", "P2S6", "other")
    counts = dict.fromkeys(units, 0)
    n_central = int((z == central).sum())
    if n_central == 0:
        return {"counts": counts, "fractions": dict.fromkeys(units, float("nan")),
                "n_central": 0}

    i, j = neighbor_list("ij", atoms, float(bond_cutoff))
    is_bond = (z[i] == central) & (z[j] == ligand)
    # Ligands touching >= 2 central atoms are bridging; this is what separates
    # P2S7 (one bridging S) from an isolated PS4 with the same ligand count.
    central_per_ligand = np.bincount(j[is_bond], minlength=n)
    n_ligands = np.bincount(i[is_bond], minlength=n)
    bridging = is_bond & (central_per_ligand[j] >= 2)
    n_bridging = np.bincount(i[bridging], minlength=n)

    ii, jj = neighbor_list("ij", atoms, float(homo_cutoff))
    is_homo = (z[ii] == central) & (z[jj] == central)
    n_homo = np.bincount(ii[is_homo], minlength=n)

    for k in np.flatnonzero(z == central):
        lig, br, homo = n_ligands[k], n_bridging[k], n_homo[k]
        if homo == 0 and lig == 4 and br == 0:
            counts["PS4"] += 1
        elif homo == 0 and lig == 4 and br == 1:
            counts["P2S7"] += 1
        elif homo == 1 and lig == 3 and br == 0:
            counts["P2S6"] += 1
        else:
            counts["other"] += 1

    return {
        "counts": counts,
        "fractions": {u: counts[u] / n_central for u in units},
        "n_central": n_central,
    }


def _speciation_fractions(spec: dict | None, atoms) -> dict:
    """Speciation fractions for a system that defines them, else ``{}``.

    Split out of :func:`assess_glass` only to keep its branch count under the
    complexity limit; it carries no logic of its own.
    """
    if not spec or not spec.get("speciation"):
        return {}
    return speciation(atoms, **spec["speciation"])["fractions"]


def assess_glass(
    structure,
    system: str | None = None,
    *,
    sublattice: str | None = None,
    rmax: float = 10.0,
    r_long: float = DEFAULT_R_LONG,
    dr: float = DEFAULT_DR,
    max_g_ceiling: float = MAX_G_CEILING,
    long_std_ceiling: float = LONG_STD_CEILING,
    min_long_bins: int = 20,
    check_noise: bool = True,
    noise_multiple: float = 2.0,
    tolerated: dict | None = None,
) -> GlassReport:
    """Apply both glass tests and return a falsey report if either fails.

    Parameters
    ----------
    structure
        ASE ``Atoms``, a ``torch_sim`` state, or a path to a structure file.
    system
        Key into :data:`GLASS_SYSTEMS` (``"SiO2"``, ``"GeO2"``, ``"LiPS"``).  When
        omitted, the chemistry rules are skipped and only disorder is judged,
        which is weaker -- pass it whenever the system is known.
    sublattice
        Central species whose g(r) is measured.  Defaults to the system's.
    min_long_bins
        Minimum histogram bins required beyond ``r_long`` for the disorder test
        to mean anything.  Fewer is a failure, not a pass -- see the note in the
        body.
    tolerated
        Rule label -> count that is acceptable.  Needed because a real glass is
        not defect-free: amorphous Li3PS4 genuinely contains P2S6(4-), whose P-P
        bond the ``"P-P pairs"`` rule counts.  Anything above the allowance is a
        failure; the default of zero is right for the oxides.
    """
    from .validation import _atoms_from

    atoms = _atoms_from(structure)
    tolerated = dict(tolerated or {})

    spec = GLASS_SYSTEMS.get(system) if system else None
    if system and spec is None:
        raise ValueError(
            f"unknown system {system!r}; known: {sorted(set(GLASS_SYSTEMS))}"
        )
    species = sublattice or (spec or {}).get("sublattice")
    if species is None:
        raise ValueError("pass `system` or `sublattice` to name the central species")

    failures: list[str] = []
    warnings: list[str] = []

    # g(r) past L/2 sees the periodic replicas of the cell rather than genuine
    # medium-range order, so a small cell cannot support this test at all.
    #
    # This FAILS rather than warns, and the asymmetry is deliberate: with a 6 A
    # cell (crystalline Li3PS4 is 32 atoms) there is a single histogram bin past
    # r_long, whose standard deviation is exactly 0.000 -- which sails through
    # any ceiling. A gate whose entire purpose is to stop unmelted crystals
    # being reported as glass must not pass one because the cell was too small
    # to look at. Build a supercell instead.
    half_min_cell = 0.5 * float(np.min(atoms.cell.lengths()))
    rmax = min(rmax, half_min_cell)

    dis = sublattice_disorder(atoms, species, rmax=rmax, r_long=r_long, dr=dr)
    disorder_ok = True
    # nan rather than 0.0 when the cell is too small to reach the noise branch:
    # a reported floor of zero would look like a measurement that came back clean.
    floor = ceiling = float("nan")
    range_ok = r_long < half_min_cell and dis["n_long_bins"] >= min_long_bins
    if not range_ok:
        # Report *only* this. The std over zero bins is nan, and letting the
        # comparison below also fire would add "long-range order survived" --
        # a statement about the structure that was never actually measured.
        disorder_ok = False
        failures.append(
            f"cell half-width is {half_min_cell:.1f} A, leaving "
            f"{dis['n_long_bins']} bin(s) beyond r_long = {r_long:g} A -- too "
            f"little range to judge long-range order (need {min_long_bins}). "
            "Use a larger supercell; this is not a verdict on the structure."
        )
    else:
        if dis["max_g"] > max_g_ceiling:
            disorder_ok = False
            failures.append(
                f"{species}-{species} g(r) peaks at {dis['max_g']:.2f} "
                f"(> {max_g_ceiling}); a glass sits near 4-5, a crystal near 12-13"
            )
        # The ceiling is the LARGER of the absolute limit and a multiple of this
        # sublattice's own counting-noise floor. Both terms are needed:
        #
        #   - the noise term rescues dilute sublattices. lips70 measures std
        #     0.504 against a noise floor of 0.309 -- an excess of 1.63x, lower
        #     than every published glass reference -- so a flat 0.5 would reject
        #     a genuinely disordered structure for being made of only 96 P atoms.
        #   - the absolute term stops the noise term running away. Noise falls as
        #     1/sqrt(N_a) while a real glass keeps std ~ 0.13, so the *ratio*
        #     grows with cell size: the same silica glass scores 1.80 at 375 Si
        #     and 3.47 at 1728 Si. A pure ratio test would fail large glasses.
        #
        # Crystals are not rescued by either: they sit 14-32x over their floor.
        #
        # The floor is measured on EVERY assessment, including passes, and that
        # is worth one extra g(r) over a shuffled cell. It was once skipped when
        # it could not change the verdict -- the ceiling is max(absolute,
        # k*noise), so a std already under the absolute limit passes whatever the
        # noise turns out to be -- but a verdict is not the only output. Skipping
        # it left `noise` nan on exactly the runs that passed, so `summary()` fell
        # back to quoting the flat limit and every good result was reported
        # against the wrong yardstick. Two real misreadings came from this:
        #
        #   - lips70 @ 1200 K printed "std = 0.489 (limit 0.500)", reading as a
        #     2% squeak. Its floor is 0.312, so it passed at 1.57x excess against
        #     an effective ceiling of 0.624 -- a comfortable margin.
        #   - GeO2 with a 10 ps superheat printed "std = 0.346 (limit 0.500)" and
        #     was taken for a glass. At 4.04x its floor of 0.086 it is partially
        #     melted, well outside the 1.6-3.5x band glasses occupy.
        #
        # The ratio is the size-independent number and the only one comparable
        # between a 1728-atom oxide sublattice and 96 P; the raw std is not.
        # Report it always. Note this is deliberately reporting only -- the ratio
        # does NOT gate, because it grows with sublattice size (the same silica
        # glass scores 1.80 at 375 Si and 3.47 at 1728 Si), so any fixed ratio
        # threshold would repeat the size-blindness this whole block exists to
        # fix. Read it against the benchmarks; do not turn it into a limit.
        ceiling = long_std_ceiling
        if check_noise:
            floor = noise_floor(atoms, species, rmax=rmax, r_long=r_long, dr=dr)
            ceiling = max(long_std_ceiling, noise_multiple * floor)
        if not (dis["long_std"] < ceiling):
            disorder_ok = False
            detail = (f"> {ceiling:.3f} = {noise_multiple:g} x noise floor "
                      f"{floor:.3f}" if ceiling > long_std_ceiling
                      else f"> {long_std_ceiling}")
            failures.append(
                f"{species}-{species} g(r) beyond {r_long:g} A has std "
                f"{dis['long_std']:.3f} ({detail}); long-range order "
                "survived, so the melt did not destroy the crystal"
            )

    counts: dict[str, int] = {}
    chemistry_ok = True
    if spec:
        counts = forbidden_contacts(atoms, spec["forbidden"])
        for label, n in counts.items():
            allowed = int(tolerated.get(label, 0))
            if n > allowed:
                chemistry_ok = False
                extra = f" (allowance {allowed})" if allowed else ""
                failures.append(f"{n} {label}{extra}")

    return GlassReport(
        system=system or f"{species} sublattice only",
        sublattice=species,
        max_g=dis["max_g"],
        long_std=dis["long_std"],
        r_long=r_long,
        counts=counts,
        # Computed after the verdict is already settled and deliberately not fed
        # back into it: a diagnostic for whoever reads the report, not a
        # criterion. See `speciation` for why no threshold is defensible.
        speciation=_speciation_fractions(spec, atoms),
        chemistry_ok=chemistry_ok,
        disorder_ok=disorder_ok,
        range_ok=range_ok,
        noise=floor,
        ceiling=ceiling,
        failures=failures,
        warnings=warnings,
    )
