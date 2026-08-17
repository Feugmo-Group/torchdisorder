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
            # Sulfur with no phosphorus neighbour has left the network as free
            # S(2-), the other half of the same reduction.
            ForbiddenRule("unbonded", "S", "P", 2.8, "P-free sulfur"),
        ],
    },
}

# Aliases for the composition-specific names the LiPS runs actually use.
for _alias in ("Li3PS4", "Li7P3S11", "LiPS_75", "lips75", "lips"):
    GLASS_SYSTEMS[_alias] = GLASS_SYSTEMS["LiPS"]


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
    counts: dict = field(default_factory=dict)
    """Rule label -> number of offending atoms/pairs found."""
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
        std = (f"std(r > {self.r_long:g} A) = {self.long_std:.3f} "
               f"(limit {LONG_STD_CEILING})" if self.range_ok
               else f"std(r > {self.r_long:g} A) = not measurable in this cell")
        lines = [
            f"{verdict}  [{self.system}]  "
            f"{self.sublattice}-{self.sublattice} sublattice: "
            f"max g = {self.max_g:.2f} (limit {MAX_G_CEILING}), {std}"
        ]
        if self.counts:
            found = ", ".join(f"{v} {k}" for k, v in self.counts.items())
            lines.append(f"  chemistry: {found}")
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
    i, j, d = neighbor_list("ijd", atoms, float(rmax))
    m = (z[i] == za) & (z[j] == zb)
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
        if not (dis["long_std"] < long_std_ceiling):
            disorder_ok = False
            failures.append(
                f"{species}-{species} g(r) beyond {r_long:g} A has std "
                f"{dis['long_std']:.3f} (> {long_std_ceiling}); long-range order "
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
        chemistry_ok=chemistry_ok,
        disorder_ok=disorder_ok,
        range_ok=range_ok,
        failures=failures,
        warnings=warnings,
    )
