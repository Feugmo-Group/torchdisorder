"""Physical-plausibility checks for optimized structures.

Why this module exists
----------------------
Agreement with an experimental S(Q)/F(Q)/G(r) is a *necessary* but badly
insufficient condition.  Structure refinement against a one-dimensional
scattering function is underdetermined: a configuration with atoms sitting
0.2 A apart can reproduce the target curve essentially perfectly while being
physically meaningless.  An audit of 35 archived TorchDisorder runs found that
25 of them (71%, spanning SiO2, GeO2, LiPS, Fe2O3 and LiPON) contained atom
overlaps that no chi-squared or R-factor would ever reveal.

The checks here are deliberately composition-agnostic and cheap, so they can run
inside the optimization loop rather than as a forensic step afterwards.

Two independent failure modes are tested:

``overlap``
    Any pair of atoms closer than ``overlap_tol`` times the sum of their
    covalent radii.  This is the unambiguous failure — real matter does not do
    this — and it is what most of the archived structures violate.

``plateau``
    A physical first coordination shell produces a flat <CN> over a range of
    cutoffs, because there is a genuine gap between the first and second shells.
    A structure whose <CN> climbs monotonically with cutoff has no first shell
    at all, only a smear of distances.  This catches the subtler case where the
    network has dissolved without any single pair being egregiously close.

Both are reported as a :class:`ValidationReport`, which is falsey when the
structure fails, so callers can simply write ``if not report:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ValidationReport",
    "validate_structure",
    "coordination_profile",
    "plateau_window",
]

# Below this fraction of the summed covalent radii, a contact is not a bond.
DEFAULT_OVERLAP_TOL = 0.6

# Cutoffs (A) over which a genuine first shell should hold <CN> roughly constant.
# These bracket a Si-O bond; see plateau_window for other chemistries.
DEFAULT_PLATEAU_CUTOFFS = (1.8, 2.0, 2.2, 2.4, 2.6)


def plateau_window(bond_cutoff: float, n: int = 5, step: float = 0.2) -> tuple:
    """Plateau cutoffs bracketing ``bond_cutoff``.

    The default window is tuned to Si-O and is simply wrong elsewhere: a P-S
    bond sits at ~2.05 A, so <CN> is exactly zero at 1.8 and 2.0 A and pristine
    crystalline Li7P3S11 fails the flatness test for want of a first shell it
    plainly has. Deriving the window from the bond cutoff makes the check mean
    the same thing in every chemistry -- and reproduces the historical default
    exactly for the Si-O cutoff of 2.2 A.
    """
    half = (n - 1) // 2
    return tuple(round(bond_cutoff + (k - half) * step, 3) for k in range(n))


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_structure`.

    Falsey when the structure failed any enabled check, so ``if not report:``
    reads naturally at call sites.
    """

    n_atoms: int
    formula: str
    density: float
    min_distance: float
    n_overlaps: int
    worst_ratio: float
    """Smallest (distance / covalent floor) over all pairs; < 1 means overlap."""
    overlap_pairs: list = field(default_factory=list)
    """Up to 10 worst offenders as (symbol_i, symbol_j, distance, floor)."""
    plateau: dict | None = None
    """Mapping of cutoff -> mean CN, when a plateau check was requested."""
    plateau_ok: bool | None = None
    failures: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        head = (
            f"{self.formula} ({self.n_atoms} atoms, {self.density:.3f} g/cm3)  "
            f"min d = {self.min_distance:.3f} A"
        )
        if not self.failures:
            return f"PASS  {head}"
        lines = [f"FAIL  {head}"]
        for f in self.failures:
            lines.append(f"  - {f}")
        for si, sj, d, floor in self.overlap_pairs[:5]:
            lines.append(f"      {si}-{sj} at {d:.3f} A (expected > {floor:.3f} A)")
        if self.plateau:
            cuts = "".join(f"{c:8.1f}" for c in self.plateau)
            vals = "".join(f"{v:8.3f}" for v in self.plateau.values())
            lines.append(f"      cutoff (A): {cuts}")
            lines.append(f"      <CN>:       {vals}")
        return "\n".join(lines)


def _atoms_from(obj):
    """Accept an ASE Atoms, a torch_sim state, or a path to a structure file."""
    from ase import Atoms

    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, (str,)) or hasattr(obj, "__fspath__"):
        from ase.io import read

        return read(str(obj))
    # torch_sim SimState (possibly batched — validate the first system)
    from torch_sim.io import state_to_atoms

    converted = state_to_atoms(obj)
    return converted[0] if isinstance(converted, list) else converted


def coordination_profile(
    atoms,
    central: int | str | None = None,
    neighbour: int | str | None = None,
    cutoffs=DEFAULT_PLATEAU_CUTOFFS,
) -> dict:
    """Mean coordination number of `central` by `neighbour` at each cutoff.

    With no species given, counts every atom's neighbours of any species.
    """
    from ase.data import atomic_numbers
    from ase.neighborlist import neighbor_list

    def _z(spec):
        if spec is None:
            return None
        return atomic_numbers[spec] if isinstance(spec, str) else int(spec)

    z_c, z_n = _z(central), _z(neighbour)
    z = atoms.get_atomic_numbers()
    centres = np.ones(len(atoms), bool) if z_c is None else (z == z_c)
    if not centres.any():
        return {}

    profile = {}
    for cutoff in cutoffs:
        i, j, _ = neighbor_list("ijd", atoms, float(cutoff))
        keep = np.ones(len(i), bool)
        if z_c is not None:
            keep &= z[i] == z_c
        if z_n is not None:
            keep &= z[j] == z_n
        counts = np.bincount(i[keep], minlength=len(atoms))
        profile[float(cutoff)] = float(counts[centres].mean())
    return profile


def validate_structure(
    structure,
    *,
    overlap_tol: float = DEFAULT_OVERLAP_TOL,
    check_plateau: bool = False,
    central=None,
    neighbour=None,
    expected_cn: float | None = None,
    cn_tol: float = 0.15,
    plateau_cutoffs=None,
    bond_cutoff: float | None = None,
    plateau_spread: float = 0.25,
) -> ValidationReport:
    """Check a structure for atom overlap and, optionally, a first-shell plateau.

    Parameters
    ----------
    structure
        ASE ``Atoms``, a ``torch_sim`` state, or a path to a structure file.
    overlap_tol
        A contact closer than ``overlap_tol * (r_i + r_j)`` covalent radii counts
        as an overlap.  0.6 is permissive enough to allow genuinely short bonds
        while still catching the 0.2-0.9 A contacts seen in failed refinements.
    check_plateau
        Also require <CN> to be flat across ``plateau_cutoffs``.  Needs
        ``central``/``neighbour`` to be meaningful for multi-species systems.
    expected_cn
        If given, the plateau must additionally sit within ``cn_tol`` of it
        (e.g. 4.0 for SiO4 tetrahedra).
    plateau_cutoffs
        Explicit cutoff window for the plateau test.  When omitted it is derived
        from ``bond_cutoff``, falling back to the Si-O default.
    bond_cutoff
        First-shell bond cutoff for this chemistry (e.g. 2.2 for Si-O, 2.5 for
        P-S).  Used to place the plateau window; pass it for anything but an
        oxide, or the window will not contain the first shell at all.
    plateau_spread
        Maximum allowed max-min spread of <CN> across the cutoff window.

    Returns
    -------
    ValidationReport
        Falsey if any enabled check failed.
    """
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    if plateau_cutoffs is None:
        plateau_cutoffs = (
            plateau_window(bond_cutoff) if bond_cutoff is not None
            else DEFAULT_PLATEAU_CUTOFFS
        )

    atoms = _atoms_from(structure)
    z = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()

    radii = covalent_radii[z]
    search = 2.0 * float(radii.max()) + 0.5
    i, j, d = neighbor_list("ijd", atoms, search)

    failures: list[str] = []
    pairs: list = []
    if len(d) == 0:
        min_d, n_overlap, worst = float("nan"), 0, float("nan")
        failures.append(f"no neighbours within {search:.1f} A — structure may be exploded")
    else:
        floor = overlap_tol * (radii[i] + radii[j])
        bad = d < floor
        # neighbor_list lists each pair twice (i,j) and (j,i).
        n_overlap = int(bad.sum() // 2)
        min_d = float(d.min())
        worst = float((d / floor).min())
        if n_overlap:
            order = np.argsort(d[bad])[:20]
            seen = set()
            for k in order:
                a, b = int(i[bad][k]), int(j[bad][k])
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((symbols[a], symbols[b], float(d[bad][k]), float(floor[bad][k])))
                if len(pairs) >= 10:
                    break
            failures.append(
                f"{n_overlap} atom pair(s) closer than {overlap_tol:.0%} of covalent "
                f"contact; closest is {min_d:.3f} A"
            )

    profile, plateau_ok = None, None
    if check_plateau:
        profile = coordination_profile(atoms, central, neighbour, plateau_cutoffs)
        if not profile:
            plateau_ok = False
            failures.append(f"no atoms of species {central!r} to build a coordination profile")
        else:
            cuts = np.array(list(profile))
            values = np.array(list(profile.values()))

            # Find the LONGEST FLAT RUN, rather than requiring the whole window to
            # be flat.  The lowest cutoffs deliberately sit below the first-shell
            # peak, so <CN> is still climbing there even for a perfect network:
            # a physical a-SiO2 gives 3.29 at 1.8 A and 3.99/4.00/4.00/4.00 above
            # it.  Judging the raw max-min spread would reject that outright, and a
            # checker that fails good structures gets switched off.
            best_i, best_j = 0, 0
            for i in range(len(values)):
                j = i
                while j + 1 < len(values) and (
                    values[i : j + 2].max() - values[i : j + 2].min() <= plateau_spread
                ):
                    j += 1
                if j - i > best_j - best_i:
                    best_i, best_j = i, j
            run = values[best_i : best_j + 1]

            # A plateau needs at least three consecutive cutoffs to mean anything.
            plateau_ok = len(run) >= 3
            if not plateau_ok:
                failures.append(
                    f"<CN> never holds flat over 3 consecutive cutoffs in "
                    f"{plateau_cutoffs[0]}-{plateau_cutoffs[-1]} A "
                    f"(varies by {values.max() - values.min():.2f}) — no resolved first shell"
                )
            if expected_cn is not None:
                # Judge the plateau itself, not the pre-shell cutoffs.
                level = float(np.median(run))
                if abs(level - expected_cn) > cn_tol:
                    plateau_ok = False
                    failures.append(
                        f"plateau <CN> = {level:.2f} over "
                        f"{cuts[best_i]:.1f}-{cuts[best_j]:.1f} A, "
                        f"expected {expected_cn:.2f} +/- {cn_tol}"
                    )

    return ValidationReport(
        n_atoms=len(atoms),
        formula=atoms.get_chemical_formula(mode="hill", empirical=True),
        density=float(atoms.get_masses().sum() / atoms.get_volume() * 1.66054),
        min_distance=min_d,
        n_overlaps=n_overlap,
        worst_ratio=worst,
        overlap_pairs=pairs,
        plateau=profile,
        plateau_ok=plateau_ok,
        failures=failures,
    )
