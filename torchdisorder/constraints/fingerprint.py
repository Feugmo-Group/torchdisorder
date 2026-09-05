"""Atom-order fingerprints, so a constraints file can only be paired with the
structure it was generated from.

Constraints are keyed by atom *index*, which makes them meaningless against any
differently-ordered copy of the same structure. That happens routinely: the
generators build indices with pymatgen, whose ``CifParser`` expands symmetry
operations and groups sites by unique label, while the trainer loads with ASE,
which reads the literal ``atom_site`` order. Same file, same atom count,
different indexing — 510 of 1125 sites disagree for ``c-SiO2.cif``.

The element guard in ``scripts/train.py`` catches the loud version of this, where
Si constraints land on oxygen. It cannot catch a permutation *within* one
element, which keeps every constrained index on a Si atom while still applying
each constraint to the wrong Si. This module closes that gap.

A fingerprint carries three things, checked in order of decreasing cheapness:

``n_atoms``
    Catches a structure of the wrong size, including the two readers disagreeing
    about how many atoms a file contains.
``symbols_sha256``
    Hash of the full symbol sequence. Catches any reorder that moves atoms
    across elements, and any change of composition.
``spot_checks``
    Up to ``N_SPOT_CHECKS`` recorded ``(index, symbol, frac_coords)`` triples
    sampled deterministically from the constrained indices, compared with a
    tolerance. This is the part that catches a same-element permutation.

Coordinates are compared rather than hashed because a hash of floats fails on
CIF round-trip noise, which is real (~1e-6) and has nothing to do with ordering.
A fractional tolerance is used so the check survives the density rescaling the
trainer applies to the cell.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "N_SPOT_CHECKS",
    "SPOT_CHECK_TOL",
    "atom_order_fingerprint",
    "verify_atom_order",
]

#: How many sites to record for the coordinate check. A pymatgen/ASE reorder
#: relocates most sites, so a few dozen samples make a silent pass vanishingly
#: unlikely while keeping the constraints file small.
N_SPOT_CHECKS = 64

#: Tolerance for the coordinate comparison, in fractional units so it is
#: independent of any rescaling of the cell. CIF round-trip noise is ~1e-6; a
#: genuine permutation moves a site by an appreciable fraction of the box.
SPOT_CHECK_TOL = 0.02


def _symbols_and_frac(structure: Any) -> tuple[list[str], np.ndarray]:
    """Read symbols and fractional coordinates from a pymatgen or ASE object."""
    if hasattr(structure, "get_scaled_positions"):  # ase.Atoms
        return (
            list(structure.get_chemical_symbols()),
            np.asarray(structure.get_scaled_positions(wrap=True), dtype=float),
        )
    if hasattr(structure, "frac_coords"):  # pymatgen Structure
        return (
            [site.specie.symbol for site in structure],
            np.asarray(structure.frac_coords, dtype=float) % 1.0,
        )
    raise TypeError(
        f"expected an ase.Atoms or pymatgen Structure, got {type(structure).__name__}"
    )


def _sample(indices: list[int], n: int) -> list[int]:
    """Pick ``n`` indices spread evenly across ``indices``, deterministically."""
    if len(indices) <= n:
        return list(indices)
    step = len(indices) / n
    return [indices[int(k * step)] for k in range(n)]


def atom_order_fingerprint(
    structure: Any,
    constrained_indices: Iterable[int] | None = None,
) -> dict:
    """Summarise the atom ordering of ``structure`` for later verification.

    Args:
        structure: the object the constraint indices refer to — an ``ase.Atoms``
            or a pymatgen ``Structure``.
        constrained_indices: indices the constraints file actually keys on. Spot
            checks are drawn from these, so the check concentrates on the atoms
            whose identity matters. Defaults to every atom.

    Returns:
        A JSON-serialisable dict to store under ``metadata["atom_order"]``.
    """
    symbols, frac = _symbols_and_frac(structure)

    if constrained_indices is None:
        pool = list(range(len(symbols)))
    else:
        pool = sorted({int(i) for i in constrained_indices})
        pool = [i for i in pool if 0 <= i < len(symbols)]
    if not pool:
        pool = list(range(len(symbols)))

    digest = hashlib.sha256(",".join(symbols).encode()).hexdigest()

    return {
        "n_atoms": len(symbols),
        "symbols_sha256": digest,
        "tolerance": SPOT_CHECK_TOL,
        "spot_checks": [
            [int(i), symbols[i], *(round(float(x), 6) for x in frac[i])]
            for i in _sample(pool, N_SPOT_CHECKS)
        ],
    }


def verify_atom_order(fingerprint: dict, structure: Any) -> list[str]:
    """Check ``structure`` against a fingerprint.

    Returns:
        A list of human-readable problems, empty when the ordering matches. The
        caller decides whether to warn or abort; nothing here raises on a
        mismatch, since a missing or malformed fingerprint is a different
        situation from a wrong one.
    """
    if not fingerprint:
        return []

    problems: list[str] = []
    symbols, frac = _symbols_and_frac(structure)

    expected_n = fingerprint.get("n_atoms")
    if expected_n is not None and expected_n != len(symbols):
        problems.append(
            f"atom count differs: constraints were generated against "
            f"{expected_n} atoms, the loaded structure has {len(symbols)}"
        )
        return problems  # every index is meaningless; further checks add noise

    expected_digest = fingerprint.get("symbols_sha256")
    if expected_digest:
        digest = hashlib.sha256(",".join(symbols).encode()).hexdigest()
        if digest != expected_digest:
            problems.append(
                "the element sequence differs, so atoms have been reordered "
                "across elements (or the composition changed)"
            )

    tol = float(fingerprint.get("tolerance", SPOT_CHECK_TOL))
    moved = []
    for entry in fingerprint.get("spot_checks", []):
        idx, symbol = int(entry[0]), str(entry[1])
        recorded = np.asarray(entry[2:5], dtype=float)
        if idx >= len(symbols):
            moved.append((idx, symbol, symbols[-1] if symbols else "?", None))
            continue
        delta = frac[idx] - recorded
        delta -= np.round(delta)  # nearest periodic image
        if symbols[idx] != symbol or np.abs(delta).max() > tol:
            moved.append((idx, symbol, symbols[idx], float(np.abs(delta).max())))

    if moved:
        n_checked = len(fingerprint.get("spot_checks", []))
        idx, want, got, dev = moved[0]
        detail = (f"holds {got}" if got != want
                  else f"is {dev:.3f} away in fractional units")
        problems.append(
            f"{len(moved)} of {n_checked} sampled sites are not where the "
            f"constraints expect them — e.g. index {idx} should be {want} "
            f"but {detail}. The atoms have been reordered, so each constraint "
            f"would land on the wrong atom."
        )

    return problems
