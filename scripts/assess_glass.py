"""Ask whether structures on disk are actually glasses.

Why
---
The melt-quench builder applies this gate to its own output, but most of the
structures in data/crystal-structures predate it, and their filenames are not
evidence: geo2_glass.cif measures byte-identically to c-GeO2.cif. This is the
tool for auditing what is already there.

The system is inferred from the composition, so a directory can be swept in one
go. Two independent verdicts are printed per structure -- chemistry and disorder
-- because they have different remedies; see torchdisorder.common.glass_quality.

Usage
-----
    # audit a whole directory, inferring the system from each formula
    poetry run python scripts/assess_glass.py data/crystal-structures/*.cif

    # one structure, system stated explicitly
    poetry run python scripts/assess_glass.py --system LiPS mystery.cif

    # allow a species that has an absolute rule (unknown labels are warned about,
    # not silently ignored -- P2S6's P-P bonds no longer need one, see speciation)
    poetry run python scripts/assess_glass.py --tolerate "O2 molecules=1" geo2_mq.cif

Exit status is 1 if any structure was rejected, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def infer_system(atoms) -> str | None:
    """Guess the ruleset from the elements present.

    Deliberately conservative: an unrecognised composition returns None and is
    reported as such rather than being judged under the wrong chemistry rules,
    which would count contacts that are perfectly normal in that system.
    """
    elements = set(atoms.get_chemical_symbols())
    if {"Si", "O"} <= elements:
        return "SiO2"
    if {"Ge", "O"} <= elements:
        return "GeO2"
    if {"P", "S"} <= elements:
        return "LiPS"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--system", default=None,
                   help="force a ruleset instead of inferring it per structure")
    p.add_argument("--tolerate", action="append", default=[], metavar="LABEL=N",
                   help="allow N occurrences of a forbidden species")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print the full report for every structure, not just failures")
    args = p.parse_args()

    from ase.io import read

    from torchdisorder.common.glass_quality import assess_glass

    tolerated = {}
    for item in args.tolerate:
        label, _, n = item.partition("=")
        if not n.strip().isdigit():
            raise SystemExit(f"--tolerate wants LABEL=N, got {item!r}")
        tolerated[label.strip()] = int(n)

    # "x noise" is the column to read, not "std>6". The raw std is not comparable
    # between chemistries -- 96 P and 1728 Si have different counting-noise floors
    # -- so the same number means different things in different rows. Glasses sit
    # at 1.6-3.5x their floor, crystals at 14-32x. Reading std alone once passed a
    # partially melted GeO2 (0.346, under the 0.5 limit, but 4.04x its floor).
    # 40, not 32: rejected runs are kept as "<name>_REJECTED.cif", which is 9
    # characters longer than the name that was asked for and overflowed the old
    # column, shifting every remaining field on exactly the rows worth reading.
    print(f"{'structure':40s} {'system':6s} {'N':>5s} {'max g':>6s} {'std>6':>7s} "
          f"{'x noise':>8s} {'chem':>4s} {'dis':>4s}  verdict")
    print("-" * 105)

    # Counted separately: a structure nobody could judge is not a structure that
    # passed, and rolling the two together is how "15 passed" ends up describing a
    # set in which only three were actually shown to be glasses.
    passed = failed = skipped = 0
    for path in args.paths:
        try:
            atoms = read(str(path))
        except Exception as exc:  # noqa: BLE001 - a bad file is a result, not a crash
            print(f"{path.name:40s} {'-':6s} {'-':>5s}  unreadable: {type(exc).__name__}")
            failed += 1
            continue

        system = args.system or infer_system(atoms)
        if system is None:
            formula = atoms.get_chemical_formula(mode="hill", empirical=True)
            print(f"{path.name:40s} {'?':6s} {len(atoms):5d}  no ruleset for {formula} "
                  "-- pass --system to judge it")
            skipped += 1
            continue

        rep = assess_glass(atoms, system, tolerated=tolerated)
        verdict = "GLASS" if rep else "NOT A GLASS"
        # nan floor means it was not measured (check_noise off, or the cell was
        # too small to judge at all); print "-" rather than "nan", which reads as
        # a failed measurement of a number that was never asked for.
        excess = ("-" if np.isnan(rep.noise) or not rep.noise
                  else f"{rep.long_std / rep.noise:.2f}x")
        print(f"{path.name:40s} {system:6s} {len(atoms):5d} {rep.max_g:6.2f} "
              f"{rep.long_std:7.3f} {excess:>8s} "
              f"{'ok' if rep.chemistry_ok else 'FAIL':>4s} "
              f"{'ok' if rep.disorder_ok else 'FAIL':>4s}  {verdict}")
        if rep:
            passed += 1
        else:
            failed += 1
        if args.verbose or not rep:
            for line in rep.summary().splitlines()[1:]:
                print(f"      {line.strip()}")

    print("-" * 105)
    tail = f", {skipped} not judged (no ruleset)" if skipped else ""
    print(f"{passed} glass, {failed} rejected{tail}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
