"""Measure what a *known-good* glass scores when judged at YOUR cell size.

Why this exists
---------------
The disorder ratio is not transferable between cell sizes. Noise falls as
1/sqrt(N_a) while a glass holds roughly constant structural signal, so the same
material scores a higher ratio the bigger the cell -- the published a-Li3PS4
below reads 4.15x at 1111 P and 1.36x at ~100 P. Comparing a 96-P candidate
against a threshold calibrated on a 1728-atom oxide, or against a published
reference measured in its own much larger cell, compares nothing.

So before trusting a disorder threshold on a NEW system, run this: take a
published amorphous model of that chemistry, carve boxes the size of the cells
you actually generate, and measure those. It answers "what does a real glass of
this material score, measured the way I measure mine" -- which is the only
version of the question that can validate or invalidate a run.

This was written after ten Li3PS4 melt-quench runs were rejected at 2.49-2.69x
against a 2.0x gate, and the gate was suspected of being the problem. It was
not: a genuine a-Li3PS4 lands at 1.36 +/- 0.14x at the same cell size, so the
rejections were real. Had it come out the other way, a week of runs would have
been reinstated instead.

The seam, and which way it biases
---------------------------------
A box cut from a periodic cell and then declared periodic has a seam: atoms near
opposite faces become artificial neighbours. This does NOT simply inflate the
measured std. It scrambles genuine long-range correlation, so the structural
signal drops -- for the a-Li3PS4 reference, from 0.344 in the full cell to 0.256
in the carved boxes. The carved ratio therefore *understates* what a real glass
of that size would score.

Bracket it rather than quoting one number. Report the carved value as the lower
bound, and as an upper bound recombine the FULL cell's structural signal with
the carved noise floor:

    upper = sqrt(signal_full^2 + floor_carved^2) / floor_carved

For a-Li3PS4 that gives 1.36x (carved) to 1.62x (upper). Both sit under 2.0x, so
the conclusion survives the ambiguity. If a bracket straddles your threshold,
the honest answer is that the threshold cannot decide that structure.

Usage
-----
    python scripts/calibrate_disorder_gate.py \
        --reference data/crystal-structures/ref_aLi3PS4_pccp.cif \
        --like data/crystal-structures/lips75_compact_from_crystal.cif \
        --species P
"""

from __future__ import annotations

import argparse

import numpy as np

RMAX_DEFAULT = 10.0


def carve(parent, origin, box):
    """A ``box``-sized periodic cell cut from ``parent`` starting at ``origin``."""
    cell = np.diag(parent.get_cell().array).copy()
    shifted = (parent.get_positions() - origin) % cell
    inside = np.all(shifted < box, axis=1)
    sub = parent[inside]
    sub.set_positions(shifted[inside])
    sub.set_cell(box)
    sub.set_pbc(True)
    return sub


def main() -> None:
    from ase.io import read

    from torchdisorder.common.glass_quality import noise_floor, sublattice_disorder

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True,
                   help="published amorphous model of the target chemistry")
    p.add_argument("--like", required=True,
                   help="a structure whose CELL matches the runs you want to judge")
    p.add_argument("--species", required=True, help="central atom of the sublattice")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--rmax", type=float, default=RMAX_DEFAULT)
    p.add_argument("--r-long", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    parent = read(args.reference)
    box = np.diag(read(args.like).get_cell().array).copy()
    kw = {"rmax": args.rmax, "r_long": args.r_long}

    def measure(atoms, label):
        dis = sublattice_disorder(atoms, args.species, **kw)
        floor = noise_floor(atoms, args.species, **kw)
        n = int((np.array(atoms.get_chemical_symbols()) == args.species).sum())
        print(f"  {label:14s} N={len(atoms):5d} {args.species}={n:4d} "
              f"std={dis['long_std']:.3f} floor={floor:.3f} "
              f"ratio={dis['long_std'] / floor:5.2f}x")
        return dis["long_std"], floor

    print(f"=== {args.reference} carved to {box.round(1)} A ===")
    std_full, floor_full = measure(parent, "full cell")
    # Structure and noise add in quadrature, so the genuine signal is what is
    # left after the floor is removed -- not the raw std.
    signal_full = float(np.sqrt(max(std_full**2 - floor_full**2, 0.0)))

    rng = np.random.default_rng(args.seed)
    cell = np.diag(parent.get_cell().array)
    rows = [measure(carve(parent, rng.random(3) * cell, box), f"sub-box {k + 1}")
            for k in range(args.samples)]

    stds = np.array([r[0] for r in rows])
    floors = np.array([r[1] for r in rows])
    ratios = stds / floors
    upper = float(np.sqrt(signal_full**2 + floors.mean() ** 2) / floors.mean())

    print(f"\n  carved ratio  {ratios.mean():.2f} +/- {ratios.std():.2f}x "
          f"(range {ratios.min():.2f}-{ratios.max():.2f})  <- lower bound")
    print(f"  upper bound   {upper:.2f}x  (full-cell signal {signal_full:.3f} "
          f"on the carved floor {floors.mean():.3f})")
    print(f"\n  A real glass of this chemistry, at this cell size, scores "
          f"{ratios.mean():.2f}-{upper:.2f}x.")
    print("  Set the gate above that band, and judge candidates against it.")


if __name__ == "__main__":
    main()
