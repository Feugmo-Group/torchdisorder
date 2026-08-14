"""Compare order parameters between a refined structure and a reference model.

Structural distances (bond lengths, angles) say whether a refinement preserved the
network. Order parameters say whether it preserved the *symmetry* of the local
environments -- which is what F_IS and the BOO parameters are actually used for
downstream, so a refinement that drifts here invalidates any conclusion drawn
from them, even if the bond lengths look acceptable.

Reports, per structure and per central atom:
    cn   coordination number
    tet  tetrahedral order
    q4   bond-orientational order, l = 4
    q6   bond-orientational order, l = 6
    fis  local inversion symmetry (Milkus & Zaccone, PRB 93, 094204 (2016))

Usage:
    poetry run python scripts/compare_order_params.py \
        --reference data/json/sio2_glass_gap.cif \
        --test outputs/<run>/final_results/final_structure.cif \
        --central Si --neighbour O --cutoff 2.2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

OPS = ["cn", "tet", "q4", "q6", "fis"]


def measure(path, central_z, neighbour_z, cutoff, max_neighbors):
    from ase.io import read
    from torch_sim.io import atoms_to_state

    from torchdisorder.engine.order_params import PyTorchOrderParameters

    atoms = read(str(path))
    state = atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float64)
    idx = torch.where(state.atomic_numbers == central_z)[0]
    if idx.numel() == 0:
        raise SystemExit(f"no atoms of Z={central_z} in {path}")

    calc = PyTorchOrderParameters(cutoff=cutoff, device="cpu",
                                  max_neighbors=max_neighbors)
    out = calc(state, idx, OPS, element_filter=[neighbour_z])
    return {k: v.detach().cpu().numpy() for k, v in out.items()}, len(idx)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, help="published / trusted model")
    p.add_argument("--test", required=True, nargs="+", help="one or more refined structures")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--central", default="Si")
    p.add_argument("--neighbour", default="O")
    p.add_argument("--cutoff", type=float, default=2.2)
    p.add_argument("--max-neighbors", type=int, default=8)
    args = p.parse_args()

    from ase.data import atomic_numbers

    cz, nz = atomic_numbers[args.central], atomic_numbers[args.neighbour]
    labels = args.labels or [f"test{i}" for i in range(len(args.test))]
    if len(labels) != len(args.test):
        raise SystemExit("--labels must match the number of --test entries")

    ref, n_ref = measure(args.reference, cz, nz, args.cutoff, args.max_neighbors)

    print("=" * 86)
    print(f"Order parameters: {args.central}-centred, {args.neighbour} neighbours, "
          f"cutoff {args.cutoff} A")
    print("=" * 86)
    print(f"\nreference: {args.reference}  ({n_ref} {args.central} centres)")
    print(f"{'':14}" + "".join(f"{o:>16s}" for o in OPS))
    print(f"{'reference':14}" + "".join(
        f"{ref[o].mean():+8.4f}±{ref[o].std():6.4f}" for o in OPS))

    rows = []
    for lab, path in zip(labels, args.test):
        got, n = measure(path, cz, nz, args.cutoff, args.max_neighbors)
        rows.append((lab, got, n))
        print(f"{lab[:14]:14}" + "".join(
            f"{got[o].mean():+8.4f}±{got[o].std():6.4f}" for o in OPS))

    print(f"\nShift from reference (test - reference):")
    print(f"{'':14}" + "".join(f"{o:>16s}" for o in OPS))
    for lab, got, _ in rows:
        cells = ""
        for o in OPS:
            d = got[o].mean() - ref[o].mean()
            # Flag a shift larger than the reference's own spread: that is a change
            # in the environments themselves, not sampling noise.
            flag = "*" if abs(d) > max(ref[o].std(), 1e-9) else " "
            cells += f"{d:+14.4f}{flag} "
        print(f"{lab[:14]:14}{cells}")

    print("\n* = shift exceeds the reference's own standard deviation, i.e. the local")
    print("  environments have genuinely changed rather than merely been resampled.")
    print("\nF_IS reference points: -1/3 exactly for an ideal tetrahedron, +1 for a")
    print("centrosymmetric environment. It is dominated by the local coordination")
    print("geometry, so for a tetrahedral network it should barely move between a")
    print("crystal and its glass -- a large F_IS shift means the tetrahedra themselves")
    print("were deformed.")


if __name__ == "__main__":
    main()
