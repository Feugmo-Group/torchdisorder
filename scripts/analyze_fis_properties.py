#!/usr/bin/env python
"""
Compute F_IS-derived structural properties for an optimized structure:
  1. F_IS by CN (violin plot with theoretical references)
  2. Polyhedral distortion index (histogram per CN group)
  3. Warren-Cowley SRO matrix
  4. F_IS spatial autocorrelation C(r)

Usage
-----
python scripts/analyze_fis_properties.py \
    --run_dir outputs/SiO2_2026-07-16/12-17-58 \
    --system Fe2O3 \
    --central Fe --neighbor O --cutoff 2.2 \
    --central_z 26 --neighbor_z 8

Output: <run_dir>/analysis/plots/fis_by_cn.png  etc.
"""

import argparse
from pathlib import Path
import sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--system", type=str, default="Unknown")
    p.add_argument("--central", type=str, required=True)
    p.add_argument("--neighbor", type=str, default=None)
    p.add_argument("--cutoff", type=float, required=True)
    p.add_argument("--central_z", type=int, required=True)
    p.add_argument("--neighbor_z", type=int, default=None)
    p.add_argument("--structure_file", type=Path, default=None)
    p.add_argument("--sro_cutoff", type=float, default=None,
                   help="Cutoff for Warren-Cowley SRO (default = same as --cutoff).")
    p.add_argument("--fis_r_max", type=float, default=8.0,
                   help="Max distance for F_IS autocorrelation (Å).")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def find_structure(run_dir: Path, structure_file: Path | None) -> Path:
    if structure_file is not None:
        return structure_file
    for candidate in [
        run_dir / "final_results" / "final_structure.cif",
        run_dir / "final_results" / "final_structure.xyz",
        run_dir / "final_structure.cif",
    ]:
        if candidate.exists():
            return candidate
    print("ERROR: No structure file found.", file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()
    run_dir = args.run_dir
    sro_cutoff = args.sro_cutoff or args.cutoff
    plot_dir = run_dir / "analysis" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    structure_file = find_structure(run_dir, args.structure_file)
    print(f"Structure: {structure_file}")

    from torchdisorder.analysis import load_structure, StructureDescriptors
    from torchdisorder.analysis.plots import (
        plot_fis_by_cn,
        plot_distortion_index,
        plot_warren_cowley,
        plot_fis_spatial_autocorrelation,
    )

    atoms = load_structure(structure_file)
    print(f"Loaded {len(atoms)} atoms — {set(atoms.get_chemical_symbols())}")
    desc = StructureDescriptors(atoms, cutoff=args.cutoff, device=args.device)

    # ------------------------------------------------------------------
    # 1. F_IS by CN
    # ------------------------------------------------------------------
    print("Computing F_IS by CN …")
    if args.neighbor:
        fis_cn = desc.fis_by_cn(
            central=args.central,
            neighbor=args.neighbor,
            cutoff=args.cutoff,
            central_z=args.central_z,
            neighbor_z=args.neighbor_z,
        )
        print("  CN distribution:")
        for cn_val, vals in sorted(fis_cn.items()):
            print(f"    CN={cn_val}: n={len(vals):4d}  "
                  f"F_IS = {vals.mean():.3f} ± {vals.std():.3f}")
        plot_fis_by_cn(fis_cn, system_name=args.system,
                       save_path=plot_dir / "fis_by_cn.png")
        print(f"  Saved fis_by_cn.png")
    else:
        print("  Skipped (no --neighbor)")

    # ------------------------------------------------------------------
    # 2. Polyhedral distortion index
    # ------------------------------------------------------------------
    if args.neighbor:
        print("Computing polyhedral distortion index …")
        di = desc.polyhedral_distortion_index(args.central, args.neighbor, args.cutoff)
        cn_for_di = desc.coordination_numbers(args.central, args.neighbor, args.cutoff)
        print(f"  DI: mean={di.mean():.4f}  std={di.std():.4f}  "
              f"max={di.max():.4f}")
        plot_distortion_index(di, cn=cn_for_di, system_name=args.system,
                              save_path=plot_dir / "distortion_index.png")
        print("  Saved distortion_index.png")
    else:
        print("  Distortion index skipped (no --neighbor)")

    # ------------------------------------------------------------------
    # 3. Warren-Cowley SRO
    # ------------------------------------------------------------------
    print(f"Computing Warren-Cowley SRO (cutoff={sro_cutoff:.2f} Å) …")
    alpha, elems = desc.warren_cowley_sro(cutoff=sro_cutoff)
    print(f"  Elements: {elems}")
    print("  α_ij matrix:")
    header = "       " + "  ".join(f"{e:>6}" for e in elems)
    print(header)
    for i, ei in enumerate(elems):
        row = f"  {ei:>4} " + "  ".join(f"{alpha[i,j]:>6.3f}" for j in range(len(elems)))
        print(row)
    plot_warren_cowley(alpha, elems, system_name=args.system,
                       save_path=plot_dir / "warren_cowley_sro.png")
    print("  Saved warren_cowley_sro.png")

    # ------------------------------------------------------------------
    # 4. F_IS spatial autocorrelation
    # ------------------------------------------------------------------
    if args.neighbor:
        print(f"Computing F_IS spatial autocorrelation (r_max={args.fis_r_max} Å) …")
        r_c, C_r = desc.fis_spatial_autocorrelation(
            central=args.central,
            neighbor=args.neighbor,
            cutoff=args.cutoff,
            central_z=args.central_z,
            neighbor_z=args.neighbor_z,
            r_max=args.fis_r_max,
        )
        # Find first zero crossing
        zc = np.where(np.diff(np.sign(C_r)))[0]
        if len(zc):
            print(f"  Correlation length (first zero): {r_c[zc[0]]:.2f} Å")
        else:
            print("  No zero crossing found within r_max")
        plot_fis_spatial_autocorrelation(r_c, C_r, system_name=args.system,
                                         save_path=plot_dir / "fis_autocorrelation.png")
        print("  Saved fis_autocorrelation.png")
    else:
        print("  F_IS autocorrelation skipped (no --neighbor)")

    print(f"\nAll plots saved to: {plot_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
