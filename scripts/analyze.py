#!/usr/bin/env python
"""
Post-training structure analysis script.

Usage
-----
python scripts/analyze.py \
    --run_dir outputs/NaTaCl6_2026-07-16/11-44-06 \
    --system NaTaCl6 \
    --central Ta \
    --neighbor Cl \
    --cutoff 2.6 \
    --central_z 73 \
    --neighbor_z 17

All output is written to <run_dir>/analysis/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-training structure analysis for TorchDisorder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=Path,
        default=Path("."),
        help="Path to the training output directory (contains final_results/).",
    )
    p.add_argument("--system", type=str, default="Unknown", help="System name (for titles).")
    p.add_argument("--central", type=str, required=True, help="Central element symbol, e.g. Ta.")
    p.add_argument("--neighbor", type=str, default=None, help="Neighbor element symbol, e.g. Cl (None = all species).")
    p.add_argument(
        "--cutoff", type=float, required=True,
        help="Neighbor cutoff distance (Å) for bonds/CN/OPs."
    )
    p.add_argument(
        "--central_z", type=int, required=True,
        help="Atomic number of the central element (e.g. 73 for Ta)."
    )
    p.add_argument(
        "--neighbor_z", type=int, default=None,
        help="Atomic number of the neighbor element (None = all species)."
    )
    p.add_argument(
        "--structure_file", type=Path, default=None,
        help="Override structure file path (default: <run_dir>/final_results/final_structure.cif)."
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for order-parameter calculations."
    )
    p.add_argument(
        "--no_3d", action="store_true",
        help="Skip 3-D PyVista renders (useful if pyvista is not installed)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    # -----------------------------------------------------------------------
    # Locate structure file
    # -----------------------------------------------------------------------
    if args.structure_file is not None:
        structure_file = Path(args.structure_file)
    else:
        cif = run_dir / "final_results" / "final_structure.cif"
        xyz = run_dir / "final_results" / "final_structure.xyz"
        if cif.exists():
            structure_file = cif
        elif xyz.exists():
            structure_file = xyz
        else:
            # Try the run dir itself
            for candidate in [run_dir / "final_structure.cif", run_dir / "final_structure.xyz"]:
                if candidate.exists():
                    structure_file = candidate
                    break
            else:
                print(
                    f"ERROR: No structure file found in {run_dir}/final_results/.\n"
                    "  Expected final_structure.cif or final_structure.xyz.\n"
                    "  Use --structure_file to specify the path explicitly.",
                    file=sys.stderr,
                )
                sys.exit(1)

    print(f"Loading structure from: {structure_file}")

    # -----------------------------------------------------------------------
    # Import analysis module
    # -----------------------------------------------------------------------
    from torchdisorder.analysis import (
        load_structure,
        plot_all,
        viz3d_all,
        StructureDescriptors,
    )
    from torchdisorder.analysis.plots import (
        plot_convergence,
        plot_steinhardt_distributions,
        plot_disorder_heatmap,
    )

    atoms = load_structure(structure_file)
    print(f"Loaded {len(atoms)} atoms — species: {set(atoms.get_chemical_symbols())}")

    # -----------------------------------------------------------------------
    # Descriptors
    # -----------------------------------------------------------------------
    desc = StructureDescriptors(atoms, cutoff=args.cutoff, device=args.device)

    print("Computing order parameters …")
    op_data = desc.order_params_per_atom(
        central_z=args.central_z,
        neighbor_z=args.neighbor_z,
        cutoff=args.cutoff,
        compute=["q4", "q6", "tet", "fis"],
    )

    if args.neighbor:
        cn = desc.coordination_numbers(args.central, args.neighbor, args.cutoff)
        lengths = desc.bond_length_distribution((args.central, args.neighbor), args.cutoff)
        angles = desc.bond_angle_distribution(args.central, args.neighbor, args.cutoff)
    else:
        cn = np.array([])
        lengths = np.array([])
        angles = np.array([])

    # Full Steinhardt parameters
    print("Computing full Steinhardt parameters ...")
    neighbor_elements = [args.neighbor] if args.neighbor else None
    q4_full = desc.steinhardt_ql(4, args.cutoff, neighbor_elements=neighbor_elements)
    q6_full = desc.steinhardt_ql(6, args.cutoff, neighbor_elements=neighbor_elements)
    w4 = desc.steinhardt_wl(4, args.cutoff, neighbor_elements=neighbor_elements)
    w6 = desc.steinhardt_wl(6, args.cutoff, neighbor_elements=neighbor_elements)

    print("Computing ring statistics …")
    ring_stats = desc.ring_statistics(args.central, args.neighbor or args.central, args.cutoff)

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Structure Analysis Summary — {args.system}")
    print(f"{'=' * 60}")
    print(f"  Structure file : {structure_file}")
    print(f"  N atoms        : {len(atoms)}")
    print(f"  Cell volume    : {atoms.get_volume():.2f} Å³")
    print()
    print(f"  {'Order Parameter':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
    print(f"  {'-' * 54}")
    for key in ("q4", "q6", "tet", "fis"):
        vals = op_data.get(key, np.array([]))
        if len(vals) > 0:
            print(
                f"  {key:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}"
                f"  {vals.min():>9.4f}  {vals.max():>9.4f}"
            )

    # Steinhardt full parameters (all atoms)
    for label, vals in (
        ("q4_full (Steinhardt)", q4_full),
        ("q6_full (Steinhardt)", q6_full),
        ("w4", w4),
        ("w6", w6),
    ):
        if len(vals) > 0:
            print(
                f"  {label:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}"
                f"  {vals.min():>9.4f}  {vals.max():>9.4f}"
            )

    if len(cn) > 0:
        print()
        print("  Coordination Number Distribution:")
        unique, counts = np.unique(cn, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    CN={int(u):2d}  →  {c:4d} atoms  ({100 * c / len(cn):.1f}%)")

    nbr_label = args.neighbor or "all"
    if len(lengths) > 0:
        print()
        print(
            f"  {args.central}-{nbr_label} bond length:  "
            f"{lengths.mean():.4f} ± {lengths.std():.4f} Å  "
            f"[{lengths.min():.3f}, {lengths.max():.3f}]"
        )

    if len(angles) > 0:
        print(
            f"  {nbr_label}-{args.central}-{nbr_label} angle:  "
            f"{angles.mean():.2f} ± {angles.std():.2f}°"
        )

    if any(v > 0 for v in ring_stats.values()):
        print()
        print("  Ring Statistics:")
        for size, count in sorted(ring_stats.items()):
            if count > 0:
                print(f"    n={size:2d}-membered rings: {count}")

    print(f"{'=' * 60}\n")

    # -----------------------------------------------------------------------
    # Matplotlib plots
    # -----------------------------------------------------------------------
    print("Generating matplotlib plots …")
    plot_all(
        atoms,
        run_dir=run_dir,
        system_name=args.system,
        central=args.central,
        neighbor=args.neighbor,
        cutoff=args.cutoff,
        central_z=args.central_z,
        neighbor_z=args.neighbor_z,
        device=args.device,
    )

    # Steinhardt distribution and disorder heatmap plots
    plot_dir = run_dir / "analysis" / "plots"
    plot_steinhardt_distributions(
        q4_full, q6_full, w4=w4, w6=w6,
        system_name=args.system,
        save_path=plot_dir / "steinhardt_distributions.png",
    )
    print("  Steinhardt distributions plot saved.")
    plot_disorder_heatmap(
        q4_full, q6_full,
        system_name=args.system,
        save_path=plot_dir / "disorder_heatmap.png",
    )
    print("  Disorder heatmap plot saved.")

    # Convergence from log
    log_file = run_dir / "train.log"
    if log_file.exists():
        from torchdisorder.analysis import plots as _plots
        _plots.plot_convergence(
            log_file,
            save_path=run_dir / "analysis" / "plots" / "convergence.png",
        )
        print(f"  Convergence plot saved.")

    print(f"  Plots saved to: {plot_dir}")
    for f in sorted(plot_dir.glob("*.png")):
        print(f"    {f.name}")

    # -----------------------------------------------------------------------
    # 3-D renders
    # -----------------------------------------------------------------------
    if not args.no_3d:
        print("\nGenerating 3-D renders …")
        try:
            viz3d_all(
                atoms,
                run_dir=run_dir,
                system_name=args.system,
                central=args.central,
                neighbor=args.neighbor,
                cutoff=args.cutoff,
                central_z=args.central_z,
                neighbor_z=args.neighbor_z,
                device=args.device,
            )
            out_3d = run_dir / "analysis" / "3d"
            print(f"  3-D renders saved to: {out_3d}")
            for f in sorted(out_3d.glob("*.png")):
                print(f"    {f.name}")
        except Exception as exc:
            print(f"  [viz3d] skipped — {exc}")

        # Voronoi tessellation
        try:
            from torchdisorder.analysis.viz3d import viz_voronoi
            out_3d = run_dir / "analysis" / "3d"
            out_3d.mkdir(parents=True, exist_ok=True)
            vtk_path = out_3d / f"{args.system}_voronoi.vtk"
            for color_by in ("volume", "nfaces", "isoperimetric"):
                png_path = out_3d / f"{args.system}_voronoi_{color_by}.png"
                viz_voronoi(
                    atoms,
                    color_by=color_by,
                    save_path=png_path,
                    save_vtk=vtk_path if color_by == "volume" else None,
                )
                print(f"  Voronoi ({color_by}): {png_path.name}")
            print(f"  VTK exported: {vtk_path.name}")
        except Exception as exc:
            print(f"  [voronoi] skipped — {exc}")
    else:
        print("\n3-D renders skipped (--no_3d).")

    print("\nDone.")


if __name__ == "__main__":
    main()
