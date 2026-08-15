"""
Generate a 3-panel comparison figure: F_IS vs q4 vs q6 distributions
for the TorchDisorder-optimized a-SiO₂ glass, with c-SiO₂ crystal
reference values marked as vertical lines.

Usage:
    python scripts/plot_fis_vs_boo.py \
        --glass outputs/SiO2_2026-07-16/12-17-58/final_results/final_structure.cif \
        --crystal data/crystal-structures/c-SiO2.cif \
        --out Tutorials/plots/fis_vs_boo_comparison.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch_sim as ts
import ase
import ase.data
import ase.io


def atoms_to_simstate(atoms, device="cpu"):
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device).unsqueeze(0)
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.int32, device=device)
    masses = torch.tensor(
        [ase.data.atomic_masses[z] for z in atoms.numbers],
        dtype=torch.float32, device=device,
    )
    pbc = torch.ones(1, 3, dtype=torch.bool, device=device)
    system_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)
    return ts.SimState(
        positions=positions, masses=masses, cell=cell,
        atomic_numbers=atomic_numbers, pbc=pbc, system_idx=system_idx,
    )


def compute_descriptors(atoms, cutoff=2.2, central_z=14, neighbor_z=8, device="cpu"):
    from torchdisorder.engine.order_params import TorchSimOrderParameters

    state = atoms_to_simstate(atoms, device=device)

    # Indices of central atoms
    numbers = np.array(atoms.numbers)
    central_mask = numbers == central_z
    atom_indices = torch.tensor(np.where(central_mask)[0], dtype=torch.long, device=device)

    op = TorchSimOrderParameters(cutoff=cutoff, device=device, fis_mode="variable_R")
    results = op(
        state,
        atom_indices,
        ["fis", "q4", "q6"],
        element_filter=[neighbor_z],
    )

    fis = results["fis"].cpu().numpy()
    q4  = results["q4"].cpu().numpy()
    q6  = results["q6"].cpu().numpy()
    return fis, q4, q6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glass",   type=Path,
                   default=Path("outputs/SiO2_mace_debug_2026-07-16/13-55-58/final_results/final_structure.cif"))
    p.add_argument("--crystal", type=Path,
                   default=Path("data/crystal-structures/c-SiO2.cif"))
    p.add_argument("--out",     type=Path,
                   default=Path("Tutorials/plots/fis_vs_boo_comparison.png"))
    p.add_argument("--cutoff",  type=float, default=2.2)
    p.add_argument("--central_z", type=int, default=14)
    p.add_argument("--neighbor_z", type=int, default=8)
    p.add_argument("--device",  type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading glass:   {args.glass}")
    glass = ase.io.read(str(args.glass))
    print(f"Loading crystal: {args.crystal}")
    crystal = ase.io.read(str(args.crystal))

    print(f"Glass:   {len(glass)} atoms")
    print(f"Crystal: {len(crystal)} atoms")

    print("Computing descriptors for glass …")
    fis_g, q4_g, q6_g = compute_descriptors(
        glass, args.cutoff, args.central_z, args.neighbor_z, args.device)

    print("Computing descriptors for crystal …")
    fis_c, q4_c, q6_c = compute_descriptors(
        crystal, args.cutoff, args.central_z, args.neighbor_z, args.device)

    crystal_vals = {
        "F_IS": float(np.mean(fis_c)),
        "q4":   float(np.mean(q4_c)),
        "q6":   float(np.mean(q6_c)),
    }
    print(f"Crystal reference — F_IS: {crystal_vals['F_IS']:.3f}  "
          f"q4: {crystal_vals['q4']:.3f}  q6: {crystal_vals['q6']:.3f}")
    print(f"Glass mean       — F_IS: {np.mean(fis_g):.3f}  "
          f"q4: {np.mean(q4_g):.3f}  q6: {np.mean(q6_g):.3f}")
    print(f"Glass std        — F_IS: {np.std(fis_g):.3f}  "
          f"q4: {np.std(q4_g):.3f}  q6: {np.std(q6_g):.3f}")

    # -----------------------------------------------------------------------
    # Figure: 3-row layout, stacked panels
    # -----------------------------------------------------------------------
    GLASS_COLOR  = "#4C72B0"
    XTAL_COLOR   = "#C44E52"
    ALPHA_HIST   = 0.75
    N_BINS       = 40

    fig = plt.figure(figsize=(8, 9))
    fig.suptitle(
        "F$_{IS}$ vs Bond-Orientational Order — a-SiO₂\n"
        "(Si atoms, first coordination shell, cutoff = 2.2 Å)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.90, bottom=0.08,
                           left=0.12, right=0.95)

    descriptors = [
        ("F$_{IS}$",  fis_g, crystal_vals["F_IS"], (-1.1, 1.1),
         "Local inversion symmetry\n"
         "measures centrosymmetry of each Si coordination shell"),
        ("$q_4$  (BOO)",  q4_g, crystal_vals["q4"],  (0, 0.75),
         "Bond-orientational order  $l=4$\n"
         "measures angular arrangement of neighbors"),
        ("$q_6$  (BOO)",  q6_g, crystal_vals["q6"],  (0, 0.75),
         "Bond-orientational order  $l=6$\n"
         "measures angular arrangement of neighbors"),
    ]

    for row, (label, glass_vals, xtal_ref, xlim, subtitle) in enumerate(descriptors):
        ax = fig.add_subplot(gs[row])

        # histogram
        bins = np.linspace(xlim[0], xlim[1], N_BINS + 1)
        ax.hist(glass_vals, bins=bins, density=True,
                color=GLASS_COLOR, alpha=ALPHA_HIST, label="a-SiO₂ (glass)")

        # crystal reference line
        ax.axvline(xtal_ref, color=XTAL_COLOR, lw=2.0, ls="--",
                   label=f"c-SiO₂ crystal  ({xtal_ref:.3f})")

        # glass mean ± std band
        mu, sigma = np.mean(glass_vals), np.std(glass_vals)
        ax.axvspan(mu - sigma, mu + sigma,
                   alpha=0.12, color=GLASS_COLOR, zorder=0)
        ax.axvline(mu, color=GLASS_COLOR, lw=1.5, ls="-", alpha=0.8,
                   label=f"glass mean  ({mu:.3f} ± {sigma:.3f})")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(subtitle, fontsize=9, color="#444444", pad=3)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)

        # annotation: Δ = |glass mean − crystal|
        delta = abs(mu - xtal_ref)
        ax.text(0.02, 0.92,
                f"Δ = {delta:.3f}   |σ = {sigma:.3f}",
                transform=ax.transAxes, fontsize=8.5,
                color="#222222", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=160, bbox_inches="tight")
    print(f"\nSaved: {args.out}")

    # Also save PDF for publication
    pdf_path = args.out.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
