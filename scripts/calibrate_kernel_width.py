"""Fit ``kernel_width`` against a reference structure before refining.

Why
---
``kernel_width`` is the real-space broadening that turns a static snapshot into
something comparable with a time-averaged diffraction measurement.  It is not a
free knob: set too small, the calculated F(Q) keeps high-Q structure the
experiment does not show, and the amplitude comes out several times too large.

Measured on SiO2 with the default 0.035 A, forwarding the *published* GAP glass --
a structure independently validated as correct -- gives a calculated F(Q) whose
amplitude is 6.8x the observed one, with the discrepancy growing from 0.6x at low
Q to 15x at high Q.  chi^2 is then dominated by that mismatch rather than by
structural error, so minimising it necessarily distorts the structure.  Every
refinement in the audit did exactly that.

This script sweeps kernel_width against a structure you already trust and reports
the value that best reproduces the measured F(Q).  Judge by **shape correlation**,
not by chi^2 or RMS alone: those keep improving as the kernel is widened, because
over-smoothing buys amplitude agreement by erasing structure.  The correlation
turns over at the point where the broadening is physically right.

Usage
-----
    poetry run python scripts/calibrate_kernel_width.py \
        --structure data/json/sio2_glass_gap.cif \
        --data data/xrd_measurements/SiO2/F_of_Q.csv --system SiO2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

SCATTERING_LENGTHS = {
    "SiO2": {"Si": 4.1491, "O": 5.803},
    "GeO2": {"Ge": 8.185, "O": 5.803},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True, help="a structure you trust")
    p.add_argument("--data", required=True, help="experimental F(Q) csv")
    p.add_argument("--system", default="SiO2", choices=sorted(SCATTERING_LENGTHS))
    p.add_argument("--q-min", type=float, default=0.5)
    p.add_argument("--q-max", type=float, default=25.0)
    p.add_argument("--widths", type=float, nargs="+",
                   default=[0.035, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40])
    args = p.parse_args()

    import pandas as pd
    from ase.io import read
    from torch_sim.io import atoms_to_state

    from torchdisorder.model.xrd import XRDModel

    atoms = read(args.structure)
    state = atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float32)

    exp = pd.read_csv(args.data)
    exp = exp[(exp.Q >= args.q_min) & (exp.Q <= args.q_max)]
    q = torch.tensor(exp.Q.values, dtype=torch.float32)
    f_obs = torch.tensor(exp.F.values, dtype=torch.float32)
    r = torch.linspace(0.01, 10.0, 1000)

    print(f"structure : {args.structure}  ({len(atoms)} atoms)")
    print(f"data      : {args.data}  ({len(exp)} points, "
          f"Q = {args.q_min}-{args.q_max} A^-1)\n")
    print(f"{'kernel_width':>13} {'RMS':>9} {'best scale':>11} {'shape corr':>11}")

    rows = []
    for kw in args.widths:
        cfg = {
            "kernel_width": kw,
            "neutron_scattering_lengths": SCATTERING_LENGTHS[args.system],
            "xray_form_factor_params": {},
            "scattering_type": "neutron",
            "chunk_size": 20000,
        }
        model = XRDModel(symbols=sorted(set(atoms.get_chemical_symbols())), config=cfg,
                         r_bins=r, q_bins=q, rdf_data=None, device="cpu")
        with torch.no_grad():
            f_calc = model(state)["F_Q"].reshape(-1)

        scale = float((f_calc * f_obs).sum() / (f_calc * f_calc).sum())
        res = f_calc - f_obs
        a = (f_calc - f_calc.mean()) / f_calc.std()
        b = (f_obs - f_obs.mean()) / f_obs.std()
        corr = float((a * b).mean())
        rows.append((kw, float(res.pow(2).mean().sqrt()), scale, corr))
        print(f"{kw:13.3f} {rows[-1][1]:9.5f} {scale:11.4f} {corr:11.4f}")

    best = max(rows, key=lambda t: t[3])
    print(f"\nBest shape correlation at kernel_width = {best[0]:.3f}  (r = {best[3]:.4f}, "
          f"amplitude scale {best[2]:.3f})")
    print("\nSet it in the data config, e.g.:")
    print(f"    kernel_width: {best[0]}")
    print("\nA scale far from 1.0 at the correlation optimum means broadening alone does")
    print("not reconcile the two -- suspect the F(Q) convention or normalisation of the")
    print("experimental file before trusting any refinement against it.")


if __name__ == "__main__":
    main()
