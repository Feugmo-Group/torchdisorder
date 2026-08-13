"""Build a physical a-SiO2 seed structure by MACE melt-quench MD.

Why this exists
---------------
``data/crystal-structures/sio2_glass.cif`` — the seed the SiO2 refinement runs start
from — is not a physical glass.  Before any optimization it already has <CN> = 3.02,
298 Si-O contacts below 1.4 A, and a minimum Si-O distance of 0.185 A.  The refinement
inherits those overlaps and never escapes them (see ``scripts/validate_fis_tetrahedron.py``
test [9]), so every F_IS number measured on the resulting glass is meaningless.

This script generates a replacement seed the honest way: take the *crystal* (which is
clean, <CN> = 4.000), expand it to the experimental glass density, melt it well above
Tm so it forgets the crystalline arrangement, then quench to 300 K under MACE.  The
output is a starting structure with a real first-shell peak, which the scattering
refinement can then fit without having to repair atom overlaps at the same time.

Usage
-----
    poetry run python scripts/build_sio2_glass.py --device mps --melt-steps 4000
    poetry run python scripts/build_sio2_glass.py --benchmark    # time 10 steps and exit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

KB_EV = 8.617333262e-5  # eV/K
SI_Z, O_Z = 14, 8


def health(atoms, cutoff: float = 2.2) -> dict:
    """Si-O coordination diagnostics — the checks the old seed fails."""
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    si_o = (z[i] == SI_Z) & (z[j] == O_Z)
    cn = np.bincount(i[si_o], minlength=len(atoms))[z == SI_Z]
    return {
        "n_atoms": len(atoms),
        "density": float(atoms.get_masses().sum() / atoms.get_volume() * 1.66054),
        "cn_mean": float(cn.mean()),
        "cn4_frac": float((cn == 4).mean()),
        "min_si_o": float(d[si_o].min()) if si_o.any() else float("nan"),
        "n_short": int((d[si_o] < 1.4).sum()),
    }


def report(label: str, atoms) -> dict:
    h = health(atoms)
    print(
        f"  {label:22s} rho={h['density']:.3f}  <CN>={h['cn_mean']:.3f}  "
        f"CN4={100 * h['cn4_frac']:5.1f}%  minSiO={h['min_si_o']:.3f}  "
        f"n(<1.4A)={h['n_short']}"
    )
    return h


def cn_plateau(atoms, cutoffs=(1.8, 2.0, 2.2, 2.4, 2.6, 3.0)) -> None:
    """A physical network shows <CN> flat at 4 across this window; the old seed rises
    monotonically with no plateau."""
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    print("     cutoff (A):  " + "".join(f"{c:7.1f}" for c in cutoffs))
    vals = []
    for c in cutoffs:
        i, j, _ = neighbor_list("ijd", atoms, c)
        m = (z[i] == SI_Z) & (z[j] == O_Z)
        vals.append(np.bincount(i[m], minlength=len(atoms))[z == SI_Z].mean())
    print("     <CN>:        " + "".join(f"{v:7.3f}" for v in vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/crystal-structures/c-SiO2.cif",
                   help="clean crystalline seed (must pass the health check)")
    p.add_argument("--output", default="data/crystal-structures/sio2_glass_mq.cif")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--density", type=float, default=2.20, help="g/cm3, experimental a-SiO2")
    p.add_argument("--melt-temp", type=float, default=4000.0, help="K; well above Tm to erase crystal memory")
    p.add_argument("--quench-temp", type=float, default=300.0)
    p.add_argument("--melt-steps", type=int, default=4000)
    p.add_argument("--quench-steps", type=int, default=6000)
    p.add_argument("--anneal-steps", type=int, default=1000, help="equilibration at quench-temp")
    p.add_argument("--timestep", type=float, default=1.0, help="fs")
    p.add_argument("--gamma", type=float, default=0.1, help="Langevin damping, ps^-1")
    p.add_argument("--relax-steps", type=int, default=200, help="final FIRE steps (0 to skip)")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark", action="store_true", help="time 10 MD steps, then exit")
    args = p.parse_args()

    import torch_sim as ts
    from ase.io import read, write
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel
    from torch_sim.units import MetalUnits

    # torch_sim's internal time unit is sqrt(amu*A^2/eV) ~= 10.18 fs, and
    # MetalUnits.time is 1 ps expressed in it.  Passing dt=1.0 means one
    # *picosecond*, not one femtosecond -- a 1000x overshoot that detonates the
    # integrator (T -> 1e16 K within 50 steps).  Convert explicitly.
    dt = args.timestep * 1e-3 * MetalUnits.time      # fs -> internal
    gamma = args.gamma / MetalUnits.time             # ps^-1 -> internal^-1

    device = torch.device(args.device)
    dtype = torch.float32 if args.device == "mps" else torch.float64
    torch.manual_seed(args.seed)

    print("=" * 74)
    print("a-SiO2 seed construction — MACE melt-quench")
    print("=" * 74)

    atoms = read(args.input)
    print("\nInput (must be clean — this is the whole point):")
    h0 = report("crystal", atoms)
    if h0["n_short"] > 0 or h0["cn_mean"] < 3.99:
        raise SystemExit(
            f"Input {args.input} is not a clean crystal "
            f"(<CN>={h0['cn_mean']:.3f}, {h0['n_short']} short contacts). "
            "Melt-quenching a broken structure reproduces the original bug."
        )

    # Expand isotropically to the experimental glass density.
    scale = (h0["density"] / args.density) ** (1 / 3)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
    report("expanded to glass rho", atoms)

    dtype_name = str(dtype).rsplit(".", 1)[-1]
    print(f"\nLoading MACE-MP (small) on {args.device} / {dtype_name}...")
    # Load on CPU first.  The checkpoint holds float64 tensors and MPS cannot
    # materialise those, so torch.load(map_location="mps") fails outright; the
    # cast to float32 has to happen before the model reaches the device.
    ase_calc = mace_mp(model="small", device="cpu", default_dtype=dtype_name)
    model = MaceModel(
        model=ase_calc.models[0], device=device, dtype=dtype,
        compute_forces=True, compute_stress=False,
    )

    state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

    def temperature(st) -> float:
        return float(ts.calc_temperature(masses=st.masses, momenta=st.momenta))

    if args.benchmark:
        st = ts.nvt_langevin_init(state, model, kT=args.melt_temp * KB_EV, seed=args.seed)
        t0 = time.time()
        for _ in range(10):
            st = ts.nvt_langevin_step(st, model, dt=dt, kT=args.melt_temp * KB_EV, gamma=gamma)
        per = (time.time() - t0) / 10
        total = args.melt_steps + args.quench_steps + args.anneal_steps
        print(f"\n  {per:.3f} s/step  ->  {total} steps = {per * total / 60:.1f} min")
        return

    # ---- Stage 1: melt -----------------------------------------------------
    print(f"\n[1/3] Melt at {args.melt_temp:.0f} K for {args.melt_steps} steps "
          f"({args.melt_steps * args.timestep / 1000:.1f} ps)")
    kT_melt = args.melt_temp * KB_EV
    st = ts.nvt_langevin_init(state, model, kT=kT_melt, seed=args.seed)
    t0 = time.time()
    for step in range(args.melt_steps):
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=kT_melt, gamma=gamma)
        if step % args.log_every == 0:
            print(f"    step {step:5d}  T={temperature(st):7.1f} K  "
                  f"({(time.time() - t0) / max(step, 1):.2f} s/step)", flush=True)

    # ---- Stage 2: linear quench -------------------------------------------
    print(f"\n[2/3] Quench {args.melt_temp:.0f} -> {args.quench_temp:.0f} K over "
          f"{args.quench_steps} steps "
          f"({(args.melt_temp - args.quench_temp) / (args.quench_steps * args.timestep * 1e-3):.2e} K/ps)")
    for step in range(args.quench_steps):
        frac = step / max(args.quench_steps - 1, 1)
        T = args.melt_temp + (args.quench_temp - args.melt_temp) * frac
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=T * KB_EV, gamma=gamma)
        if step % args.log_every == 0:
            print(f"    step {step:5d}  T_set={T:7.1f} K  T={temperature(st):7.1f} K", flush=True)

    # ---- Stage 3: anneal at final temperature ------------------------------
    print(f"\n[3/3] Anneal at {args.quench_temp:.0f} K for {args.anneal_steps} steps")
    kT_final = args.quench_temp * KB_EV
    for step in range(args.anneal_steps):
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=kT_final, gamma=gamma)
        if step % args.log_every == 0:
            print(f"    step {step:5d}  T={temperature(st):7.1f} K", flush=True)

    print("\nAfter melt-quench:")
    glass = ts.io.state_to_atoms(st)[0]
    report("quenched", glass)

    # ---- Optional FIRE relaxation -----------------------------------------
    if args.relax_steps > 0:
        print(f"\nFIRE relaxation ({args.relax_steps} steps max)")
        fst = ts.fire_init(st, model)
        for step in range(args.relax_steps):
            fst = ts.fire_step(fst, model)
            fmax = float(fst.forces.norm(dim=-1).max())
            if fmax < 0.05:
                print(f"    converged at step {step}, |F|max={fmax:.4f} eV/A")
                break
        else:
            print(f"    stopped at |F|max={float(fst.forces.norm(dim=-1).max()):.4f} eV/A")
        glass = ts.io.state_to_atoms(fst)[0]
        report("relaxed", glass)

    # ---- Verdict -----------------------------------------------------------
    print("\nCoordination plateau (a physical network is flat at 4.0):")
    cn_plateau(glass)

    h = health(glass)
    ok = h["n_short"] == 0 and 3.9 <= h["cn_mean"] <= 4.1 and h["min_si_o"] > 1.4
    print("\n" + "=" * 74)
    print(f"VERDICT: {'PASS — usable as a refinement seed' if ok else 'FAIL — do not use'}")
    if not ok:
        print("  Try a longer melt, a slower quench, or a smaller timestep.")
    print("=" * 74)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), glass)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
