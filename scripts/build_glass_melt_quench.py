"""Generate an amorphous structure from a crystal by MLIP melt-quench.

Where this fits
---------------
Three routes exist from a crystal to an amorphous model, and they are not
interchangeable:

  1. Published model      -- best when one exists and was made for your material.
                             The silica GAP set was trained on silica; a
                             general-purpose potential will not beat it there.
  2. Melt-quench (this)   -- the physics does the work. The potential knows the
                             energetics, so the ring statistics come out right
                             without ever consulting the scattering data. Use when
                             no published model exists for your system.
  3. WWW bond switching   -- discrete topology moves accepted on chi^2 plus
                             constraints. Useful when no potential is available,
                             but it uses the objective as its acceptance criterion,
                             so a compromised objective can accept bad topology.

Rattling a crystal is NOT on this list, and the failure is quantitative: on c-SiO2
at glass density, a 0.3 A rattle leaves only 44% of Si four-fold with atoms 0.46 A
apart, while the second-shell count stays at the crystalline 4.0. Displacement
degrades coordination faster than it creates disorder and never changes topology.

What melt-quench does and does not fix
--------------------------------------
It produces a physically plausible network in the right basin. It does not know
about your sample: the quench rate is many orders of magnitude faster than any
experiment, so expect an excess of coordination defects, and the result should
then be refined against the measured data. The health gate below and
scripts/compare_to_literature.py are the checks that decide whether it worked.

The melt must actually reach its setpoint
-----------------------------------------
Velocities initialise at the target temperature and equipartition immediately
halves it, so the thermostat has to make up the rest. At gamma = 0.1 ps^-1 that
takes ~10 ps, and a 20 ps melt therefore spends most of its life cold: the SiO2
run that produced q4 = 0.297 against the published 0.142 had only reached 3730 K
of its 4000 K setpoint. Nothing was wrong with the potential. Check the printed
temperature trace before drawing any conclusion from the output -- if T is not at
setpoint within the first ~10% of the melt, the run is invalid.

Usage
-----
    poetry run python scripts/build_glass_melt_quench.py \\
        --input data/crystal-structures/c-SiO2.cif \\
        --central Si --neighbour O --cutoff 2.2 --expected-cn 4 \\
        --density 2.20 --melt-steps 30000 --quench-steps 30000 \\
        --output data/crystal-structures/sio2_meltquench.cif

    # time ten steps and stop
    poetry run python scripts/build_glass_melt_quench.py ... --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

KB_EV = 8.617333262e-5  # eV/K


def health(atoms, central_z, neighbour_z, cutoff, short_frac=0.85):
    """Coordination diagnostics for the central-neighbour pair."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    m = (z[i] == central_z) & (z[j] == neighbour_z)
    cn = np.bincount(i[m], minlength=len(atoms))[z == central_z]
    floor = short_frac * (covalent_radii[central_z] + covalent_radii[neighbour_z])
    return {
        "density": float(atoms.get_masses().sum() / atoms.get_volume() * 1.66054),
        "cn_mean": float(cn.mean()) if cn.size else float("nan"),
        "cn_exact_frac": float((cn == round(cn.mean())).mean()) if cn.size else 0.0,
        "min_bond": float(d[m].min()) if m.any() else float("nan"),
        "n_short": int((d[m] < floor).sum()),
        "floor": float(floor),
    }


def report(label, atoms, cz, nz, cutoff, expected_cn):
    h = health(atoms, cz, nz, cutoff)
    frac4 = 0.0
    from ase.neighborlist import neighbor_list
    z = atoms.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    m = (z[i] == cz) & (z[j] == nz)
    cn = np.bincount(i[m], minlength=len(atoms))[z == cz]
    if cn.size:
        frac4 = float((cn == expected_cn).mean())
    print(f"  {label:24s} rho={h['density']:.3f}  <CN>={h['cn_mean']:.3f}  "
          f"CN={expected_cn}: {100*frac4:5.1f}%  min={h['min_bond']:.3f}  "
          f"short={h['n_short']}")
    return h, frac4


def angles(atoms, centre_z, nb_z, cutoff):
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, _d, D = neighbor_list("ijdD", atoms, cutoff)
    m = (z[i] == centre_z) & (z[j] == nb_z)
    ii, DD = i[m], D[m]
    out = []
    for c in np.unique(ii):
        v = DD[ii == c]
        v = v / np.linalg.norm(v, axis=1)[:, None]
        for p in range(len(v)):
            for q in range(p + 1, len(v)):
                out.append(np.degrees(np.arccos(np.clip(v[p] @ v[q], -1.0, 1.0))))
    return np.array(out) if out else np.array([np.nan])


def _load_potential(args, device, dtype, dtype_name):
    """Build a torch_sim model for the chosen foundation potential.

    Imports live inside each branch on purpose. Only one potential can occupy an
    environment at a time: mace-torch pins e3nn==0.4.4 while SevenNet and
    MatterSim both require e3nn>=0.6, so installing either alongside MACE breaks
    it. The backends therefore live in separate conda envs, and a top-level
    import of all of them would fail in every one of them.
    """
    if args.potential == "mace":
        from mace.calculators.foundations_models import mace_mp
        from torch_sim.models.mace import MaceModel

        # Load on CPU first: the checkpoint holds float64 tensors, which MPS
        # cannot materialise, so torch.load(map_location='mps') fails outright.
        ase_calc = mace_mp(model=args.model, device="cpu", default_dtype=dtype_name)
        return MaceModel(model=ase_calc.models[0], device=device, dtype=dtype,
                         compute_forces=True, compute_stress=False)

    if args.potential == "mattersim":
        from mattersim.forcefield import Potential
        from torch_sim.models.mattersim import MatterSimModel

        # MatterSim is the one foundation model trained across the temperature
        # range this script actually samples (to ~5000 K); MPtrj and Alexandria,
        # behind MACE-MP/MPA, are near-equilibrium sets.
        # load_training_state=False: the checkpoint carries optimiser state we
        # have no use for at inference, and loading it is both slower and a
        # needless failure mode.
        pot = Potential.from_checkpoint(load_path=args.model, device=str(device),
                                        load_training_state=False)
        return MatterSimModel(model=pot, device=device, dtype=dtype)

    if args.potential == "orb":
        from orb_models.forcefield import pretrained
        from torch_sim.models.orb import OrbModel

        orb = getattr(pretrained, args.model)(device=str(device))
        return OrbModel(model=orb, device=device, dtype=dtype)

    if args.potential == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        from torch_sim.models.sevennet import SevenNetModel

        calc = SevenNetCalculator(args.model, device=str(device))
        return SevenNetModel(model=calc.model, device=device, dtype=dtype)

    if args.potential == "graphpes":
        from torch_sim.models.graphpes import GraphPESWrapper

        # Unlike the branches above, --model here is a PATH to a local checkpoint,
        # not a foundation-model name. The reason to want one: every universal
        # potential we have tried reduces P(V) to P(IV) in a Li-P-S melt, forming
        # P-P bonds and shedding sulfur as free S2-, and lowering the temperature
        # far enough to suppress it leaves the crystal only half-melted. A model
        # trained on the Li2S-P2S5 tie-line itself (LiPS-25) is the way out.
        # GraphPESWrapper accepts the path directly and calls load_model on it.
        return GraphPESWrapper(model=args.model, device=device, dtype=dtype,
                               compute_forces=True, compute_stress=False)

    raise SystemExit(f"unknown potential: {args.potential}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="clean crystalline structure")
    p.add_argument("--output", required=True)
    p.add_argument("--central", required=True, help="e.g. Si, Ge")
    p.add_argument("--neighbour", required=True, help="e.g. O")
    p.add_argument("--cutoff", type=float, required=True)
    p.add_argument("--expected-cn", type=int, default=4)
    p.add_argument("--density", type=float, required=True, help="g/cm3")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    # "small" is MACE-MP-0 L0 from December 2023 -- the oldest and weakest
    # foundation model, and the one this script used to default to. mace-torch's
    # own default since 3.10 is medium-mpa-0 (MPtrj + Alexandria), and it warns
    # when you pick anything else. Other options: medium-0b3, and the OMat24
    # models (small/medium-omat-0), which are usually strongest but carry the
    # Academic Software License -- loading one is accepting its terms.
    p.add_argument("--model", default="medium-mpa-0",
                   help="foundation model name/checkpoint for the chosen --potential "
                        "(default: medium-mpa-0, a MACE model)")
    p.add_argument("--potential", default="mace",
                   choices=["mace", "mattersim", "orb", "sevennet", "graphpes"],
                   help="which potential to drive the dynamics. Each needs its own "
                        "conda env -- they have conflicting e3nn pins. Note that "
                        "'graphpes' is not a foundation model: it loads a local "
                        "checkpoint given by --model.")
    p.add_argument("--melt-temp", type=float, default=4000.0)
    p.add_argument("--quench-temp", type=float, default=300.0)
    p.add_argument("--melt-steps", type=int, default=30000)
    p.add_argument("--quench-steps", type=int, default=30000)
    p.add_argument("--anneal-steps", type=int, default=5000)
    p.add_argument("--timestep", type=float, default=1.0, help="fs")
    # 1.0 ps^-1 reaches the setpoint in 2-3 ps.  Do not lower this without checking
    # the temperature trace: 0.1 leaves a 20 ps melt cold and it silently reads as
    # a crystalline result rather than as a failed run.
    p.add_argument("--gamma", type=float, default=1.0, help="Langevin damping, ps^-1")
    p.add_argument("--relax-steps", type=int, default=300)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()

    import torch_sim as ts
    from ase.data import atomic_numbers
    from ase.io import read, write
    from torch_sim.units import MetalUnits

    cz, nz = atomic_numbers[args.central], atomic_numbers[args.neighbour]

    # torch_sim's internal time unit is sqrt(amu*A^2/eV) ~ 10.18 fs, and
    # MetalUnits.time is one picosecond expressed in it.  Passing dt=1.0 means one
    # PICOsecond, a 1000x overshoot that sends the temperature to ~1e16 K within
    # fifty steps.  Convert explicitly.
    dt = args.timestep * 1e-3 * MetalUnits.time
    gamma = args.gamma / MetalUnits.time

    device = torch.device(args.device)
    dtype = torch.float32 if args.device == "mps" else torch.float64
    torch.manual_seed(args.seed)

    print("=" * 74)
    print(f"Melt-quench: {args.central}-{args.neighbour}, target rho = {args.density} g/cm3")
    print("=" * 74)

    atoms = read(args.input)
    print(f"\ninput ({len(atoms)} atoms):")
    h0, f0 = report("crystal", atoms, cz, nz, args.cutoff, args.expected_cn)
    if h0["n_short"] > 0:
        raise SystemExit(
            f"input has {h0['n_short']} contacts below {h0['floor']:.2f} A. "
            "Melt-quenching a broken structure reproduces the original problem.")

    atoms.set_cell(atoms.get_cell() * (h0["density"] / args.density) ** (1 / 3),
                   scale_atoms=True)
    report("expanded to glass rho", atoms, cz, nz, args.cutoff, args.expected_cn)

    dtype_name = str(dtype).rsplit(".", 1)[-1]
    print(f"\nloading {args.potential} ({args.model}) on {args.device}/{dtype_name}")
    model = _load_potential(args, device, dtype, dtype_name)
    state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

    def temperature(st):
        return float(ts.calc_temperature(masses=st.masses, momenta=st.momenta))

    if args.benchmark:
        st = ts.nvt_langevin_init(state, model, kT=args.melt_temp * KB_EV, seed=args.seed)
        t0 = time.time()
        for _ in range(10):
            st = ts.nvt_langevin_step(st, model, dt=dt, kT=args.melt_temp * KB_EV,
                                      gamma=gamma)
        per = (time.time() - t0) / 10
        total = args.melt_steps + args.quench_steps + args.anneal_steps
        print(f"\n  {per:.3f} s/step -> {total} steps = {per*total/3600:.1f} h")
        return

    # ---- melt --------------------------------------------------------------
    ps = args.melt_steps * args.timestep / 1000
    print(f"\n[1/3] melt at {args.melt_temp:.0f} K, {args.melt_steps} steps ({ps:.1f} ps)")
    print("      long enough to lose the crystal: a 0.1 ps run leaves the network")
    print("      topology untouched and yields a hot crystal, not a glass.")
    kT = args.melt_temp * KB_EV
    st = ts.nvt_langevin_init(state, model, kT=kT, seed=args.seed)
    t0 = time.time()
    for step in range(args.melt_steps):
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=kT, gamma=gamma)
        if step % args.log_every == 0:
            print(f"      {step:6d}  T={temperature(st):7.1f} K  "
                  f"({(time.time()-t0)/max(step,1):.2f} s/step)", flush=True)

    # ---- quench ------------------------------------------------------------
    rate = (args.melt_temp - args.quench_temp) / (args.quench_steps * args.timestep * 1e-3)
    print(f"\n[2/3] quench {args.melt_temp:.0f} -> {args.quench_temp:.0f} K over "
          f"{args.quench_steps} steps ({rate:.2e} K/ps)")
    for step in range(args.quench_steps):
        frac = step / max(args.quench_steps - 1, 1)
        T = args.melt_temp + (args.quench_temp - args.melt_temp) * frac
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=T * KB_EV, gamma=gamma)
        if step % args.log_every == 0:
            print(f"      {step:6d}  T_set={T:7.1f}  T={temperature(st):7.1f} K", flush=True)

    # ---- anneal ------------------------------------------------------------
    print(f"\n[3/3] anneal at {args.quench_temp:.0f} K, {args.anneal_steps} steps")
    kT = args.quench_temp * KB_EV
    for step in range(args.anneal_steps):
        st = ts.nvt_langevin_step(st, model, dt=dt, kT=kT, gamma=gamma)
        if step % args.log_every == 0:
            print(f"      {step:6d}  T={temperature(st):7.1f} K", flush=True)

    glass = ts.io.state_to_atoms(st)[0]
    print("\nafter quench:")
    report("quenched", glass, cz, nz, args.cutoff, args.expected_cn)

    if args.relax_steps > 0:
        print(f"\nFIRE relaxation ({args.relax_steps} steps max)")
        fst = ts.fire_init(st, model)
        for step in range(args.relax_steps):
            fst = ts.fire_step(fst, model)
            fmax = float(fst.forces.norm(dim=-1).max())
            if fmax < 0.05:
                print(f"      converged at step {step}, |F|max={fmax:.4f} eV/A")
                break
        glass = ts.io.state_to_atoms(fst)[0]
        report("relaxed", glass, cz, nz, args.cutoff, args.expected_cn)

    # ---- verdict -----------------------------------------------------------
    from torchdisorder.common.validation import validate_structure

    rep = validate_structure(glass, check_plateau=True, central=args.central,
                             neighbour=args.neighbour, expected_cn=float(args.expected_cn))
    intra = angles(glass, cz, nz, args.cutoff)
    inter = angles(glass, nz, cz, args.cutoff)
    print(f"\n{args.neighbour}-{args.central}-{args.neighbour} = "
          f"{intra.mean():.2f} +/- {intra.std():.2f} deg")
    print(f"{args.central}-{args.neighbour}-{args.central} = "
          f"{inter.mean():.2f} +/- {inter.std():.2f} deg")
    print("\n" + rep.summary())

    print("\n" + "=" * 74)
    if rep:
        print("VERDICT: PASS -- but a passing health check is NOT proof of a glass.")
        print("A hot crystal also passes. Confirm with the angular spread above and")
        print("with scripts/compare_to_literature.py against a published model.")
    else:
        print("VERDICT: FAIL -- do not use. Try a longer melt or a slower quench.")
    print("=" * 74)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), glass, format="cif")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
