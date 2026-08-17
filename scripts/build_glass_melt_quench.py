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
    # Optional first stage, off by default. For SiO2 a single 4000 K melt works
    # and this is unnecessary; it exists for GeO2 and Li-P-S, where the crystal
    # will not melt below the temperature at which the potential starts
    # destroying the chemistry, so no single melt temperature succeeds.
    p.add_argument("--superheat-temp", type=float, default=None,
                   help="temperature of an optional brief first stage, hotter than "
                        "--melt-temp, to break up the crystal before holding at a "
                        "cooler --melt-temp that preserves the chemistry")
    p.add_argument("--superheat-steps", type=int, default=0,
                   help="steps at --superheat-temp (0 = single-stage melt, default). "
                        "Keep it short: this stage is where chemistry is lost.")
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
    p.add_argument("--system", default=None,
                   help="glass-quality ruleset (SiO2, GeO2, LiPS). Enables the "
                        "forbidden-species and sublattice-disorder gate, and reports "
                        "both live during the run. Without it the final verdict rests "
                        "on coordination alone, which cannot tell a glass from an "
                        "unmelted crystal.")
    p.add_argument("--tolerate", action="append", default=[], metavar="LABEL=N",
                   help="allow N occurrences of a forbidden species, e.g. "
                        "'P-P pairs=8' for the P2S6 units a real Li3PS4 glass has")
    args = p.parse_args()

    if args.superheat_steps > 0 and args.superheat_temp is None:
        raise SystemExit("--superheat-steps needs --superheat-temp")
    if args.superheat_temp is not None and args.superheat_temp <= args.melt_temp:
        raise SystemExit(
            f"--superheat-temp ({args.superheat_temp:.0f} K) must exceed --melt-temp "
            f"({args.melt_temp:.0f} K); the point is to break the crystal hot and "
            "then heal it cooler.")

    tolerated = {}
    for item in args.tolerate:
        label, _, n = item.partition("=")
        if not n.strip().isdigit():
            raise SystemExit(f"--tolerate wants LABEL=N, got {item!r}")
        tolerated[label.strip()] = int(n)

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
    # graph-pes checkpoints are TorchScript, serialised at the precision they were
    # trained in, and LiPS-25's are float32. Handing one a float64 state fails
    # inside the interpreter with "both inputs should have same dtype" -- after
    # the melt has already started, so the job burns a GPU slot before dying.
    if args.potential == "graphpes":
        dtype = torch.float32
    else:
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

    # Live chemistry + disorder, so a doomed run announces itself early. A 3000-atom
    # neighbour list once every --log-every MLIP steps costs nothing measurable.
    monitor = None
    if args.system:
        from torchdisorder.common.glass_quality import (
            GLASS_SYSTEMS, forbidden_contacts, sublattice_disorder)

        if args.system not in GLASS_SYSTEMS:
            raise SystemExit(f"--system {args.system!r} not in "
                             f"{sorted(set(GLASS_SYSTEMS))}")
        spec = GLASS_SYSTEMS[args.system]

        def monitor(st):
            snap = ts.io.state_to_atoms(st)[0]
            bad = forbidden_contacts(snap, spec["forbidden"])
            dis = sublattice_disorder(snap, spec["sublattice"],
                                      rmax=0.5 * float(min(snap.cell.lengths())))
            chem = " ".join(f"{k.split()[0]}={v}" for k, v in bad.items())
            std = dis["long_std"]
            # nan when the cell is too small to have a beyond-6 A range at all;
            # say so rather than printing "nan" every log line.
            shown = "n/a" if np.isnan(std) else f"{std:.3f}"
            return f"std={shown} {chem}"

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

    # ---- dynamics ----------------------------------------------------------
    # One schedule of (label, T_start, T_end, steps) rather than a loop per stage.
    # The stages differ only in their temperature ramp, and keeping them as
    # separate hand-written loops is what made the optional superheat stage
    # awkward enough to skip.
    stages = []
    if args.superheat_steps > 0:
        stages.append(("superheat", args.superheat_temp, args.superheat_temp,
                       args.superheat_steps))
    stages.append(("melt", args.melt_temp, args.melt_temp, args.melt_steps))
    stages.append(("quench", args.melt_temp, args.quench_temp, args.quench_steps))
    if args.anneal_steps > 0:
        stages.append(("anneal", args.quench_temp, args.quench_temp, args.anneal_steps))

    if args.superheat_steps > 0:
        print(f"\ntwo-stage melt: {args.superheat_temp:.0f} K to destroy the crystal, "
              f"then {args.melt_temp:.0f} K to heal it.")
        print("      Rationale: the melting temperature and the temperature at which")
        print("      the potential wrecks the chemistry can be the same number, and")
        print("      no single-temperature melt then works. Chemical damage accrues")
        print("      with TIME at high T while the crystal breaks up quickly, so a")
        print("      brief superheat followed by a cooler hold can beat both.")

    st = ts.nvt_langevin_init(state, model, kT=stages[0][1] * KB_EV, seed=args.seed)
    t0 = time.time()
    done = 0
    for n, (label, t_start, t_end, steps) in enumerate(stages, start=1):
        ps = steps * args.timestep / 1000
        head = f"\n[{n}/{len(stages)}] {label} "
        if t_start == t_end:
            print(f"{head}at {t_start:.0f} K, {steps} steps ({ps:.1f} ps)")
        else:
            rate = abs(t_start - t_end) / max(ps, 1e-12)
            print(f"{head}{t_start:.0f} -> {t_end:.0f} K over {steps} steps "
                  f"({ps:.1f} ps, {rate:.2e} K/ps)")
        if label == "melt":
            print("      long enough to lose the crystal: a 0.1 ps run leaves the")
            print("      network topology untouched and yields a hot crystal.")

        for step in range(steps):
            frac = step / max(steps - 1, 1)
            T = t_start + (t_end - t_start) * frac
            st = ts.nvt_langevin_step(st, model, dt=dt, kT=T * KB_EV, gamma=gamma)
            done += 1
            if step % args.log_every == 0:
                msg = (f"      {step:6d}  T_set={T:7.1f}  T={temperature(st):7.1f} K  "
                       f"({(time.time() - t0) / max(done, 1):.2f} s/step)")
                # Chemistry as the run proceeds, not only at the end. The GeO2
                # dilemma is a race -- the crystal has to break up before the
                # potential starts shedding O2 -- and you cannot see which is
                # winning from a single measurement on the final structure.
                if monitor:
                    msg += "  " + monitor(st)
                print(msg, flush=True)

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

    # bond_cutoff is not optional in practice: without it the plateau window
    # defaults to the Si-O bracket of 1.8-2.6 A, so a P-S first shell at 2.05 A
    # is judged against cutoffs that do not contain it and pristine crystalline
    # Li7P3S11 fails for want of a shell it plainly has.
    rep = validate_structure(glass, check_plateau=True, central=args.central,
                             neighbour=args.neighbour, expected_cn=float(args.expected_cn),
                             bond_cutoff=args.cutoff)
    intra = angles(glass, cz, nz, args.cutoff)
    inter = angles(glass, nz, cz, args.cutoff)
    print(f"\n{args.neighbour}-{args.central}-{args.neighbour} = "
          f"{intra.mean():.2f} +/- {intra.std():.2f} deg")
    print(f"{args.central}-{args.neighbour}-{args.central} = "
          f"{inter.mean():.2f} +/- {inter.std():.2f} deg")
    print("\n" + rep.summary())

    # The health check above is necessary and nowhere near sufficient: it is
    # coordination-based, and an unmelted crystal scores a *perfect* coordination.
    # Three such runs were reported as successes on 2026-08-17. The glass gate is
    # the test that separates them, so run it whenever --system names a ruleset.
    glass_rep = None
    if args.system:
        from torchdisorder.common.glass_quality import assess_glass

        glass_rep = assess_glass(glass, args.system, tolerated=tolerated)
        print("\n" + glass_rep.summary())

    print("\n" + "=" * 74)
    if not rep:
        print("VERDICT: FAIL (health) -- do not use. Overlapping atoms or no first shell.")
    elif glass_rep is None:
        print("VERDICT: INCONCLUSIVE -- health checks pass, but nothing here tested")
        print("whether this is a glass at all, and a hot crystal passes them too.")
        print("Re-run with --system SiO2|GeO2|LiPS to get a real verdict.")
    elif not glass_rep:
        what = []
        if not glass_rep.range_ok:
            what.append("the cell is too small to judge disorder -- this is not a "
                        "verdict on the structure, so re-run from a larger supercell")
        elif not glass_rep.disorder_ok:
            what.append("the melt did not destroy the crystal (hotter or longer, "
                        "or try --superheat-temp)")
        if not glass_rep.chemistry_ok:
            what.append("the potential wrecked the chemistry (cooler melt, a shorter "
                        "--superheat-steps, or a system-specific potential)")
        print("VERDICT: FAIL (not a glass) -- do not use.")
        for w in what:
            print(f"  -> {w}")
    else:
        print("VERDICT: PASS -- disordered and chemically clean.")
        print("Confirm against a published model with scripts/compare_to_literature.py.")
    print("=" * 74)

    # A rejected structure is still written, because it is the only evidence of
    # what went wrong -- but under a name that cannot be mistaken for a usable
    # model, and with a non-zero exit so sacct does not show the job as COMPLETED.
    # Every invalid structure that got used downstream looked like a clean success
    # in the queue history.
    ok = bool(rep) and (glass_rep is None or bool(glass_rep))
    out = Path(args.output)
    if not ok:
        out = out.with_name(f"{out.stem}_REJECTED{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), glass, format="cif")
    print(f"\nwrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
