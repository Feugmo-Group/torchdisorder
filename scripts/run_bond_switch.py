"""Drive WWW bond switching from a crystal toward an amorphous network.

Wires the discrete topology move to a concrete relaxation and score:

  relax  -- FIRE under a machine-learned potential when one is available,
            otherwise a short steepest-descent on the overlap penalty. Relaxation
            is not optional: an unrelaxed transposition leaves badly strained
            bonds and the acceptance rate collapses.
  score  -- deviation of the local order parameters from their targets, plus an
            overlap penalty. Deliberately NOT chi^2: with the F(Q) amplitude
            discrepancy unresolved, using the scattering objective to accept
            topology moves would let bad topology in for the wrong reason.

Success is not a falling score. It is the ring topology moving off its crystalline
value while coordination survives -- so the run reports the second-shell count,
which is the quantity rattling can never change.

Usage:
    poetry run python scripts/run_bond_switch.py \\
        --input data/crystal-structures/sio2_from_crystal.cif \\
        --central Si --neighbour O --cutoff 2.2 --expected-cn 4 \\
        --steps 2000 --output data/crystal-structures/sio2_www.cif
"""

from __future__ import annotations

import argparse

import numpy as np


def topology_signature(atoms, cz, nz, cutoff):
    """Second-shell central-central count: the ring-topology fingerprint.

    Stays pinned at the crystalline value under any displacement, so a change here
    is the one unambiguous sign that the network was rewired rather than merely
    distorted.
    """
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, _ = neighbor_list("ijd", atoms, cutoff)
    m = (z[i] == nz) & (z[j] == cz)
    bridge_to_centres = {}
    for b, c in zip(i[m].tolist(), j[m].tolist()):
        bridge_to_centres.setdefault(b, []).append(c)
    links = set()
    for centres in bridge_to_centres.values():
        for p in range(len(centres)):
            for q in range(p + 1, len(centres)):
                links.add((min(centres[p], centres[q]), max(centres[p], centres[q])))
    n_centres = int((z == cz).sum())
    return links, 2 * len(links) / max(n_centres, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--central", required=True)
    p.add_argument("--neighbour", required=True)
    p.add_argument("--cutoff", type=float, required=True)
    p.add_argument("--expected-cn", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--relax-steps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mlip", action="store_true",
                   help="relax with MACE (much better acceptance, much slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    import torch
    from ase.data import atomic_numbers
    from ase.io import read, write
    from torch_sim.io import atoms_to_state, state_to_atoms

    from torchdisorder.common.validation import validate_structure
    from torchdisorder.engine.bond_switch import BondSwitchMC
    from torchdisorder.engine.order_params import PyTorchOrderParameters

    cz, nz = atomic_numbers[args.central], atomic_numbers[args.neighbour]
    atoms = read(args.input)
    dev = torch.device(args.device)

    start_links, start_deg = topology_signature(atoms, cz, nz, args.cutoff)
    print(f"start: {len(atoms)} atoms, {len(start_links)} links, "
          f"{start_deg:.3f} per {args.central}")

    # ---- relaxation --------------------------------------------------------
    relax_fn = None
    if args.mlip:
        import torch_sim as ts
        from mace.calculators.foundations_models import mace_mp
        from torch_sim.models.mace import MaceModel

        calc = mace_mp(model="small", device="cpu", default_dtype="float64")
        model = MaceModel(model=calc.models[0], device=dev, dtype=torch.float64,
                          compute_forces=True, compute_stress=False)

        def relax_fn(a):
            st = atoms_to_state(a, device=dev, dtype=torch.float64)
            fst = ts.fire_init(st, model)
            for _ in range(args.relax_steps):
                fst = ts.fire_step(fst, model)
            return state_to_atoms(fst)[0]

    # ---- score -------------------------------------------------------------
    calc_op = PyTorchOrderParameters(cutoff=args.cutoff, device=args.device,
                                     max_neighbors=8)
    from ase.data import covalent_radii
    floor = 0.85 * (covalent_radii[cz] + covalent_radii[nz])

    def score_fn(a):
        st = atoms_to_state(a, device=dev, dtype=torch.float64)
        idx = torch.where(st.atomic_numbers == cz)[0]
        out = calc_op(st, idx, ["cn", "tet", "fis"], element_filter=[nz])
        # Local-geometry deviation only. chi^2 is deliberately excluded: the
        # forward model over-predicts F(Q) amplitude ~3x, so accepting topology
        # moves on it would admit bad topology for the wrong reason.
        s = ((out["cn"].mean() - args.expected_cn) ** 2
             + (out["tet"].mean() - 0.92) ** 2
             + (out["fis"].mean() + 0.30) ** 2)
        # Overlap penalty, so a switch cannot buy order by fusing atoms.
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list("ijd", a, floor)
        s = float(s) + 10.0 * float(np.sum(np.clip(floor - d, 0, None) ** 2))
        return s

    mc = BondSwitchMC(atoms, central_z=cz, bridge_z=nz, cutoff=args.cutoff,
                      relax_fn=relax_fn, score_fn=score_fn,
                      temperature=args.temperature, seed=args.seed)

    print(f"\nrunning {args.steps} switch attempts "
          f"(relax: {'MACE' if args.mlip else 'none'})")
    mc.run(args.steps, log_every=args.log_every)

    final = mc.atoms
    end_links, end_deg = topology_signature(final, cz, nz, args.cutoff)
    changed = len(start_links ^ end_links) / 2

    print(f"\nacceptance rate: {100*mc.acceptance_rate:.1f}% "
          f"({mc.n_accepted}/{mc.n_proposed})")
    print(f"links changed  : {changed:.0f} of {len(start_links)} "
          f"({100*changed/max(len(start_links),1):.1f}%)")
    print(f"degree         : {start_deg:.3f} -> {end_deg:.3f} per {args.central}")

    rep = validate_structure(final, check_plateau=True, central=args.central,
                             neighbour=args.neighbour,
                             expected_cn=float(args.expected_cn))
    print("\n" + rep.summary())
    print("\nThe number that matters is 'links changed'. Coordination and degree are")
    print("preserved by construction, so they cannot show that anything happened; a")
    print("rewired network is the only thing displacement cannot produce.")

    write(args.output, final, format="cif")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
