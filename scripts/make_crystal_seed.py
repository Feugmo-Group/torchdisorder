"""Build a crystalline starting structure for crystal -> amorphous refinement.

The question this supports
--------------------------
Can the refinement *find* a glass, starting from a crystal and driven by the
experimental scattering data plus the local-environment constraints? That is a
different and harder question than "does the refinement preserve a glass it was
handed", which is what comparing against a published seed measures.

What this script does NOT do
----------------------------
It does not try to make a glass by displacing atoms. That does not work, and the
failure is quantitative: rattling a crystal degrades coordination faster than it
creates glass-like disorder, and it never alters ring topology. Measured on
c-SiO2 scaled to glass density:

    rattle   <CN>    CN=4     min Si-O   O-Si-O spread   Si-Si/Si
      0.1 A  3.995   99.5%      1.291 A      7.25 deg      3.96
      0.2 A  3.760   78.1%      0.865 A     13.92 deg      3.46
      0.3 A  3.299   44.3%      0.457 A     20.38 deg      3.10
    published GAP glass:
             4.001   99.7%      1.517 A       5.38 deg      4.00

The published glass occupies a corner rattling cannot reach -- 5.4 degrees of
angular spread with 99.7% four-fold coordination. Si-Si/Si is the second-shell
count, i.e. the network topology: it stays at the crystal's 4.0 under any
displacement until the network simply collapses. A one-dimensional F(Q) cannot
repair topology, so the disorder has to come from the optimization, not the seed.

So the rattle here is deliberately tiny -- just enough to break the exact
crystalline symmetry, which would otherwise leave the gradients degenerate -- while
keeping every bond outside the covalent floor. The refinement supplies the
disorder.

Usage
-----
    poetry run python scripts/make_crystal_seed.py \
        --input data/crystal-structures/c-SiO2.cif \
        --density 2.20 --rattle 0.02 --output data/crystal-structures/sio2_from_crystal.cif
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="clean crystalline structure")
    p.add_argument("--output", required=True)
    p.add_argument("--density", type=float, required=True, help="target glass density, g/cm3")
    p.add_argument("--rattle", type=float, default=0.02,
                   help="symmetry-breaking displacement, A (default 0.02)")
    p.add_argument("--central", default="Si")
    p.add_argument("--neighbour", default="O")
    p.add_argument("--cutoff", type=float, default=2.2)
    p.add_argument("--expected-cn", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from ase.io import read, write

    from torchdisorder.common.validation import validate_structure

    atoms = read(args.input)
    rho0 = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
    print(f"input: {args.input}  {len(atoms)} atoms, rho = {rho0:.4f} g/cm3")

    check = validate_structure(atoms, check_plateau=True, central=args.central,
                               neighbour=args.neighbour, expected_cn=args.expected_cn)
    if not check:
        raise SystemExit(f"input is not a clean crystal:\n{check.summary()}")

    # Expand isotropically to the experimental glass density.
    atoms.set_cell(atoms.get_cell() * (rho0 / args.density) ** (1 / 3), scale_atoms=True)
    print(f"expanded to rho = "
          f"{atoms.get_masses().sum()/atoms.get_volume()*1.66054:.4f} g/cm3")

    if args.rattle > 0:
        rng = np.random.default_rng(args.seed)
        atoms.set_positions(atoms.get_positions()
                            + rng.normal(0.0, args.rattle, (len(atoms), 3)))
        print(f"applied {args.rattle} A symmetry-breaking rattle (seed {args.seed})")

    report = validate_structure(atoms, check_plateau=True, central=args.central,
                                neighbour=args.neighbour, expected_cn=args.expected_cn)
    print("\n" + report.summary())
    if not report:
        raise SystemExit(
            "\nSeed is already unphysical before any refinement. Reduce --rattle: "
            "the point is to break symmetry, not to create disorder.")

    write(args.output, atoms, format="cif")
    print(f"\nwrote {args.output}")
    print("\nThis is a CRYSTAL, by construction -- its ring topology is crystalline and")
    print("its angular spread is near zero. Whether it becomes a glass is the result")
    print("of the refinement, and is exactly what should be judged afterwards against")
    print("a published model with scripts/compare_to_literature.py.")


if __name__ == "__main__":
    main()
