"""
Glass structure generator for Li-argyrodite (Li₆PS₅X, X = Cl/Br/I).

Reads a Li₆PS₅X crystal CIF, builds a disordered glass model, and writes:
  - <output>.cif            : glass supercell
  - <output>_constraints.json : per-P-atom order-parameter constraints

Usage:
    python argyrodite_generator.py \\
        --input data/crystal-structures/Li6PS5Cl.cif \\
        --halide Cl \\
        --supercell 3,3,3 \\
        --target_density 1.80 \\
        --disorder 0.3 \\
        --seed 42 \\
        --output argyrodite_Li6PS5Cl

The generator classifies each P environment as:
  PS4    — four S/X neighbours, no halide substitution
  PS3X   — three S + one halide neighbour
  PS2X2  — two S + two halide neighbours
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.analysis.local_env import CrystalNN


HALIDES = {"Cl", "Br", "I"}

# Neutron scattering lengths (fm)
SCATTERING_LENGTHS = {
    "Li": -1.90, "P": 5.13, "S": 2.847,
    "Cl": 9.577, "Br": 6.795, "I": 5.28,
}


# ---------------------------------------------------------------------------
# Environment classification
# ---------------------------------------------------------------------------

class PEnvironmentClassifier:
    """Classify P environments in an argyrodite structure."""

    def __init__(self, structure: Structure, halide: str, cutoff: float = 3.2):
        self.structure = structure
        self.halide = halide
        self.cutoff = cutoff

    def _get_sp_neighbors(self, idx: int) -> Dict[str, List[int]]:
        neighbors: Dict[str, List[int]] = defaultdict(list)
        site = self.structure[idx]
        for j, other in enumerate(self.structure):
            if j == idx:
                continue
            dist = site.distance(other)
            if dist < self.cutoff:
                neighbors[other.specie.symbol].append(j)
        return neighbors

    def classify_p_site(self, p_idx: int) -> Dict:
        nbrs = self._get_sp_neighbors(p_idx)
        n_s = len(nbrs.get("S", []))
        n_x = len(nbrs.get(self.halide, []))
        n_p = len(nbrs.get("P", []))

        if n_p == 0 and n_x == 0:
            env = "PS4"
            label = f"PS4 (no halide)"
        elif n_p == 0 and n_x == 1:
            env = "PS3X"
            label = f"PS3{self.halide} (one halide)"
        elif n_p == 0 and n_x == 2:
            env = "PS2X2"
            label = f"PS2{self.halide}2 (two halide)"
        else:
            env = "P_other"
            label = f"P_other (P{n_p}S{n_s}X{n_x})"

        s_neighbor_indices = nbrs.get("S", [])
        x_neighbor_indices = nbrs.get(self.halide, [])

        return {
            "type": env,
            "label": label,
            "n_s": n_s,
            "n_x": n_x,
            "n_p": n_p,
            "s_neighbor_indices": s_neighbor_indices,
            "x_neighbor_indices": x_neighbor_indices,
            "coordination": n_s + n_x,
        }

    def classify_all_p(self) -> Dict[int, Dict]:
        results = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "P":
                results[i] = self.classify_p_site(i)
        return results


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------

def load_and_supercell(cif_path: str, supercell: tuple) -> Structure:
    parser = CifParser(cif_path)
    structures = parser.get_structures(primitive=False)
    struct = structures[0]
    struct.make_supercell(list(supercell))
    return struct


def apply_disorder(struct: Structure, halide: str, disorder: float, rng: np.random.Generator) -> Structure:
    """Randomly displace atoms by up to disorder Å to break crystalline order."""
    new_coords = []
    for site in struct:
        disp = rng.uniform(-disorder, disorder, 3)
        new_coords.append(site.frac_coords + struct.lattice.get_fractional_coords(disp))
    from pymatgen.core import Lattice
    return Structure(struct.lattice, [s.specie for s in struct], new_coords)


def scale_to_density(struct: Structure, target_density: float) -> Structure:
    """Scale the cell volume to match target density (g/cm³)."""
    mass = sum(site.specie.atomic_mass for site in struct)  # in amu
    mass_kg = float(mass) * 1.66054e-27
    target_vol_m3 = mass_kg / (target_density * 1e3)
    target_vol_A3 = target_vol_m3 * 1e30
    scale = (target_vol_A3 / struct.volume) ** (1.0 / 3.0)
    new_lattice = struct.lattice.scale(target_vol_A3)
    return Structure(new_lattice, [s.specie for s in struct], [s.frac_coords for s in struct])


# ---------------------------------------------------------------------------
# Constraint JSON builder
# ---------------------------------------------------------------------------

def build_constraint_json(
    struct: Structure,
    p_envs: Dict[int, Dict],
    halide: str,
) -> Dict:
    """Build the constraint JSON consumed by TorchDisorder."""

    atom_constraints = {}
    for p_idx, env in p_envs.items():
        n_x = env["n_x"]
        # Target tet depends on how many halides substitute S:
        # pure PS4: tet ≈ 0.85; with halide substitution slightly lower
        tet_target = max(0.70, 0.85 - 0.05 * n_x)

        neighbor_indices = env["s_neighbor_indices"] + env["x_neighbor_indices"]

        atom_constraints[str(p_idx)] = {
            "atom_index": p_idx,
            "element": "P",
            "environment": env["type"],
            "environment_label": env["label"],
            "target_coordination": env["coordination"],
            "order_parameters": {
                "tet": {
                    "target": tet_target,
                    "min": 0.60,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": f"Tetrahedrality around P ({env['type']})",
                },
                "cn": {
                    "target": float(env["coordination"]),
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": f"P coordination number ({env['type']})",
                },
            },
            "halide_substitution": n_x,
            "neighbor_indices": neighbor_indices,
            "cn": env["coordination"],
        }

    # Count environment types
    env_counts = defaultdict(int)
    for env in p_envs.values():
        env_counts[env["type"]] += 1

    return {
        "overlap_repulsion": {
            "enabled": True,
            "r_min": 1.0,
            "weight": 1.0,
            "description": "Quadratic penalty preventing atomic overlaps.",
        },
        "system": "argyrodite",
        "halide": halide,
        "n_p_atoms": len(p_envs),
        "environment_counts": dict(env_counts),
        "atom_constraints": atom_constraints,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Argyrodite glass structure generator")
    parser.add_argument("--input", required=True, help="Input CIF (Li6PS5X crystal)")
    parser.add_argument("--halide", default="Cl", choices=["Cl", "Br", "I"],
                        help="Halide species (Cl/Br/I)")
    parser.add_argument("--supercell", default="3,3,3",
                        help="Supercell as a,b,c (default: 3,3,3)")
    parser.add_argument("--target_density", type=float, default=1.80,
                        help="Target glass density in g/cm³")
    parser.add_argument("--disorder", type=float, default=0.2,
                        help="Displacement amplitude for disorder (Å)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="argyrodite_Li6PS5Cl",
                        help="Output filename prefix (no extension)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    supercell = tuple(int(x) for x in args.supercell.split(","))

    print(f"Loading {args.input} ...")
    struct = load_and_supercell(args.input, supercell)
    print(f"  Supercell: {struct.num_sites} atoms, {struct.volume:.1f} Å³")

    print(f"Scaling to density {args.target_density} g/cm³ ...")
    struct = scale_to_density(struct, args.target_density)

    print(f"Applying disorder (amplitude={args.disorder} Å) ...")
    struct = apply_disorder(struct, args.halide, args.disorder, rng)

    print(f"Classifying P environments (halide={args.halide}) ...")
    classifier = PEnvironmentClassifier(struct, args.halide)
    p_envs = classifier.classify_all_p()
    counts = defaultdict(int)
    for env in p_envs.values():
        counts[env["type"]] += 1
    print(f"  P environments: {dict(counts)}")

    out_cif = Path(args.output).with_suffix(".cif")
    out_json = Path(str(args.output) + "_constraints.json")

    print(f"Writing CIF → {out_cif}")
    CifWriter(struct).write_file(str(out_cif))

    print(f"Writing constraints → {out_json}")
    constraints = build_constraint_json(struct, p_envs, args.halide)
    with open(out_json, "w") as f:
        json.dump(constraints, f, indent=2)

    print(f"\nDone. {len(p_envs)} P atoms constrained.")
    print(f"  CIF        : {out_cif}")
    print(f"  Constraints: {out_json}")


if __name__ == "__main__":
    main()
