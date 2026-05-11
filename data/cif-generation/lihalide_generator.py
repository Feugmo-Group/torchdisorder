"""
Glass structure generator for Li-halide electrolytes (Li₃MCl₆, M = Y/In/Sc).

Reads a Li₃MCl₆ crystal CIF, builds a disordered glass model, and writes:
  - <output>.cif              : glass supercell
  - <output>_constraints.json : per-M-atom order-parameter constraints

Usage:
    python lihalide_generator.py \\
        --input data/crystal-structures/Li3YCl6.cif \\
        --metal Y \\
        --supercell 3,3,2 \\
        --target_density 2.55 \\
        --disorder 0.2 \\
        --seed 42 \\
        --output lihalide_Li3YCl6

Each M atom is characterized by:
  - oct : octahedral order parameter (6-fold MCl₆)
  - cn  : coordination number (target = 6)
  - q6  : orientational order (bond-orientational Q6)
  - di  : bond-length distortion index (std/mean of M-Cl distances)

Li atoms use random_icp disorder (no per-atom constraints).
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter, CifParser


METAL_SYMBOLS = {"Y", "In", "Sc"}

NEUTRON_SL = {
    "Li": -1.90, "Y": 7.75, "In": 4.065, "Sc": 12.29, "Cl": 9.577,
}


# ---------------------------------------------------------------------------
# Environment characterisation
# ---------------------------------------------------------------------------

class MClEnvironmentAnalyzer:
    """Analyse [MCl₆] octahedral environments in a Li₃MCl₆ structure."""

    def __init__(self, structure: Structure, metal: str, cutoff: float = 3.5):
        self.structure = structure
        self.metal = metal
        self.cutoff = cutoff

    def _get_cl_neighbors(self, idx: int) -> List[int]:
        """Return indices of Cl atoms within cutoff of site idx."""
        site = self.structure[idx]
        cl_indices = []
        for j, other in enumerate(self.structure):
            if j == idx:
                continue
            if other.specie.symbol == "Cl" and site.distance(other) < self.cutoff:
                cl_indices.append(j)
        return cl_indices

    def _bond_stats(self, m_idx: int, cl_indices: List[int]):
        """Return mean and std of M-Cl bond lengths."""
        site = self.structure[m_idx]
        dists = [site.distance(self.structure[j]) for j in cl_indices]
        if len(dists) == 0:
            return 0.0, 0.0
        dists = np.array(dists)
        return float(dists.mean()), float(dists.std())

    def analyse_m_site(self, m_idx: int) -> Dict:
        cl_nbrs = self._get_cl_neighbors(m_idx)
        n_cl = len(cl_nbrs)
        mean_d, std_d = self._bond_stats(m_idx, cl_nbrs)
        di = std_d / max(mean_d, 1e-6)

        # For ideal octahedra: oct ≈ 0.85, q6 ≈ 0.57, di ≈ 0
        return {
            "atom_index": m_idx,
            "element": self.metal,
            "n_cl": n_cl,
            "mean_bond_length": mean_d,
            "distortion_index": di,
            "cl_neighbor_indices": cl_nbrs,
        }

    def analyse_all_m(self) -> Dict[int, Dict]:
        results = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == self.metal:
                results[i] = self.analyse_m_site(i)
        return results


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------

def load_and_supercell(cif_path: str, supercell: tuple) -> Structure:
    parser = CifParser(cif_path)
    struct = parser.get_structures(primitive=False)[0]
    struct.make_supercell(list(supercell))
    return struct


def apply_disorder(struct: Structure, disorder: float, rng: np.random.Generator) -> Structure:
    """Randomly displace atoms to break crystalline periodicity."""
    new_frac = []
    for site in struct:
        disp_cart = rng.uniform(-disorder, disorder, 3)
        disp_frac = struct.lattice.get_fractional_coords(disp_cart)
        new_frac.append(site.frac_coords + disp_frac)
    return Structure(struct.lattice, [s.specie for s in struct], new_frac)


def scale_to_density(struct: Structure, target_density: float) -> Structure:
    mass = sum(site.specie.atomic_mass for site in struct)
    mass_kg = float(mass) * 1.66054e-27
    target_vol_A3 = (mass_kg / (target_density * 1e3)) * 1e30
    new_lattice = struct.lattice.scale(target_vol_A3)
    return Structure(new_lattice, [s.specie for s in struct], [s.frac_coords for s in struct])


# ---------------------------------------------------------------------------
# Constraint JSON builder
# ---------------------------------------------------------------------------

def build_constraint_json(m_envs: Dict[int, Dict], metal: str) -> Dict:
    atom_constraints = {}
    for m_idx, env in m_envs.items():
        n_cl = env["n_cl"]
        di_crystal = env["distortion_index"]

        # Targets: ideal octahedron, allow some glass distortion
        oct_target = 0.80
        q6_target = 0.50
        di_target = max(di_crystal, 0.01)   # allow at least crystal-level distortion

        atom_constraints[str(m_idx)] = {
            "atom_index": m_idx,
            "element": metal,
            "environment": f"{metal}Cl6",
            "environment_label": f"[{metal}Cl₆] octahedron",
            "target_coordination": 6,
            "order_parameters": {
                "oct": {
                    "target": oct_target,
                    "min": 0.50,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": f"Octahedral order for [{metal}Cl₆]",
                },
                "cn": {
                    "target": float(max(n_cl, 6)),
                    "tolerance": 1.0,
                    "weight": 1.5,
                    "description": f"{metal}-Cl coordination number",
                },
                "q6": {
                    "target": q6_target,
                    "min": 0.20,
                    "max": 1.0,
                    "weight": 1.0,
                    "description": "Bond-orientational Q6",
                },
                "di": {
                    "target": di_target,
                    "min": 0.0,
                    "max": 0.20,
                    "weight": 0.5,
                    "description": "Bond-length distortion index",
                },
            },
            "cl_neighbor_indices": env["cl_neighbor_indices"],
            "mean_bond_length": env["mean_bond_length"],
            "crystal_di": di_crystal,
        }

    return {
        "overlap_repulsion": {
            "enabled": True,
            "r_min": 1.0,
            "weight": 1.0,
            "description": "Quadratic penalty preventing atomic overlaps.",
        },
        "system": f"Li3{metal}Cl6",
        "metal": metal,
        "n_m_atoms": len(m_envs),
        "atom_constraints": atom_constraints,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Li-halide glass structure generator")
    parser.add_argument("--input", required=True, help="Input CIF (Li3MCl6 crystal)")
    parser.add_argument("--metal", required=True, choices=["Y", "In", "Sc"],
                        help="Metal species")
    parser.add_argument("--supercell", default="3,3,2",
                        help="Supercell as a,b,c (default: 3,3,2)")
    parser.add_argument("--target_density", type=float, default=2.55,
                        help="Target glass density in g/cm³")
    parser.add_argument("--disorder", type=float, default=0.2,
                        help="Displacement amplitude for disorder (Å)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output filename prefix (default: lihalide_Li3<M>Cl6)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"lihalide_Li3{args.metal}Cl6"

    rng = np.random.default_rng(args.seed)
    supercell = tuple(int(x) for x in args.supercell.split(","))

    print(f"Loading {args.input} ...")
    struct = load_and_supercell(args.input, supercell)
    print(f"  Supercell: {struct.num_sites} atoms, {struct.volume:.1f} Å³")

    print(f"Scaling to density {args.target_density} g/cm³ ...")
    struct = scale_to_density(struct, args.target_density)

    print(f"Applying disorder (amplitude={args.disorder} Å) ...")
    struct = apply_disorder(struct, args.disorder, rng)

    print(f"Analysing [{args.metal}Cl₆] octahedral environments ...")
    analyzer = MClEnvironmentAnalyzer(struct, args.metal)
    m_envs = analyzer.analyse_all_m()
    cn_vals = [env["n_cl"] for env in m_envs.values()]
    print(f"  {args.metal} atoms: {len(m_envs)}, mean CN = {np.mean(cn_vals):.2f}")

    out_cif = Path(args.output).with_suffix(".cif")
    out_json = Path(str(args.output) + "_constraints.json")

    print(f"Writing CIF → {out_cif}")
    CifWriter(struct).write_file(str(out_cif))

    print(f"Writing constraints → {out_json}")
    constraints = build_constraint_json(m_envs, args.metal)
    with open(out_json, "w") as f:
        json.dump(constraints, f, indent=2)

    print(f"\nDone. {len(m_envs)} {args.metal} atoms constrained.")
    print(f"  CIF        : {out_cif}")
    print(f"  Constraints: {out_json}")


if __name__ == "__main__":
    main()
