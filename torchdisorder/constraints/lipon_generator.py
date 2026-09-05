"""
LiPON Glass Structure Constraint Generator
===========================================

Generates v6-compatible constraint files for LiPON (lithium phosphorus oxynitride)
glass structures. Classifies P environments by O and N coordination, with special
emphasis on the PO3N environment that gives LiPON its ionic conductivity.

LiPON glass key features:
    - P-O bonds: 1.557-1.562 Å, P-N bonds: ~1.557-1.562 Å (gap at ~2.99 Å)
    - N substitutes for O in the phosphate tetrahedral network
    - PO3N is the key environment: N-substituted tetrahedral
    - Li ionic conductivity correlates with PO3N fraction

Environment types (by n_O, n_N count within cutoff):
    - PO4:  n_O=4, n_N=0 → standard orthophosphate tetrahedral
    - PO3N: n_O=3, n_N=1 → N-substituted tetrahedral (key LiPON feature)
    - PO2N: n_O=2, n_N=1 → defect, target CN=4
    - PO1N: n_O=1, n_N=1 → strongly defected, target CN=4
    - P_other: catch-all for unrecognized environments

Features:
    - Supercell generation from CIF files
    - Separate O and N neighbor counting within a single cutoff
    - v6-format constraint JSON output

Usage:
    # Generate constraints from CIF (no supercell)
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_glass --supercell 1000

    # Manual replication
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_glass --replicate 2 2 2

Output files:
    - {output}.cif                   : Structure file (supercell if requested)
    - {output}_constraints.json      : v6-format constraints
    - {output}_P_environments.json   : Machine-readable environment data
    - {output}_P_environments.txt    : Human-readable summary
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import argparse

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from torchdisorder.constraints.fingerprint import atom_order_fingerprint


def create_supercell(
    structure: Structure,
    target_atoms: Optional[int] = None,
    replicate: Optional[List[int]] = None,
) -> Tuple[Structure, List[int]]:
    """
    Create a supercell from a pymatgen Structure.

    Parameters
    ----------
    structure : Structure
        Unit cell to replicate
    target_atoms : int, optional
        Target number of atoms. Will find closest supercell.
    replicate : list of int, optional
        Manual [na, nb, nc] replication factors.

    Returns
    -------
    Structure
        Supercell structure
    list
        Replication factors used [na, nb, nc]
    """
    n_unit = len(structure)

    if target_atoms is not None:
        # Find best replication to get close to target_atoms
        ratio = target_atoms / n_unit
        n_rep = max(1, round(ratio ** (1 / 3)))

        # Try to find optimal [na, nb, nc]
        best_rep = [n_rep, n_rep, n_rep]
        best_diff = abs(n_unit * n_rep**3 - target_atoms)

        # Search nearby combinations
        for na in range(max(1, n_rep - 2), n_rep + 4):
            for nb in range(max(1, n_rep - 2), n_rep + 4):
                for nc in range(max(1, n_rep - 2), n_rep + 4):
                    n_total = n_unit * na * nb * nc
                    diff = abs(n_total - target_atoms)
                    if diff < best_diff:
                        best_diff = diff
                        best_rep = [na, nb, nc]

        replicate = best_rep
        print(f"  Auto-computed replication: {replicate} → {n_unit * np.prod(replicate)} atoms (target: {target_atoms})")

    elif replicate is None:
        replicate = [1, 1, 1]

    # Create supercell using pymatgen
    if replicate != [1, 1, 1]:
        structure = structure.copy()
        structure.make_supercell(replicate)

    return structure, replicate


class LiPONEnvironmentClassifier:
    """
    Classify P environments in LiPON by O and N coordination counts.

    P is the central atom; neighbors are O and N combined.
    O and N are counted separately to distinguish environment types.

    Environment types:
        - PO4:   n_O=4, n_N=0 → standard orthophosphate tetrahedral
        - PO3N:  n_O=3, n_N=1 → N-substituted tetrahedral (key LiPON feature)
        - PO2N:  n_O=2, n_N=1 → defect environment
        - PO1N:  n_O=1, n_N=1 → strongly defected
        - P_other: catch-all

    Note: P-O and P-N bonds in LiPON are both ~1.557-1.562 Å (gap at ~2.99 Å).
    Default cutoff is 1.95 Å to capture all P-O and P-N bonds.
    """

    def __init__(self, structure: Structure, p_cutoff: float = 1.95):
        """
        Args:
            structure: Pymatgen Structure object
            p_cutoff: P-(O,N) bond cutoff distance in Å (default 1.95 Å)
                      P-O and P-N bonds are typically 1.557-1.562 Å
        """
        self.structure = structure
        self.p_cutoff = p_cutoff

    def get_on_neighbors_of_p(self, p_index: int) -> Tuple[List[int], List[int]]:
        """
        Get indices of O and N atoms within cutoff of a P atom.

        Returns
        -------
        o_neighbors : list of int
            Indices of O atoms within cutoff
        n_neighbors : list of int
            Indices of N atoms within cutoff
        """
        p_site = self.structure[p_index]
        o_neighbors = []
        n_neighbors = []
        for j, other in enumerate(self.structure):
            if j == p_index:
                continue
            symbol = other.specie.symbol
            if symbol not in ("O", "N"):
                continue
            if p_site.distance(other) <= self.p_cutoff:
                if symbol == "O":
                    o_neighbors.append(j)
                else:
                    n_neighbors.append(j)
        return o_neighbors, n_neighbors

    def classify_p_site(self, p_index: int) -> Dict:
        """
        Classify a single P site by its O and N coordination environment.

        Returns:
            dict with keys: 'type', 'label', 'cn', 'n_o', 'n_n', 'neighbors'
        """
        o_neigh, n_neigh = self.get_on_neighbors_of_p(p_index)
        n_o = len(o_neigh)
        n_n = len(n_neigh)
        cn = n_o + n_n

        # Classify by (n_O, n_N)
        if n_o == 4 and n_n == 0:
            env_type = "PO4"
            label = "PO4 (orthophosphate tetrahedral)"
        elif n_o == 3 and n_n == 1:
            env_type = "PO3N"
            label = "PO3N (N-substituted tetrahedral)"
        elif n_o == 2 and n_n == 1:
            env_type = "PO2N"
            label = "PO2N (defect, CN=3 O+N)"
        elif n_o == 1 and n_n == 1:
            env_type = "PO1N"
            label = "PO1N (strongly defected)"
        else:
            env_type = "P_other"
            label = f"P_other (nO={n_o}, nN={n_n}, CN={cn})"

        return {
            "type": env_type,
            "label": label,
            "cn": cn,
            "n_o": n_o,
            "n_n": n_n,
            "neighbors": {"O": o_neigh, "N": n_neigh},
        }

    def classify_all_p(self) -> Dict[int, Dict]:
        """Classify all P sites in the structure."""
        out = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "P":
                out[i] = self.classify_p_site(i)
        return out

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        """Calculate statistics of P environments."""
        counts = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1

        total_p = len(classifications)
        fractions = {}
        if total_p > 0:
            for k, v in counts.items():
                fractions[k] = 100.0 * v / total_p

        return {
            "counts": dict(counts),
            "fractions": fractions,
            "total_p": total_p,
        }


class LiPONConstraintWriter:
    """
    Writes v6-format constraints and environment summaries for P sites in LiPON.

    v6 Format Features:
        - Uses "environment" key (not "environment_type")
        - Includes "environment_priorities" section
        - Compatible with EnvironmentConstrainedOptimizer

    The PO3N environment has the highest priority (3.0) as it is the key
    N-substituted environment responsible for LiPON's ionic conductivity.
    """

    # Order parameters for each P environment
    ENVIRONMENT_ORDER_PARAMETERS = {
        "PO4": {  # Standard orthophosphate tetrahedral
            "order_parameters": {
                "tet": {
                    "target": 0.88,
                    "min": 0.70,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around P (PO4)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "P-(O+N) coordination number",
                },
            },
            "element_filter": [7, 8, 15],  # N=7, O=8, P=15
            "cutoff": 1.95,
        },
        "PO3N": {  # N-substituted tetrahedral - key LiPON environment
            "order_parameters": {
                "tet": {
                    "target": 0.85,
                    "min": 0.65,
                    "max": 1.0,
                    "weight": 2.5,
                    "description": "Tetrahedrality around P (PO3N)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 2.0,
                    "description": "P-(O+N) coordination number",
                },
            },
            "element_filter": [7, 8, 15],
            "cutoff": 1.95,
        },
        "PO2N": {  # Defect - optimizer should find more neighbors
            "order_parameters": {
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "P-(O+N) coordination number (defect)",
                },
            },
            "element_filter": [7, 8, 15],
            "cutoff": 1.95,
        },
        "PO1N": {  # Strongly defected
            "order_parameters": {
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "P-(O+N) coordination number (strongly defected)",
                },
            },
            "element_filter": [7, 8, 15],
            "cutoff": 1.95,
        },
        "P_other": {  # Catch-all
            "order_parameters": {
                "cn": {
                    "target": 4.0,
                    "tolerance": 1.0,
                    "weight": 0.5,
                    "description": "P-(O+N) coordination number (other)",
                },
            },
            "element_filter": [7, 8, 15],
            "cutoff": 1.95,
        },
    }

    # v6: Environment priorities for adaptive penalty weighting
    ENVIRONMENT_PRIORITIES = {
        "PO4": 2.0,    # Standard tetrahedral - important
        "PO3N": 3.0,   # Highest - N-substituted tetrahedral, key LiPON feature
        "PO2N": 1.5,   # Defect - moderate
        "PO1N": 1.0,   # Strongly defected - low
        "P_other": 0.5,  # Catch-all - lowest
    }

    def __init__(
        self,
        structure: Structure,
        classifier: LiPONEnvironmentClassifier,
        include_environments: List[str] = None,
    ):
        """
        Args:
            structure: Pymatgen Structure object
            classifier: LiPONEnvironmentClassifier instance
            include_environments: List of environments to include.
                                  If None, includes all known environments.
                                  Options: 'PO4', 'PO3N', 'PO2N', 'PO1N', 'P_other'
        """
        self.structure = structure
        self.classifier = classifier

        # Filter which environments to constrain
        if include_environments is None:
            self.include_environments = list(self.ENVIRONMENT_ORDER_PARAMETERS.keys())
        else:
            # Validate
            valid = set(self.ENVIRONMENT_ORDER_PARAMETERS.keys())
            for env in include_environments:
                if env not in valid:
                    raise ValueError(f"Unknown environment '{env}'. Valid: {valid}")
            self.include_environments = include_environments

    def _to_jsonable(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_constraints(self, classifications: Dict[int, Dict], stats: Dict) -> Dict:
        """
        Generate v6-format constraints for EnvironmentConstrainedOptimizer.
        """
        constraints = {
            "cutoff": self.classifier.p_cutoff,
            "element_filter": [7, 8, 15],  # N=7, O=8, P=15
            "atom_constraints": {},
            "environment_priorities": {},
        }

        present_envs = set()

        for p_idx, data in classifications.items():
            env_type = data["type"]

            if env_type not in self.include_environments:
                continue

            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue

            present_envs.add(env_type)
            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]

            atom_constraint = {
                "atom_index": p_idx,
                "element": "P",
                "environment": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "n_o": data["n_o"],
                "n_n": data["n_n"],
                "order_parameters": dict(env_params["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "n_neighbor_indices": data["neighbors"]["N"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(p_idx)] = atom_constraint

        for env_type in present_envs:
            constraints["environment_priorities"][env_type] = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)

        constraints["global_constraints"] = {
            "description": "Per-P order parameter constraints for LiPON glass",
            "total_p_atoms": stats["total_p"],
            "environment_fractions": stats["fractions"],
            "included_environments": list(self.include_environments),
        }

        constraints["metadata"] = {
            "version": "v6",
            "atom_order": atom_order_fingerprint(
                self.structure, constraints["atom_constraints"].keys()
            ),
            "structure_type": "lipon",
            "total_atoms": self.structure.num_sites,
            "composition": str(self.structure.composition),
            "included_environments": list(self.include_environments),
            "order_parameter_types": list(set(
                op for env_type in self.include_environments
                if env_type in self.ENVIRONMENT_ORDER_PARAMETERS
                for op in self.ENVIRONMENT_ORDER_PARAMETERS[env_type]["order_parameters"].keys()
            )),
            "notes": (
                "PO3N is the N-substituted tetrahedral environment that gives LiPON its ionic conductivity. "
                "F_IS distinguishes PO4 from PO3N. "
                "v6 constraints for EnvironmentConstrainedOptimizer with adaptive penalties."
            ),
            "environment_types": {
                "PO4": "Standard orthophosphate tetrahedral (n_O=4, n_N=0)",
                "PO3N": "N-substituted tetrahedral (n_O=3, n_N=1) - key LiPON feature",
                "PO2N": "Defect environment (n_O=2, n_N=1)",
                "PO1N": "Strongly defected (n_O=1, n_N=1)",
                "P_other": "Catch-all for unrecognized P environments",
            },
        }

        return self._to_jsonable(constraints)

    def write_outputs(self, output_prefix: str, classifications: Dict[int, Dict], stats: Dict):
        """Write all output files including CIF structure."""
        # 0) Write structure CIF file
        cif_file = f"{output_prefix}.cif"
        writer = CifWriter(self.structure)
        writer.write_file(cif_file)
        print(f"Wrote structure to: {cif_file}")

        # 1) v6-format constraints JSON
        constraints = self.generate_constraints(classifications, stats)
        constraints_file = f"{output_prefix}_constraints.json"
        with open(constraints_file, "w") as f:
            json.dump(constraints, f, indent=2)
        print(f"Wrote v6 constraints to: {constraints_file}")
        print(f"  - {len(constraints['atom_constraints'])} P atoms with constraints")
        print(f"  - Environments: {list(constraints['environment_priorities'].keys())}")

        # 2) Machine-readable environments JSON
        env_json = {
            "statistics": self._to_jsonable(stats),
            "classifications": {
                str(k): {
                    "type": v["type"],
                    "label": v["label"],
                    "cn": int(v["cn"]),
                    "n_o": int(v["n_o"]),
                    "n_n": int(v["n_n"]),
                    "coords": [float(x) for x in self.structure[k].coords],
                    "o_neighbors": [int(i) for i in v["neighbors"]["O"]],
                    "n_neighbors": [int(i) for i in v["neighbors"]["N"]],
                }
                for k, v in classifications.items()
            },
        }
        env_file = f"{output_prefix}_P_environments.json"
        with open(env_file, "w") as f:
            json.dump(env_json, f, indent=2)
        print(f"Wrote environment data to: {env_file}")

        # 3) Human-readable text summary
        summary_file = f"{output_prefix}_P_environments.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("P ENVIRONMENT SUMMARY FOR LiPON GLASS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total P atoms: {stats['total_p']}\n")
            f.write(f"Cutoff (P-O/N): {self.classifier.p_cutoff:.3f} Å\n\n")

            f.write("Environment distribution (by P coordination):\n")
            f.write("-" * 70 + "\n")
            for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
                priority = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)
                f.write(f"  {env_type:12s}: {frac:6.2f}%  (count={stats['counts'][env_type]}, priority={priority})\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("ORDER PARAMETER CONSTRAINTS BY ENVIRONMENT\n")
            f.write("=" * 70 + "\n\n")

            for env_type in sorted(stats["counts"].keys()):
                if env_type in self.ENVIRONMENT_ORDER_PARAMETERS:
                    count = stats["counts"][env_type]
                    priority = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)
                    f.write(f"\n{env_type} environments ({count} atoms, priority={priority}):\n")
                    f.write("-" * 70 + "\n")
                    for op_name, op_params in self.ENVIRONMENT_ORDER_PARAMETERS[env_type]["order_parameters"].items():
                        f.write(f"  {op_name}: target={op_params.get('target', 'N/A')}, ")
                        if "min" in op_params:
                            f.write(f"range=[{op_params['min']:.2f}, {op_params['max']:.2f}], ")
                        if "tolerance" in op_params:
                            f.write(f"tolerance={op_params['tolerance']:.2f}, ")
                        f.write(f"weight={op_params['weight']:.1f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED P SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")

            # Group by type
            by_type = defaultdict(list)
            for p_idx, data in classifications.items():
                by_type[data["type"]].append((p_idx, data))

            for env_type in ["PO4", "PO3N", "PO2N", "PO1N", "P_other"]:
                if env_type in by_type:
                    sites = by_type[env_type]
                    f.write(f"\n{env_type} sites ({len(sites)} atoms):\n")
                    f.write("-" * 70 + "\n")
                    for p_idx, data in sites:
                        c = self.structure[p_idx].coords
                        f.write(
                            f"  Index {p_idx:6d}: {data['label']:38s} "
                            f"CN={data['cn']} (nO={data['n_o']}, nN={data['n_n']}) "
                            f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                        )
                        f.write(f"      O neighbors: {data['neighbors']['O']}\n")
                        f.write(f"      N neighbors: {data['neighbors']['N']}\n")

        print(f"Wrote summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LiPON environment + v6 constraint generator with supercell support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate constraints from LiPON CIF (no supercell)
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_glass --supercell 1000

    # Manual replication (2x2x2)
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --output lipon_2x2x2 --replicate 2 2 2

    # Only PO4 and PO3N environments
    python -m torchdisorder.constraints.lipon_generator --input LiPON_defected.cif --environments PO4 PO3N --output lipon_glass

Environment types:
    PO4   - Standard orthophosphate tetrahedral (n_O=4, n_N=0)
    PO3N  - N-substituted tetrahedral (n_O=3, n_N=1) - KEY LiPON FEATURE [highest priority]
    PO2N  - Defect environment (n_O=2, n_N=1)
    PO1N  - Strongly defected (n_O=1, n_N=1)
    P_other - Catch-all for unrecognized environments

Output files:
    {output}.cif                    - Structure file (supercell if requested)
    {output}_constraints.json       - v6-format constraints
    {output}_P_environments.json    - Machine-readable data
    {output}_P_environments.txt     - Human-readable summary

Notes:
    - P-O and P-N bonds are both ~1.557-1.562 Å (gap at ~2.99 Å)
    - Default cutoff is 1.95 Å to capture all P-O and P-N bonds
    - O and N neighbors are counted separately for environment classification
    - PO3N has priority 3.0 (highest) as the key conductivity-enabling environment
        """
    )
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="lipon_glass", help="Output file prefix")
    parser.add_argument("--cutoff", type=float, default=1.95, help="P-(O,N) cutoff distance in Å (default: 1.95)")
    parser.add_argument(
        "--supercell",
        type=int,
        default=None,
        metavar="N",
        help="Target number of atoms for supercell (auto-compute replication)"
    )
    parser.add_argument(
        "--replicate",
        type=int,
        nargs=3,
        default=None,
        metavar=("NA", "NB", "NC"),
        help="Manual replication factors [na nb nc]"
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=None,
        choices=["PO4", "PO3N", "PO2N", "PO1N", "P_other"],
        help="Environment types to include (default: all)."
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("LiPON Constraint Generator (v6 format)")
    print(f"{'=' * 70}")

    # Load structure
    print(f"\nLoading structure from: {args.input}")
    structure = Structure.from_file(args.input)
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Number of sites: {structure.num_sites}")

    # Create supercell if requested
    if args.supercell is not None or args.replicate is not None:
        print(f"\nCreating supercell...")
        structure, rep = create_supercell(
            structure,
            target_atoms=args.supercell,
            replicate=args.replicate
        )
        print(f"  Replication: {rep}")
        print(f"  New structure: {structure.num_sites} atoms")
        print(f"  Formula: {structure.composition.reduced_formula}")

    # Classify P environments
    print(f"\nClassifying P environments (cutoff={args.cutoff} Å)...")
    classifier = LiPONEnvironmentClassifier(structure, p_cutoff=args.cutoff)
    classifications = classifier.classify_all_p()
    stats = classifier.get_statistics(classifications)

    print(f"  Total P atoms: {stats['total_p']}")
    for env_type, frac in stats["fractions"].items():
        print(f"    {env_type}: {frac:.1f}% ({stats['counts'][env_type]} atoms)")

    # Show which environments will be included
    include_envs = args.environments
    if include_envs:
        print(f"\n  Including only: {include_envs}")
        excluded = [e for e in stats["counts"].keys() if e not in include_envs]
        if excluded:
            print(f"  Excluding: {excluded}")
    else:
        print(f"\n  Including all environments")

    # Write outputs
    print(f"\nWriting output files...")
    writer = LiPONConstraintWriter(structure, classifier, include_environments=include_envs)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
