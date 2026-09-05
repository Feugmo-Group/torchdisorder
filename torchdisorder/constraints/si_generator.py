"""
Amorphous Silicon Constraint Generator
=======================================

Generates v6-compatible constraint files for amorphous silicon (a-Si) structures.
Classifies Si environments by coordination number and outputs constraints for
environment-based optimization.

Amorphous silicon is a pure elemental system with:
    - Si-Si bonds: ~2.352 Å (gap at ~3.84 Å)
    - Predominantly 4-fold (tetrahedral) coordination
    - Defect sites with CN=3 or CN=5

Features:
    - Supercell generation from CIF files
    - Environment classification (Si4, Si3, Si5)
    - v6-format constraint JSON output

Usage:
    # Generate constraints from CIF (no supercell)
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_glass --supercell 1000

    # Manual replication
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_glass --replicate 3 3 3

Output files:
    - {output}.cif                   : Structure file (supercell if requested)
    - {output}_constraints.json      : v6-format constraints
    - {output}_Si_environments.json  : Machine-readable environment data
    - {output}_Si_environments.txt   : Human-readable summary
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


class SiEnvironmentClassifier:
    """
    Classify Si environments in amorphous silicon by Si-Si coordination number (CN).

    Environment types:
        - Si4: Tetrahedral (CN=4) - ideal amorphous silicon
        - Si3: Undercoordinated (CN=3) - defect site
        - Si5: Overcoordinated (CN=5) - defect site

    Note: Si-Si bond length in a-Si is ~2.352 Å with a gap at ~3.84 Å.
    Default cutoff is set to 2.6 Å to safely capture the first shell.
    """

    def __init__(self, structure: Structure, si_si_cutoff: float = 2.6):
        """
        Args:
            structure: Pymatgen Structure object
            si_si_cutoff: Si-Si bond cutoff distance in Å (default 2.6 Å)
                          Si-Si bonds are typically ~2.352 Å in a-Si
        """
        self.structure = structure
        self.si_si_cutoff = si_si_cutoff

    def get_si_neighbors_of_si(self, si_index: int) -> List[int]:
        """Get indices of Si atoms within cutoff of central Si atom."""
        si_site = self.structure[si_index]
        si_neighbors = []
        for j, other in enumerate(self.structure):
            if j == si_index:
                continue
            if other.specie.symbol != "Si":
                continue
            if si_site.distance(other) <= self.si_si_cutoff:
                si_neighbors.append(j)
        return si_neighbors

    def classify_si_site(self, si_index: int) -> Dict:
        """
        Classify a single Si site by its coordination environment.

        Returns:
            dict with keys: 'type', 'label', 'cn', 'neighbors'
        """
        si_neigh = self.get_si_neighbors_of_si(si_index)
        cn = len(si_neigh)

        # Environment label by CN
        if cn == 4:
            env_type = "Si4"
            label = "Si4 (tetrahedral)"
        elif cn == 3:
            env_type = "Si3"
            label = "Si3 (undercoordinated)"
        elif cn == 5:
            env_type = "Si5"
            label = "Si5 (five-coordinate)"
        else:
            env_type = "Si_unknown"
            label = f"Unknown (Si{cn})"

        return {
            "type": env_type,
            "label": label,
            "cn": cn,
            "neighbors": {"Si": si_neigh},
        }

    def classify_all_si(self) -> Dict[int, Dict]:
        """Classify all Si sites in the structure."""
        out = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "Si":
                out[i] = self.classify_si_site(i)
        return out

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        """Calculate statistics of Si environments."""
        counts = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1

        total_si = len(classifications)
        fractions = {}
        if total_si > 0:
            for k, v in counts.items():
                fractions[k] = 100.0 * v / total_si

        return {
            "counts": dict(counts),
            "fractions": fractions,
            "total_si": total_si,
        }


class SiConstraintWriter:
    """
    Writes v6-format constraints and environment summaries for Si sites in a-Si.

    v6 Format Features:
        - Uses "environment" key (not "environment_type")
        - Includes "environment_priorities" section
        - Compatible with EnvironmentConstrainedOptimizer

    Note: Tetrahedral order parameter target is set to 0.95 reflecting the
    highly ordered tetrahedral network expected in a-Si.
    """

    # Order parameters for each Si environment
    ENVIRONMENT_ORDER_PARAMETERS = {
        "Si4": {  # Tetrahedral - ideal amorphous silicon
            "order_parameters": {
                "tet": {
                    "target": 0.95,
                    "min": 0.80,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around Si (Si4)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "Si-Si coordination number",
                },
            },
            "element_filter": [14],  # Si=14
            "cutoff": 2.6,
        },
        "Si3": {  # Undercoordinated - defect
            "order_parameters": {
                "cn": {
                    "target": 3.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Si-Si coordination (undercoordinated)",
                },
            },
            "element_filter": [14],
            "cutoff": 2.6,
        },
        "Si5": {  # Five-coordinate - defect
            "order_parameters": {
                "cn": {
                    "target": 5.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Si-Si coordination (five-coordinate)",
                },
            },
            "element_filter": [14],
            "cutoff": 2.6,
        },
    }

    # v6: Environment priorities for adaptive penalty weighting
    ENVIRONMENT_PRIORITIES = {
        "Si4": 2.5,  # Tetrahedral - most important, dominant environment in a-Si
        "Si3": 1.0,  # Undercoordinated - defect, less strict
        "Si5": 1.0,  # Five-coordinate - defect, less strict
    }

    def __init__(
        self,
        structure: Structure,
        classifier: SiEnvironmentClassifier,
        include_environments: List[str] = None,
    ):
        """
        Args:
            structure: Pymatgen Structure object
            classifier: SiEnvironmentClassifier instance
            include_environments: List of environments to include (e.g., ['Si4']).
                                  If None, includes all known environments.
                                  Options: 'Si4', 'Si3', 'Si5'
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
            "cutoff": self.classifier.si_si_cutoff,
            "element_filter": [14],  # Si=14
            "atom_constraints": {},
            "environment_priorities": {},
        }

        present_envs = set()

        for si_idx, data in classifications.items():
            env_type = data["type"]

            if env_type not in self.include_environments:
                continue

            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue

            present_envs.add(env_type)
            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]

            atom_constraint = {
                "atom_index": si_idx,
                "element": "Si",
                "environment": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(env_params["order_parameters"]),
                "si_neighbor_indices": data["neighbors"]["Si"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(si_idx)] = atom_constraint

        for env_type in present_envs:
            constraints["environment_priorities"][env_type] = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)

        constraints["global_constraints"] = {
            "description": "Per-Si order parameter constraints for amorphous silicon",
            "total_si_atoms": stats["total_si"],
            "environment_fractions": stats["fractions"],
            "included_environments": list(self.include_environments),
        }

        constraints["metadata"] = {
            "version": "v6",
            "atom_order": atom_order_fingerprint(
                self.structure, constraints["atom_constraints"].keys()
            ),
            "structure_type": "a-si",
            "total_atoms": self.structure.num_sites,
            "composition": str(self.structure.composition),
            "included_environments": list(self.include_environments),
            "order_parameter_types": list(set(
                op for env_type in self.include_environments
                if env_type in self.ENVIRONMENT_ORDER_PARAMETERS
                for op in self.ENVIRONMENT_ORDER_PARAMETERS[env_type]["order_parameters"].keys()
            )),
            "notes": "v6 constraints for EnvironmentConstrainedOptimizer with adaptive penalties",
            "environment_types": {
                "Si4": "Tetrahedral Si4 (CN=4) - ideal amorphous silicon",
                "Si3": "Undercoordinated Si3 (CN=3) - defect",
                "Si5": "Five-coordinate Si5 (CN=5) - defect",
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
        print(f"  - {len(constraints['atom_constraints'])} Si atoms with constraints")
        print(f"  - Environments: {list(constraints['environment_priorities'].keys())}")

        # 2) Machine-readable environments JSON
        env_json = {
            "statistics": self._to_jsonable(stats),
            "classifications": {
                str(k): {
                    "type": v["type"],
                    "label": v["label"],
                    "cn": int(v["cn"]),
                    "coords": [float(x) for x in self.structure[k].coords],
                    "si_neighbors": [int(i) for i in v["neighbors"]["Si"]],
                }
                for k, v in classifications.items()
            },
        }
        env_file = f"{output_prefix}_Si_environments.json"
        with open(env_file, "w") as f:
            json.dump(env_json, f, indent=2)
        print(f"Wrote environment data to: {env_file}")

        # 3) Human-readable text summary
        summary_file = f"{output_prefix}_Si_environments.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Si ENVIRONMENT SUMMARY FOR AMORPHOUS SILICON\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total Si atoms: {stats['total_si']}\n")
            f.write(f"Cutoff (Si-Si): {self.classifier.si_si_cutoff:.3f} Å\n\n")

            f.write("Environment distribution (by Si CN):\n")
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
            f.write("DETAILED Si SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")

            # Group by type
            by_type = defaultdict(list)
            for si_idx, data in classifications.items():
                by_type[data["type"]].append((si_idx, data))

            for env_type in ["Si4", "Si3", "Si5", "Si_unknown"]:
                if env_type in by_type:
                    sites = by_type[env_type]
                    f.write(f"\n{env_type} sites ({len(sites)} atoms):\n")
                    f.write("-" * 70 + "\n")
                    for si_idx, data in sites:
                        c = self.structure[si_idx].coords
                        f.write(
                            f"  Index {si_idx:6d}: {data['label']:24s} "
                            f"CN={data['cn']} "
                            f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                        )
                        f.write(f"      Si neighbors: {data['neighbors']['Si']}\n")

        print(f"Wrote summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Amorphous silicon environment + v6 constraint generator with supercell support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate constraints from crystalline Si (no supercell)
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_glass --supercell 1000

    # Generate supercell with ~2000 atoms
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_large --supercell 2000

    # Manual replication (3x3x3)
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --output si_3x3x3 --replicate 3 3 3

    # Only tetrahedral Si4 (normal a-Si) - RECOMMENDED
    python -m torchdisorder.constraints.si_generator --input c-Si.cif --environments Si4 --output si_glass

Environment types:
    Si4 - Tetrahedral Si (CN=4) - Normal a-Si structure [DEFAULT]
    Si3 - Undercoordinated Si (CN=3) - Defect/dangling bond
    Si5 - Five-coordinate Si (CN=5) - Defect site

Output files:
    {output}.cif                    - Structure file (supercell if requested)
    {output}_constraints.json       - v6-format constraints
    {output}_Si_environments.json   - Machine-readable data
    {output}_Si_environments.txt    - Human-readable summary

Notes:
    - Si-Si bond length in a-Si is ~2.352 Å (gap at ~3.84 Å)
    - Default cutoff is 2.6 Å
    - Pure elemental system: only Si-Si bonds are tracked
        """
    )
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="si_glass", help="Output file prefix")
    parser.add_argument("--cutoff", type=float, default=2.6, help="Si-Si cutoff distance in Å (default: 2.6)")
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
        choices=["Si4", "Si3", "Si5"],
        help="Environment types to include (default: all). Use 'Si4' for normal a-Si."
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("Amorphous Silicon Constraint Generator (v6 format)")
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

    # Classify Si environments
    print(f"\nClassifying Si environments (cutoff={args.cutoff} Å)...")
    classifier = SiEnvironmentClassifier(structure, si_si_cutoff=args.cutoff)
    classifications = classifier.classify_all_si()
    stats = classifier.get_statistics(classifications)

    print(f"  Total Si atoms: {stats['total_si']}")
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
    writer = SiConstraintWriter(structure, classifier, include_environments=include_envs)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
