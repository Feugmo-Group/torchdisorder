"""
GeO2 Glass Structure Constraint Generator
==========================================

Generates v6-compatible constraint files for GeO2 glass structures.
Classifies Ge environments by coordination number and outputs
constraints for environment-based optimization.

GeO2 glass is structurally similar to SiO2 glass but with:
    - Longer Ge-O bonds (~1.73-1.88 Å vs ~1.60-1.62 Å for Si-O)
    - Similar tetrahedral network structure
    - Higher tendency for Ge to adopt higher coordination (5, 6) under pressure

Usage:
    python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --output geo2_glass

Output files:
    - {output}.cif                   : Structure file (copy of input)
    - {output}_constraints.json      : v6-format constraints for EnvironmentConstrainedOptimizer
    - {output}_Ge_environments.json  : Machine-readable environment data
    - {output}_Ge_environments.txt   : Human-readable summary
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
import argparse

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


class GeEnvironmentClassifier:
    """
    Classify Ge environments in GeO2 by Ge-O coordination number (CN).
    
    Environment types:
        - Ge4: GeO4 tetrahedral (CN=4) - ideal glass former
        - Ge3: GeO3 undercoordinated (CN=3) - defect
        - Ge5: GeO5 overcoordinated (CN=5) - intermediate
        - Ge6: GeO6 octahedral (CN=6) - high pressure phase (rutile-type)
    
    Note: Ge has a larger ionic radius than Si, so Ge-O bonds are longer
    (typically 1.73-1.88 Å) and the default cutoff is set accordingly.
    """

    def __init__(self, structure: Structure, ge_o_cutoff: float = 2.4):
        """
        Args:
            structure: Pymatgen Structure object
            ge_o_cutoff: Ge-O bond cutoff distance in Å (default 2.4 Å)
                        Ge-O bonds are typically 1.73-1.88 Å
        """
        self.structure = structure
        self.ge_o_cutoff = ge_o_cutoff

    def get_o_neighbors_of_ge(self, ge_index: int) -> List[int]:
        """Get indices of O atoms within cutoff of Ge atom."""
        ge_site = self.structure[ge_index]
        o_neighbors = []
        for j, other in enumerate(self.structure):
            if j == ge_index:
                continue
            if other.specie.symbol != "O":
                continue
            if ge_site.distance(other) <= self.ge_o_cutoff:
                o_neighbors.append(j)
        return o_neighbors

    def classify_ge_site(self, ge_index: int) -> Dict:
        """
        Classify a single Ge site by its coordination environment.
        
        Returns:
            dict with keys: 'type', 'label', 'cn', 'neighbors'
        """
        o_neigh = self.get_o_neighbors_of_ge(ge_index)
        cn = len(o_neigh)

        # Environment label by CN
        if cn == 4:
            env_type = "Ge4"
            label = "GeO4 (tetrahedral)"
        elif cn == 3:
            env_type = "Ge3"
            label = "GeO3 (undercoordinated)"
        elif cn == 5:
            env_type = "Ge5"
            label = "GeO5 (five-coordinate)"
        elif cn == 6:
            env_type = "Ge6"
            label = "GeO6 (octahedral)"
        else:
            env_type = "Ge_unknown"
            label = f"Unknown (GeO{cn})"

        return {
            "type": env_type,
            "label": label,
            "cn": cn,
            "neighbors": {"O": o_neigh},
        }

    def classify_all_ge(self) -> Dict[int, Dict]:
        """Classify all Ge sites in the structure."""
        out = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "Ge":
                out[i] = self.classify_ge_site(i)
        return out

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        """Calculate statistics of Ge environments."""
        counts = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1

        total_ge = len(classifications)
        fractions = {}
        if total_ge > 0:
            for k, v in counts.items():
                fractions[k] = 100.0 * v / total_ge

        return {
            "counts": dict(counts),
            "fractions": fractions,
            "total_ge": total_ge,
        }


class GeO2ConstraintWriter:
    """
    Writes v6-format constraints and environment summaries for Ge sites.
    
    v6 Format Features:
        - Uses "environment" key (not "environment_type")
        - Includes "environment_priorities" section
        - Compatible with EnvironmentConstrainedOptimizer
    
    Note: Tetrahedral order parameter targets are slightly lower for GeO2
    than SiO2 due to the larger Ge-O bond length allowing more flexibility.
    """

    # Order parameters for each Ge environment
    ENVIRONMENT_ORDER_PARAMETERS = {
        "Ge4": {  # Tetrahedral - ideal glass structure
            "order_parameters": {
                "tet": {
                    "target": 0.80,  # Slightly lower than Si due to larger radius
                    "min": 0.65,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around Ge (GeO4)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "Ge-O coordination number",
                },
            },
            "element_filter": [8, 32],  # O=8, Ge=32
            "cutoff": 2.4,
        },
        "Ge3": {  # Undercoordinated - defect
            "order_parameters": {
                "cn": {
                    "target": 3.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Ge-O coordination (undercoordinated)",
                },
            },
            "element_filter": [8, 32],
            "cutoff": 2.4,
        },
        "Ge5": {  # Five-coordinate - intermediate
            "order_parameters": {
                "cn": {
                    "target": 5.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Ge-O coordination (five-coordinate)",
                },
            },
            "element_filter": [8, 32],
            "cutoff": 2.4,
        },
        "Ge6": {  # Octahedral - high pressure (rutile-type GeO2)
            "order_parameters": {
                "cn": {
                    "target": 6.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "Ge-O coordination (octahedral)",
                },
            },
            "element_filter": [8, 32],
            "cutoff": 2.4,
        },
    }

    # v6: Environment priorities for adaptive penalty weighting
    ENVIRONMENT_PRIORITIES = {
        "Ge4": 2.0,   # Tetrahedral - most important for glass, strict geometry
        "Ge3": 1.0,   # Undercoordinated - defect, less strict
        "Ge5": 1.2,   # Five-coordinate - sometimes found in glasses
        "Ge6": 1.5,   # Octahedral - if present, maintain (densified glass)
    }

    def __init__(
        self, 
        structure: Structure, 
        classifier: GeEnvironmentClassifier,
        include_environments: List[str] = None,
    ):
        """
        Args:
            structure: Pymatgen Structure object
            classifier: GeEnvironmentClassifier instance
            include_environments: List of environments to include (e.g., ['Ge4']).
                                  If None, includes all known environments.
                                  Options: 'Ge4', 'Ge3', 'Ge5', 'Ge6'
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
        
        v6 Format:
            - "environment" key per atom (not "environment_type")
            - "environment_priorities" section for adaptive penalties
            - "version": "v6" in metadata
            
        Only environments in self.include_environments are included.
        """
        constraints = {
            "cutoff": self.classifier.ge_o_cutoff,
            "element_filter": [8, 32],  # O=8, Ge=32
            "atom_constraints": {},
            "environment_priorities": {},  # v6: for adaptive penalties
        }

        # Track which environments are present
        present_envs = set()

        for ge_idx, data in classifications.items():
            env_type = data["type"]
            
            # Skip environments not in include list
            if env_type not in self.include_environments:
                continue
            
            # Skip unknown environments
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue

            present_envs.add(env_type)
            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]

            # v6 FORMAT: Use "environment" key (not "environment_type")
            atom_constraint = {
                "atom_index": ge_idx,
                "element": "Ge",
                "environment": env_type,  # v6 key name
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(env_params["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(ge_idx)] = atom_constraint

        # v6: Add environment priorities for present environments only
        for env_type in present_envs:
            constraints["environment_priorities"][env_type] = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)

        # Global constraints
        constraints["global_constraints"] = {
            "description": "Per-Ge order parameter constraints for GeO2 glass",
            "total_ge_atoms": stats["total_ge"],
            "environment_fractions": stats["fractions"],
            "included_environments": list(self.include_environments),
        }

        # Metadata
        constraints["metadata"] = {
            "version": "v6",  # Mark as v6 format
            "structure_type": "geo2",
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
                "Ge4": "Tetrahedral GeO4 (quartz-like)",
                "Ge3": "Undercoordinated GeO3 (defect)",
                "Ge5": "Five-coordinate GeO5 (intermediate)",
                "Ge6": "Octahedral GeO6 (rutile-like, high pressure)",
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
        print(f"  - {len(constraints['atom_constraints'])} Ge atoms with constraints")
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
                    "o_neighbors": [int(i) for i in v["neighbors"]["O"]],
                }
                for k, v in classifications.items()
            },
        }
        env_file = f"{output_prefix}_Ge_environments.json"
        with open(env_file, "w") as f:
            json.dump(env_json, f, indent=2)
        print(f"Wrote environment data to: {env_file}")

        # 3) Human-readable text summary
        summary_file = f"{output_prefix}_Ge_environments.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Ge ENVIRONMENT SUMMARY FOR GeO2\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total Ge atoms: {stats['total_ge']}\n")
            f.write(f"Cutoff (Ge-O): {self.classifier.ge_o_cutoff:.3f} Å\n\n")

            f.write("Environment distribution (by Ge CN):\n")
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
            f.write("DETAILED Ge SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")
            
            # Group by type
            by_type = defaultdict(list)
            for ge_idx, data in classifications.items():
                by_type[data["type"]].append((ge_idx, data))
            
            for env_type in ["Ge4", "Ge3", "Ge5", "Ge6", "Ge_unknown"]:
                if env_type in by_type:
                    sites = by_type[env_type]
                    f.write(f"\n{env_type} sites ({len(sites)} atoms):\n")
                    f.write("-" * 70 + "\n")
                    for ge_idx, data in sites:
                        c = self.structure[ge_idx].coords
                        f.write(
                            f"  Index {ge_idx:6d}: {data['label']:24s} "
                            f"CN={data['cn']} "
                            f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                        )
                        f.write(f"      O neighbors: {data['neighbors']['O']}\n")

        print(f"Wrote summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="GeO2 environment + v6 constraint generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate constraints from crystalline GeO2 (all environments)
    python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --output geo2_glass

    # Only tetrahedral Ge4 (normal glass) - RECOMMENDED for most cases
    python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --environments Ge4 --output geo2_glass

    # Tetrahedral + octahedral (for densified/high-pressure glass)
    python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --environments Ge4 Ge6 --output geo2_densified

    # Use custom cutoff
    python -m torchdisorder.constraints.geo2_generator --input structure.cif --cutoff 2.5 --output geo2

Environment types:
    Ge4 - Tetrahedral GeO4 (CN=4) - Normal glass structure [DEFAULT]
    Ge3 - Undercoordinated GeO3 (CN=3) - Defect/surface site
    Ge5 - Five-coordinate GeO5 (CN=5) - Intermediate (more common than Si5)
    Ge6 - Octahedral GeO6 (CN=6) - High-pressure phase (rutile-like)

Output files:
    {output}_constraints.json      - v6-format constraints
    {output}_Ge_environments.json  - Machine-readable data
    {output}_Ge_environments.txt   - Human-readable summary

Notes:
    - GeO2 has longer Ge-O bonds (~1.73-1.88 Å) than Si-O (~1.60-1.62 Å)
    - Default cutoff is 2.4 Å (vs 2.2 Å for SiO2)
    - Ge shows higher tendency for 5/6-fold coordination under pressure
        """
    )
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="geo2", help="Output file prefix")
    parser.add_argument("--cutoff", type=float, default=2.4, help="Ge-O cutoff distance in Å (default: 2.4)")
    parser.add_argument(
        "--environments", 
        nargs="+", 
        default=None,
        choices=["Ge4", "Ge3", "Ge5", "Ge6"],
        help="Environment types to include (default: all). Use 'Ge4' for normal glass."
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("GeO2 Constraint Generator (v6 format)")
    print(f"{'=' * 70}")

    # Load structure
    print(f"\nLoading structure from: {args.input}")
    structure = Structure.from_file(args.input)
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Number of sites: {structure.num_sites}")

    # Classify Ge environments
    print(f"\nClassifying Ge environments (cutoff={args.cutoff} Å)...")
    classifier = GeEnvironmentClassifier(structure, ge_o_cutoff=args.cutoff)
    classifications = classifier.classify_all_ge()
    stats = classifier.get_statistics(classifications)

    print(f"  Total Ge atoms: {stats['total_ge']}")
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
    writer = GeO2ConstraintWriter(structure, classifier, include_environments=include_envs)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
