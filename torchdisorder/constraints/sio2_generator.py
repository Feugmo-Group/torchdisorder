"""
SiO2 Glass Structure Constraint Generator
==========================================

Generates v6-compatible constraint files for SiO2 glass structures.
Classifies Si environments by coordination number and outputs
constraints for environment-based optimization.

Features:
    - Supercell generation from CIF files
    - Environment classification (Si4, Si3, Si5, Si6)
    - v6-format constraint JSON output

Usage:
    # Generate constraints from CIF (no supercell)
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass --supercell 1000

    # Manual replication
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass --replicate 3 3 3

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
        n_rep = max(1, round(ratio ** (1/3)))
        
        # Try to find optimal [na, nb, nc]
        best_rep = [n_rep, n_rep, n_rep]
        best_diff = abs(n_unit * n_rep**3 - target_atoms)
        
        # Search nearby combinations
        for na in range(max(1, n_rep-2), n_rep+4):
            for nb in range(max(1, n_rep-2), n_rep+4):
                for nc in range(max(1, n_rep-2), n_rep+4):
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
    Classify Si environments in SiO2 by Si-O coordination number (CN).
    
    Environment types:
        - Si4: SiO4 tetrahedral (CN=4) - ideal glass former
        - Si3: SiO3 undercoordinated (CN=3) - defect
        - Si5: SiO5 overcoordinated (CN=5) - defect
        - Si6: SiO6 octahedral (CN=6) - high pressure phase
    """

    def __init__(self, structure: Structure, si_o_cutoff: float = 2.2):
        """
        Args:
            structure: Pymatgen Structure object
            si_o_cutoff: Si-O bond cutoff distance in Å (default 2.2 Å)
        """
        self.structure = structure
        self.si_o_cutoff = si_o_cutoff

    def get_o_neighbors_of_si(self, si_index: int) -> List[int]:
        """Get indices of O atoms within cutoff of Si atom."""
        si_site = self.structure[si_index]
        o_neighbors = []
        for j, other in enumerate(self.structure):
            if j == si_index:
                continue
            if other.specie.symbol != "O":
                continue
            if si_site.distance(other) <= self.si_o_cutoff:
                o_neighbors.append(j)
        return o_neighbors

    def classify_si_site(self, si_index: int) -> Dict:
        """
        Classify a single Si site by its coordination environment.
        
        Returns:
            dict with keys: 'type', 'label', 'cn', 'neighbors'
        """
        o_neigh = self.get_o_neighbors_of_si(si_index)
        cn = len(o_neigh)

        # Environment label by CN
        if cn == 4:
            env_type = "Si4"
            label = "SiO4 (tetrahedral)"
        elif cn == 3:
            env_type = "Si3"
            label = "SiO3 (undercoordinated)"
        elif cn == 5:
            env_type = "Si5"
            label = "SiO5 (overcoordinated)"
        elif cn == 6:
            env_type = "Si6"
            label = "SiO6 (octahedral)"
        else:
            env_type = "Si_unknown"
            label = f"Unknown (SiO{cn})"

        return {
            "type": env_type,
            "label": label,
            "cn": cn,
            "neighbors": {"O": o_neigh},
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


class SiO2ConstraintWriter:
    """
    Writes v6-format constraints and environment summaries for Si sites.
    
    v6 Format Features:
        - Uses "environment" key (not "environment_type")
        - Includes "environment_priorities" section
        - Compatible with EnvironmentConstrainedOptimizer
    """

    # Order parameters for each Si environment.
    #
    # These are defaults.  Prefer --measure-targets, which replaces them with values
    # taken from the input structure: the hard-coded tet target of 0.85 is already off
    # for a real glass (the published GAP model measures 0.918), and a target the
    # reference does not itself satisfy pulls the refinement away from physical.
    ENVIRONMENT_ORDER_PARAMETERS = {
        "Si4": {  # Tetrahedral - ideal glass structure
            "order_parameters": {
                "tet": {
                    "target": 0.85,
                    "min": 0.7,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around Si (SiO4)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "Si-O coordination number",
                },
                # F_IS is the parameter most sensitive to deformation of the
                # tetrahedron itself.  On refined structures it moved by up to 12x the
                # reference spread while q4 and q6 stayed inside their own noise, so
                # constraining cn and tet alone lets the polyhedra distort unnoticed.
                # -1/3 is exact for an ideal tetrahedron (Td has no inversion centre);
                # a real glass sits a little above it (GAP a-SiO2: -0.300).
                "fis": {
                    "target": -0.3333,
                    "tolerance": 0.08,
                    "weight": 1.5,
                    "description": "Local inversion symmetry of the SiO4 unit "
                                   "(Milkus & Zaccone, PRB 93, 094204 (2016))",
                },
            },
            "element_filter": [8],  # O only: Si-Si is 3.08 A, far outside the cutoff
            "cutoff": 2.2,
        },
        "Si3": {  # Undercoordinated - defect
            "order_parameters": {
                "cn": {
                    "target": 3.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Si-O coordination (undercoordinated)",
                },
            },
            "element_filter": [8, 14],
            "cutoff": 2.2,
        },
        "Si5": {  # Overcoordinated - defect
            "order_parameters": {
                "cn": {
                    "target": 5.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Si-O coordination (overcoordinated)",
                },
            },
            "element_filter": [8, 14],
            "cutoff": 2.2,
        },
        "Si6": {  # Octahedral - high pressure
            "order_parameters": {
                "cn": {
                    "target": 6.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Si-O coordination (octahedral)",
                },
                # An octahedron HAS an inversion centre, so F_IS = +1 exactly -- the
                # opposite end of the range from the tetrahedral -1/3.  This is the
                # case that the old mean-of-ratios combination reported as +1/3.
                "fis": {
                    "target": 1.0,
                    "tolerance": 0.10,
                    "weight": 1.5,
                    "description": "Local inversion symmetry of the SiO6 unit "
                                   "(+1 for a centrosymmetric octahedron)",
                },
            },
            "element_filter": [8],
            "cutoff": 2.2,
        },
    }

    # v6: Environment priorities for adaptive penalty weighting
    ENVIRONMENT_PRIORITIES = {
        "Si4": 2.0,   # Tetrahedral - most important, strict geometry
        "Si3": 1.0,   # Undercoordinated - defect, less strict
        "Si5": 1.0,   # Overcoordinated - defect, less strict
        "Si6": 1.5,   # Octahedral - if present, maintain
    }

    def __init__(
        self, 
        structure: Structure, 
        classifier: SiEnvironmentClassifier,
        include_environments: List[str] = None,
        env_params: Dict = None,
    ):
        """
        Args:
            structure: Pymatgen Structure object
            classifier: SiEnvironmentClassifier instance
            include_environments: List of environments to include (e.g., ['Si4']).
                                  If None, includes all known environments.
                                  Options: 'Si4', 'Si3', 'Si5', 'Si6'
        """
        self.structure = structure
        self.classifier = classifier
        # Instance-level copy so --measure-targets can override the class defaults
        # without mutating them for every other writer in the process.
        self.env_params = env_params or self.ENVIRONMENT_ORDER_PARAMETERS
        
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
            "cutoff": self.classifier.si_o_cutoff,
            "element_filter": [8, 14],  # O=8, Si=14
            "atom_constraints": {},
            "environment_priorities": {},  # v6: for adaptive penalties
        }

        # Track which environments are present
        present_envs = set()

        for si_idx, data in classifications.items():
            env_type = data["type"]
            
            # Skip environments not in include list
            if env_type not in self.include_environments:
                continue
            
            # Skip unknown environments
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue

            present_envs.add(env_type)
            env_params = self.env_params[env_type]

            # v6 FORMAT: Use "environment" key (not "environment_type")
            atom_constraint = {
                "atom_index": si_idx,
                "element": "Si",
                "environment": env_type,  # v6 key name
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(env_params["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(si_idx)] = atom_constraint

        # v6: Add environment priorities for present environments only
        for env_type in present_envs:
            constraints["environment_priorities"][env_type] = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)

        # Global constraints
        constraints["global_constraints"] = {
            "description": "Per-Si order parameter constraints for SiO2 glass",
            "total_si_atoms": stats["total_si"],
            "environment_fractions": stats["fractions"],
            "included_environments": list(self.include_environments),
        }

        # Metadata
        constraints["metadata"] = {
            "version": "v6",  # Mark as v6 format
            "structure_type": "sio2",
            "total_atoms": self.structure.num_sites,
            "composition": str(self.structure.composition),
            "included_environments": list(self.include_environments),
            "order_parameter_types": list(set(
                op for env_type in self.include_environments
                if env_type in self.ENVIRONMENT_ORDER_PARAMETERS
                for op in self.env_params[env_type]["order_parameters"].keys()
            )),
            "notes": "v6 constraints for EnvironmentConstrainedOptimizer with adaptive penalties",
            "environment_types": {
                "Si4": "Tetrahedral SiO4",
                "Si3": "Undercoordinated SiO3 (defect)",
                "Si5": "Overcoordinated SiO5 (defect)",
                "Si6": "Octahedral SiO6 (high pressure)",
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
                    "o_neighbors": [int(i) for i in v["neighbors"]["O"]],
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
            f.write("Si ENVIRONMENT SUMMARY FOR SiO2\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total Si atoms: {stats['total_si']}\n")
            f.write(f"Cutoff (Si-O): {self.classifier.si_o_cutoff:.3f} Å\n\n")

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
                    for op_name, op_params in self.env_params[env_type]["order_parameters"].items():
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
            
            for env_type in ["Si4", "Si3", "Si5", "Si6", "Si_unknown"]:
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
                        f.write(f"      O neighbors: {data['neighbors']['O']}\n")

        print(f"Wrote summary to: {summary_file}")


def measure_targets_from_structure(structure, classifications, env_params,
                                   cutoff, central_z=14):
    """Replace hard-coded OP targets with values measured on this structure.

    A target the reference structure does not itself satisfy pulls the refinement
    away from physical: the shipped tet target of 0.85 sits 0.07 below what a
    published a-SiO2 model actually measures, so the optimizer is rewarded for
    flattening tetrahedra that were already correct.  Measuring instead ties every
    target to a structure you have independently validated.

    Returns the env_params dict with 'target' fields overwritten, plus a report.
    """
    import copy

    import torch
    from ase import Atoms
    from torch_sim.io import atoms_to_state

    from torchdisorder.engine.order_params import PyTorchOrderParameters

    atoms = Atoms(
        numbers=[site.specie.Z for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )
    state = atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float64)

    out = copy.deepcopy(env_params)
    report = {}
    by_env = {}
    for idx, data in classifications.items():
        by_env.setdefault(data["type"], []).append(int(idx))

    for env_type, indices in by_env.items():
        if env_type not in out:
            continue
        ops = sorted(out[env_type]["order_parameters"])
        efilter = out[env_type].get("element_filter")
        calc = PyTorchOrderParameters(
            cutoff=out[env_type].get("cutoff", cutoff), device="cpu", max_neighbors=8
        )
        vals = calc(state, torch.tensor(indices), ops, element_filter=efilter)
        report[env_type] = {}
        for op in ops:
            v = vals[op].detach().cpu().numpy()
            mean, std = float(v.mean()), float(v.std())
            out[env_type]["order_parameters"][op]["target"] = round(mean, 4)
            out[env_type]["order_parameters"][op]["measured_std"] = round(std, 4)
            report[env_type][op] = (mean, std)
    return out, report


def main():
    parser = argparse.ArgumentParser(
        description="SiO2 environment + v6 constraint generator with supercell support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate constraints from crystalline SiO2 (no supercell)
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass

    # Generate supercell with ~1000 atoms
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass --supercell 1000

    # Generate supercell with ~2000 atoms
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_large --supercell 2000

    # Manual replication (3x3x3)
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_3x3x3 --replicate 3 3 3

    # Only tetrahedral Si4 (normal glass) - RECOMMENDED
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --environments Si4 --output sio2_glass

    # Tetrahedral + octahedral (for densified/high-pressure glass)
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --environments Si4 Si6 --output sio2_densified

Environment types:
    Si4  - Tetrahedral SiO4 (CN=4) - Normal glass structure [DEFAULT]
    Si3  - Undercoordinated SiO3 (CN=3) - Defect/surface site
    Si5  - Overcoordinated SiO5 (CN=5) - Transient/high-pressure
    Si6  - Octahedral SiO6 (CN=6) - High-pressure phase (stishovite-like)

Output files:
    {output}.cif                    - Structure file (supercell if requested)
    {output}_constraints.json       - v6-format constraints
    {output}_Si_environments.json   - Machine-readable data
    {output}_Si_environments.txt    - Human-readable summary
        """
    )
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="sio2", help="Output file prefix")
    parser.add_argument("--cutoff", type=float, default=2.2, help="Si-O cutoff distance in Å (default: 2.2)")
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
        choices=["Si4", "Si3", "Si5", "Si6"],
        help="Environment types to include (default: all). Use 'Si4' for normal glass."
    )
    parser.add_argument(
        "--measure-targets",
        action="store_true",
        help="Measure order-parameter targets from the input structure instead of "
             "using the built-in defaults. Recommended when the input is a structure "
             "you have independently validated.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("SiO2 Constraint Generator (v6 format)")
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
    classifier = SiEnvironmentClassifier(structure, si_o_cutoff=args.cutoff)
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
    env_params = SiO2ConstraintWriter.ENVIRONMENT_ORDER_PARAMETERS
    if args.measure_targets:
        print("\nMeasuring order-parameter targets from the input structure...")
        env_params, report = measure_targets_from_structure(
            structure, classifications, env_params, args.cutoff)
        for env, ops in sorted(report.items()):
            for op, (mean, std) in sorted(ops.items()):
                print(f"    {env:6s} {op:4s} target = {mean:+.4f}  (spread {std:.4f})")

    writer = SiO2ConstraintWriter(structure, classifier,
                                  include_environments=include_envs,
                                  env_params=env_params)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
