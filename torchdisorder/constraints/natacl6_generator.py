"""
NaTaCl6 Halide Constraint Generator
=====================================

Generates v6-compatible constraint files for NaTaCl6-type halide structures.
Classifies Ta (and optionally Na) environments by coordination number.

NaTaCl6 is an elpasolite-related compound where isolated TaCl6 octahedra
are linked by Na counter-ions.  It is a model system for halide perovskite
physics and radiation detection.  The key structural unit is the TaCl6²⁻
octahedron (CN=6, Ta-Cl ~2.29–2.49 Å).

Bond distances observed in the CIF (amorphous supercell, P1):
    Ta-Cl: 2.29, 2.32, 2.32, 2.43, 2.44, 2.49 Å  →  CN=6, cutoff 2.6 Å
    Na-Cl: 2.85, 2.88, 2.89, 2.97, 2.99, 3.04 Å  →  CN=6, cutoff 3.2 Å

Environment types:
    Ta6  — TaCl6 octahedral (CN=6) — dominant and expected in glass
    Ta5  — TaCl5 five-coord  (CN=5) — defect / under-coordinated
    Ta7  — TaCl7 seven-coord (CN=7) — over-coordinated distortion
    Na6  — NaCl6 octahedral  (CN=6) — Na counter-ion environment
    Na8  — NaCl8 cubic       (CN=8) — Na in more open cavity

Usage:
    python -m torchdisorder.constraints.natacl6_generator \\
        --input NaTaCl6.cif --output natacl6_glass

    # Only Ta (ignore Na sites)
    python -m torchdisorder.constraints.natacl6_generator \\
        --input NaTaCl6.cif --central Ta --output natacl6_ta_only

Output files:
    {output}_constraints.json      — v6-format constraints
    {output}_Ta_environments.json  — machine-readable data
    {output}_Ta_environments.txt   — human-readable summary
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


def create_supercell(
    structure: Structure,
    target_atoms: Optional[int] = None,
    replicate: Optional[List[int]] = None,
) -> Tuple[Structure, List[int]]:
    n_unit = len(structure)
    if target_atoms is not None:
        ratio = target_atoms / n_unit
        n_rep = max(1, round(ratio ** (1 / 3)))
        best_rep = [n_rep, n_rep, n_rep]
        best_diff = abs(n_unit * n_rep**3 - target_atoms)
        for na in range(max(1, n_rep - 2), n_rep + 4):
            for nb in range(max(1, n_rep - 2), n_rep + 4):
                for nc in range(max(1, n_rep - 2), n_rep + 4):
                    diff = abs(n_unit * na * nb * nc - target_atoms)
                    if diff < best_diff:
                        best_diff = diff
                        best_rep = [na, nb, nc]
        replicate = best_rep
        print(f"  Auto replication: {replicate} → {n_unit * int(np.prod(replicate))} atoms")
    elif replicate is None:
        replicate = [1, 1, 1]
    if replicate != [1, 1, 1]:
        structure = structure.copy()
        structure.make_supercell(replicate)
    return structure, replicate


class TaEnvironmentClassifier:
    """Classify Ta environments in NaTaCl6 by Ta-Cl coordination number.

    Ta-Cl cutoff guide:
        2.6 Å — default; clean gap between CN=6 bonds (≤2.49) and next shell (≥4.87)
        2.8 Å — permissive; use for heavily distorted glass structures
    """

    def __init__(self, structure: Structure, ta_cl_cutoff: float = 2.6):
        self.structure = structure
        self.ta_cl_cutoff = ta_cl_cutoff

    def get_cl_neighbors(self, ta_index: int) -> List[int]:
        ta_site = self.structure[ta_index]
        return [
            j for j, other in enumerate(self.structure)
            if j != ta_index and other.specie.symbol == "Cl"
            and ta_site.distance(other) <= self.ta_cl_cutoff
        ]

    def classify_ta_site(self, ta_index: int) -> Dict:
        cl_neigh = self.get_cl_neighbors(ta_index)
        cn = len(cl_neigh)
        if cn == 6:
            env_type, label = "Ta6", "TaCl6 (octahedral)"
        elif cn == 5:
            env_type, label = "Ta5", "TaCl5 (five-coordinate, defect)"
        elif cn == 7:
            env_type, label = "Ta7", "TaCl7 (seven-coordinate, over-coordinated)"
        else:
            env_type, label = "Ta_unknown", f"Unknown (TaCl{cn})"
        return {"type": env_type, "label": label, "cn": cn, "neighbors": {"Cl": cl_neigh}}

    def classify_all_ta(self) -> Dict[int, Dict]:
        return {
            i: self.classify_ta_site(i)
            for i, site in enumerate(self.structure)
            if site.specie.symbol == "Ta"
        }

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        counts: Dict[str, int] = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1
        total = len(classifications)
        return {
            "counts": dict(counts),
            "fractions": {k: 100.0 * v / total for k, v in counts.items()} if total else {},
            "total_ta": total,
        }


class NaTaCl6ConstraintWriter:
    """Write v6-format constraints for NaTaCl6-type structures.

    The primary structural motif is the TaCl6²⁻ octahedron.  Octahedral
    distortion (measured via the q4 bond-order parameter and F_IS) is the
    key link between local structure and optical / electronic properties.

    Atomic numbers: Ta=73, Cl=17, Na=11
    """

    ENVIRONMENT_ORDER_PARAMETERS = {
        "Ta6": {
            "order_parameters": {
                "cn": {
                    "target": 6.0,
                    "tolerance": 0.5,
                    "weight": 2.0,
                    "description": "Ta-Cl coordination number (octahedral)",
                },
                "q4": {
                    "target": 0.764,
                    "min": 0.55,
                    "max": 1.0,
                    "weight": 1.5,
                    "description": "Cubic/octahedral bond-angle order (TaCl6)",
                },
            },
            "element_filter": [17, 73],
            "cutoff": 2.6,
        },
        "Ta5": {
            "order_parameters": {
                "cn": {
                    "target": 5.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Ta-Cl coordination (five-coordinate, defect)",
                },
            },
            "element_filter": [17, 73],
            "cutoff": 2.6,
        },
        "Ta7": {
            "order_parameters": {
                "cn": {
                    "target": 7.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Ta-Cl coordination (seven-coordinate, over-coordinated)",
                },
            },
            "element_filter": [17, 73],
            "cutoff": 2.6,
        },
    }

    ENVIRONMENT_PRIORITIES = {
        "Ta6": 2.5,
        "Ta5": 1.0,
        "Ta7": 1.0,
    }

    def __init__(
        self,
        structure: Structure,
        classifier: TaEnvironmentClassifier,
        include_environments: Optional[List[str]] = None,
    ):
        self.structure = structure
        self.classifier = classifier
        valid = set(self.ENVIRONMENT_ORDER_PARAMETERS)
        if include_environments is None:
            self.include_environments = list(valid)
        else:
            for env in include_environments:
                if env not in valid:
                    raise ValueError(f"Unknown environment '{env}'. Valid: {valid}")
            self.include_environments = include_environments

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_constraints(self, classifications: Dict[int, Dict], stats: Dict) -> Dict:
        constraints: Dict = {
            "cutoff": self.classifier.ta_cl_cutoff,
            "element_filter": [17, 73],
            "atom_constraints": {},
            "environment_priorities": {},
        }
        present_envs: set = set()
        for ta_idx, data in classifications.items():
            env_type = data["type"]
            if env_type not in self.include_environments:
                continue
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue
            present_envs.add(env_type)
            ep = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]
            constraints["atom_constraints"][str(ta_idx)] = {
                "atom_index": ta_idx,
                "element": "Ta",
                "environment": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(ep["order_parameters"]),
                "cl_neighbor_indices": data["neighbors"]["Cl"],
                "cn": data["cn"],
            }
        for env in present_envs:
            constraints["environment_priorities"][env] = self.ENVIRONMENT_PRIORITIES.get(env, 1.0)
        constraints["global_constraints"] = {
            "description": "Per-Ta order parameter constraints for NaTaCl6 glass",
            "total_ta_atoms": stats["total_ta"],
            "environment_fractions": stats["fractions"],
            "included_environments": self.include_environments,
        }
        constraints["metadata"] = {
            "version": "v6",
            "structure_type": "natacl6",
            "total_atoms": self.structure.num_sites,
            "composition": str(self.structure.composition),
            "included_environments": self.include_environments,
            "order_parameter_types": list({
                op
                for env in self.include_environments
                if env in self.ENVIRONMENT_ORDER_PARAMETERS
                for op in self.ENVIRONMENT_ORDER_PARAMETERS[env]["order_parameters"]
            }),
            "notes": (
                "v6 constraints for EnvironmentConstrainedOptimizer. "
                "Primary motif: TaCl6 octahedra (Ta-Cl cutoff 2.6 Å). "
                "q4 target 0.764 = ideal octahedral value. "
                "F_IS expected to be highly sensitive to octahedral tilting."
            ),
            "environment_types": {
                "Ta6": "Octahedral TaCl6 (ideal, CN=6)",
                "Ta5": "Five-coordinate TaCl5 (defect)",
                "Ta7": "Seven-coordinate TaCl7 (over-coordinated)",
            },
        }
        return self._to_jsonable(constraints)

    def write_outputs(self, output_prefix: str, classifications: Dict[int, Dict], stats: Dict):
        cif_file = f"{output_prefix}.cif"
        CifWriter(self.structure).write_file(cif_file)
        print(f"Wrote structure to: {cif_file}")

        constraints = self.generate_constraints(classifications, stats)
        with open(f"{output_prefix}_constraints.json", "w") as f:
            json.dump(constraints, f, indent=2)
        print(f"Wrote v6 constraints: {len(constraints['atom_constraints'])} Ta atoms")

        env_json = {
            "statistics": self._to_jsonable(stats),
            "classifications": {
                str(k): {
                    "type": v["type"],
                    "label": v["label"],
                    "cn": int(v["cn"]),
                    "coords": [float(x) for x in self.structure[k].coords],
                    "cl_neighbors": [int(i) for i in v["neighbors"]["Cl"]],
                }
                for k, v in classifications.items()
            },
        }
        with open(f"{output_prefix}_Ta_environments.json", "w") as f:
            json.dump(env_json, f, indent=2)

        with open(f"{output_prefix}_Ta_environments.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Ta ENVIRONMENT SUMMARY FOR NaTaCl6\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms : {self.structure.num_sites}\n")
            f.write(f"Total Ta    : {stats['total_ta']}\n")
            f.write(f"Cutoff Ta-Cl: {self.classifier.ta_cl_cutoff:.3f} Å\n\n")
            f.write("Environment distribution:\n")
            f.write("-" * 70 + "\n")
            for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
                pri = self.ENVIRONMENT_PRIORITIES.get(env_type, 1.0)
                f.write(
                    f"  {env_type:12s}: {frac:6.2f}%  "
                    f"(count={stats['counts'][env_type]}, priority={pri})\n"
                )
            f.write("\n")
            by_type: Dict = defaultdict(list)
            for ta_idx, data in classifications.items():
                by_type[data["type"]].append((ta_idx, data))
            for env_type in ["Ta6", "Ta5", "Ta7", "Ta_unknown"]:
                if env_type not in by_type:
                    continue
                sites = by_type[env_type]
                f.write(f"\n{env_type} sites ({len(sites)} atoms):\n")
                f.write("-" * 70 + "\n")
                for ta_idx, data in sites[:20]:
                    c = self.structure[ta_idx].coords
                    f.write(
                        f"  Index {ta_idx:6d}: {data['label']:36s} "
                        f"CN={data['cn']} "
                        f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                    )
                if len(sites) > 20:
                    f.write(f"  ... ({len(sites) - 20} more)\n")
        print(f"Wrote summary: {output_prefix}_Ta_environments.txt")


def main():
    parser = argparse.ArgumentParser(
        description="NaTaCl6 environment + v6 constraint generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (all Ta environments)
    python -m torchdisorder.constraints.natacl6_generator \\
        --input NaTaCl6.cif --output natacl6_glass

    # Only octahedral Ta6
    python -m torchdisorder.constraints.natacl6_generator \\
        --input NaTaCl6.cif --environments Ta6 --output natacl6_oct

    # Supercell ~1000 atoms
    python -m torchdisorder.constraints.natacl6_generator \\
        --input NaTaCl6.cif --supercell 1000 --output natacl6_large

Environment types:
    Ta6 — TaCl6 octahedral  (CN=6) — dominant in glass  [DEFAULT]
    Ta5 — TaCl5 five-coord  (CN=5) — defect
    Ta7 — TaCl7 seven-coord (CN=7) — over-coordinated

Cutoff guide (Ta-Cl):
    2.6 Å — default; clean gap between bonds (≤2.49) and next shell (≥4.87)
    2.8 Å — permissive for heavily distorted glass
        """,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="natacl6_glass")
    parser.add_argument("--cutoff", type=float, default=2.6, help="Ta-Cl cutoff in Å (default: 2.6)")
    parser.add_argument("--supercell", type=int, default=None, metavar="N")
    parser.add_argument("--replicate", type=int, nargs=3, default=None, metavar=("NA", "NB", "NC"))
    parser.add_argument("--environments", nargs="+", default=None, choices=["Ta6", "Ta5", "Ta7"])
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("NaTaCl6 Constraint Generator (v6 format)")
    print(f"{'=' * 70}\n")

    structure = Structure.from_file(args.input)
    print(f"Loaded: {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

    if args.supercell is not None or args.replicate is not None:
        structure, rep = create_supercell(structure, args.supercell, args.replicate)
        print(f"Supercell: {rep} → {structure.num_sites} atoms")

    classifier = TaEnvironmentClassifier(structure, ta_cl_cutoff=args.cutoff)
    classifications = classifier.classify_all_ta()
    stats = classifier.get_statistics(classifications)

    print(f"\nTa environment distribution (cutoff={args.cutoff} Å):")
    for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
        print(f"  {env_type}: {frac:.1f}% ({stats['counts'][env_type]} atoms)")

    writer = NaTaCl6ConstraintWriter(structure, classifier, args.environments)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
