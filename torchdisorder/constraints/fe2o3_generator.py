"""
Fe2O3 Glass Structure Constraint Generator
===========================================

Generates v6-compatible constraint files for iron oxide (Fe2O3) glass structures.
Classifies Fe environments by coordination number and outputs constraints for
environment-based optimization.

ε-Fe₂O₃ is a unique multiferroic: it contains both tetrahedral (Fe1, CN=4)
and octahedral (Fe2/3/4, CN=6) iron sites.  The tetrahedral site is absent
in α/γ-Fe₂O₃ and is responsible for the spontaneous polarisation.
F_IS is expected to differ sharply between these two local environments,
making Fe₂O₃ an excellent test case for the inversion-symmetry metric.

Environment types:
    Fe4  — FeO₄ tetrahedral  (CN=4) — unique to ε-Fe₂O₃, priority=3.0
    Fe6  — FeO₆ octahedral   (CN=6) — dominant in all Fe₂O₃ polymorphs
    Fe5  — FeO₅ five-coord   (CN=5) — defect or disordered glass site

Fe-O bond lengths:
    Tetrahedral (Fe1): ~1.87 Å   → cutoff 2.05 Å separates tet from oct
    Octahedral  (Fe2-4): ~1.97–2.10 Å → cutoff 2.2 Å captures all 6 bonds

Usage:
    python -m torchdisorder.constraints.fe2o3_generator \\
        --input epsilon_Fe2O3.cif --output fe2o3_glass

    # Only octahedral (when tetrahedral site is absent)
    python -m torchdisorder.constraints.fe2o3_generator \\
        --input my_fe2o3.cif --cutoff 2.3 --environments Fe6 --output fe2o3_oct

Output files:
    {output}_constraints.json      — v6-format constraints
    {output}_Fe_environments.json  — machine-readable environment data
    {output}_Fe_environments.txt   — human-readable summary
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from torchdisorder.constraints.fingerprint import atom_order_fingerprint


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
        print(f"  Auto-computed replication: {replicate} → {n_unit * int(np.prod(replicate))} atoms")
    elif replicate is None:
        replicate = [1, 1, 1]
    if replicate != [1, 1, 1]:
        structure = structure.copy()
        structure.make_supercell(replicate)
    return structure, replicate


class FeEnvironmentClassifier:
    """Classify Fe environments in iron oxide by Fe-O coordination number.

    Fe-O cutoff guide:
        2.05 Å — captures only tetrahedral Fe1 bonds (~1.87 Å)
        2.2  Å — standard; captures all tet + oct bonds up to 2.10 Å
        2.4  Å — permissive; use when structure has elongated oct bonds
    """

    def __init__(self, structure: Structure, fe_o_cutoff: float = 2.2):
        self.structure = structure
        self.fe_o_cutoff = fe_o_cutoff

    def get_o_neighbors(self, fe_index: int) -> List[int]:
        fe_site = self.structure[fe_index]
        return [
            j for j, other in enumerate(self.structure)
            if j != fe_index and other.specie.symbol == "O"
            and fe_site.distance(other) <= self.fe_o_cutoff
        ]

    def classify_fe_site(self, fe_index: int) -> Dict:
        o_neigh = self.get_o_neighbors(fe_index)
        cn = len(o_neigh)
        if cn == 4:
            env_type, label = "Fe4", "FeO4 (tetrahedral, ε-Fe₂O₃ Fe1 site)"
        elif cn == 6:
            env_type, label = "Fe6", "FeO6 (octahedral)"
        elif cn == 5:
            env_type, label = "Fe5", "FeO5 (five-coordinate, defect)"
        else:
            env_type, label = "Fe_unknown", f"Unknown (FeO{cn})"
        return {"type": env_type, "label": label, "cn": cn, "neighbors": {"O": o_neigh}}

    def classify_all_fe(self) -> Dict[int, Dict]:
        return {
            i: self.classify_fe_site(i)
            for i, site in enumerate(self.structure)
            if site.specie.symbol == "Fe"
        }

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        counts: Dict[str, int] = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1
        total = len(classifications)
        return {
            "counts": dict(counts),
            "fractions": {k: 100.0 * v / total for k, v in counts.items()} if total else {},
            "total_fe": total,
        }


class Fe2O3ConstraintWriter:
    """Write v6-format constraints for Fe₂O₃ structures.

    Reference: Ohkoshi et al. Nature Chem. 2010 (ε-Fe₂O₃ multiferroic);
               Milkus & Zaccone PRB 2016 (F_IS).
    """

    ENVIRONMENT_ORDER_PARAMETERS = {
        "Fe4": {
            "order_parameters": {
                "tet": {
                    "target": 0.78,
                    "min": 0.60,
                    "max": 1.0,
                    "weight": 2.5,
                    "description": "Tetrahedrality of FeO4 (ε-Fe2O3 Fe1 site)",
                },
                "cn": {
                    "target": 4.0,
                    "tolerance": 0.5,
                    "weight": 2.0,
                    "description": "Fe-O coordination number (tetrahedral)",
                },
            },
            "element_filter": [8, 26],
            "cutoff": 2.2,
        },
        "Fe6": {
            "order_parameters": {
                "cn": {
                    "target": 6.0,
                    "tolerance": 0.5,
                    "weight": 1.5,
                    "description": "Fe-O coordination number (octahedral)",
                },
            },
            "element_filter": [8, 26],
            "cutoff": 2.2,
        },
        "Fe5": {
            "order_parameters": {
                "cn": {
                    "target": 5.0,
                    "tolerance": 0.5,
                    "weight": 1.0,
                    "description": "Fe-O coordination number (five-coordinate, defect)",
                },
            },
            "element_filter": [8, 26],
            "cutoff": 2.2,
        },
    }

    ENVIRONMENT_PRIORITIES = {
        "Fe4": 3.0,
        "Fe6": 2.0,
        "Fe5": 1.0,
    }

    def __init__(
        self,
        structure: Structure,
        classifier: FeEnvironmentClassifier,
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
            "cutoff": self.classifier.fe_o_cutoff,
            "element_filter": [8, 26],
            "atom_constraints": {},
            "environment_priorities": {},
        }
        present_envs: set = set()
        for fe_idx, data in classifications.items():
            env_type = data["type"]
            if env_type not in self.include_environments:
                continue
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue
            present_envs.add(env_type)
            ep = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]
            constraints["atom_constraints"][str(fe_idx)] = {
                "atom_index": fe_idx,
                "element": "Fe",
                "environment": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(ep["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "cn": data["cn"],
            }
        for env in present_envs:
            constraints["environment_priorities"][env] = self.ENVIRONMENT_PRIORITIES.get(env, 1.0)
        constraints["global_constraints"] = {
            "description": "Per-Fe order parameter constraints for Fe2O3 glass",
            "total_fe_atoms": stats["total_fe"],
            "environment_fractions": stats["fractions"],
            "included_environments": self.include_environments,
        }
        constraints["metadata"] = {
            "version": "v6",
            "atom_order": atom_order_fingerprint(
                self.structure, constraints["atom_constraints"].keys()
            ),
            "structure_type": "fe2o3",
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
                "Fe4 = tetrahedral (unique to ε-Fe2O3, multiferroic Fe1 site). "
                "Fe6 = octahedral (dominant in all polymorphs). "
                "F_IS distinguishes these environments experimentally."
            ),
            "environment_types": {
                "Fe4": "Tetrahedral FeO4 (ε-Fe2O3 Fe1 site, multiferroic)",
                "Fe6": "Octahedral FeO6 (Fe2/3/4 sites)",
                "Fe5": "Five-coordinate FeO5 (defect / disordered glass)",
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
        print(f"Wrote v6 constraints: {len(constraints['atom_constraints'])} Fe atoms")

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
        with open(f"{output_prefix}_Fe_environments.json", "w") as f:
            json.dump(env_json, f, indent=2)

        with open(f"{output_prefix}_Fe_environments.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Fe ENVIRONMENT SUMMARY FOR Fe2O3\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms : {self.structure.num_sites}\n")
            f.write(f"Total Fe    : {stats['total_fe']}\n")
            f.write(f"Cutoff Fe-O : {self.classifier.fe_o_cutoff:.3f} Å\n\n")
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
            for fe_idx, data in classifications.items():
                by_type[data["type"]].append((fe_idx, data))
            for env_type in ["Fe4", "Fe5", "Fe6", "Fe_unknown"]:
                if env_type not in by_type:
                    continue
                sites = by_type[env_type]
                f.write(f"\n{env_type} sites ({len(sites)} atoms):\n")
                f.write("-" * 70 + "\n")
                for fe_idx, data in sites[:20]:
                    c = self.structure[fe_idx].coords
                    f.write(
                        f"  Index {fe_idx:6d}: {data['label']:42s} "
                        f"CN={data['cn']} "
                        f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                    )
                if len(sites) > 20:
                    f.write(f"  ... ({len(sites) - 20} more)\n")
        print(f"Wrote summary: {output_prefix}_Fe_environments.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Fe2O3 environment + v6 constraint generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (mixed tet+oct environments)
    python -m torchdisorder.constraints.fe2o3_generator \\
        --input epsilon_Fe2O3_iso_1.cif --output fe2o3_glass

    # Only octahedral (when CIF has no tetrahedral Fe)
    python -m torchdisorder.constraints.fe2o3_generator \\
        --input my_fe2o3.cif --environments Fe6 --output fe2o3_oct

    # Supercell
    python -m torchdisorder.constraints.fe2o3_generator \\
        --input epsilon_Fe2O3.cif --supercell 1000 --output fe2o3_large

Environment types:
    Fe4 — FeO4 tetrahedral  (CN=4, ε-Fe2O3 Fe1, multiferroic site)
    Fe6 — FeO6 octahedral   (CN=6, dominant in all polymorphs)
    Fe5 — FeO5 five-coord   (CN=5, defect / disordered glass)

Cutoff guide:
    2.05 Å — tetrahedral bonds only (~1.87 Å)
    2.20 Å — default; captures all oct bonds up to 2.10 Å  [DEFAULT]
    2.40 Å — permissive; use for elongated bonds or glass structures
        """,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="fe2o3_glass")
    parser.add_argument("--cutoff", type=float, default=2.2, help="Fe-O cutoff in Å (default: 2.2)")
    parser.add_argument("--supercell", type=int, default=None, metavar="N")
    parser.add_argument("--replicate", type=int, nargs=3, default=None, metavar=("NA", "NB", "NC"))
    parser.add_argument("--environments", nargs="+", default=None, choices=["Fe4", "Fe5", "Fe6"])
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("Fe2O3 Constraint Generator (v6 format)")
    print(f"{'=' * 70}\n")

    structure = Structure.from_file(args.input)
    print(f"Loaded: {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

    if args.supercell is not None or args.replicate is not None:
        structure, rep = create_supercell(structure, args.supercell, args.replicate)
        print(f"Supercell: {rep} → {structure.num_sites} atoms")

    classifier = FeEnvironmentClassifier(structure, fe_o_cutoff=args.cutoff)
    classifications = classifier.classify_all_fe()
    stats = classifier.get_statistics(classifications)

    print(f"\nFe environment distribution (cutoff={args.cutoff} Å):")
    for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
        print(f"  {env_type}: {frac:.1f}% ({stats['counts'][env_type]} atoms)")

    writer = Fe2O3ConstraintWriter(structure, classifier, args.environments)
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
