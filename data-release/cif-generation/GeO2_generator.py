import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

from pymatgen.core import Structure


class GeEnvironmentClassifier:
    """
    Classify Ge environments in GeO2 by Ge-O coordination number (CN).
    Primary target: GeO4 tetrahedra (CN=4).
    """

    def __init__(self, structure: Structure, ge_o_cutoff: float = 2.1):
        self.structure = structure
        self.ge_o_cutoff = ge_o_cutoff

    def get_o_neighbors_of_ge(self, ge_index: int) -> List[int]:
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
            label = "GeO5 (overcoordinated)"
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
        out = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "Ge":
                out[i] = self.classify_ge_site(i)
        return out

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
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
    Writes constraints and environment summaries for Ge sites.
    Only enforces tetrahedral ('tet') order parameter constraint on Ge.
    """

    # Tetrahedral constraint for GeO4
    ENVIRONMENT_ORDER_PARAMETERS = {
        "Ge4": {
            "order_parameters": {
                "tet": {
                    "target": 0.7,
                    "min": 0.6,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around Ge (GeO4)",
                }
            },
            "element_filter": [8, 32],  # O=8, Ge=32
            "cutoff": 2.1,
        }
    }

    def __init__(self, structure: Structure, classifier: GeEnvironmentClassifier):
        self.structure = structure
        self.classifier = classifier

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def generate_constraints(self, classifications: Dict[int, Dict], stats: Dict) -> Dict:
        constraints = {
            "cutoff": self.classifier.ge_o_cutoff,
            "element_filter": [8, 32],  # O=8, Ge=32
            "atom_constraints": {},
            "global_constraints": {
                "description": "Per-Ge tetrahedral order parameter constraints for GeO2",
                "total_ge_atoms": stats["total_ge"],
                "environment_fractions": stats["fractions"],
            },
            "metadata": {
                "structure_type": "geo2",
                "total_atoms": self.structure.num_sites,
                "composition": str(self.structure.composition),
                "order_parameter_types": ["tet"],
                "notes": "Constraints intended for augmented Lagrangian / TorchSimOrderParameters-style workflows",
            },
        }

        for ge_idx, data in classifications.items():
            env_type = data["type"]
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue  # only constrain tetrahedral Ge

            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]
            atom_constraint = {
                "atom_index": ge_idx,
                "element": "Ge",
                "environment_type": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(env_params["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(ge_idx)] = atom_constraint

        return self._to_jsonable(constraints)

    def write_outputs(self, output_prefix: str, classifications: Dict[int, Dict], stats: Dict):
        # 1) constraints json
        constraints = self.generate_constraints(classifications, stats)
        with open(f"{output_prefix}_constraints.json", "w") as f:
            json.dump(constraints, f, indent=2)

        # 2) machine-readable environments json
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
        with open(f"{output_prefix}_Ge_environments.json", "w") as f:
            json.dump(env_json, f, indent=2)

        # 3) human-readable text summary
        with open(f"{output_prefix}_Ge_environments.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Ge ENVIRONMENT SUMMARY FOR GeO2\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total Ge atoms: {stats['total_ge']}\n")
            f.write(f"Cutoff (Ge-O): {self.classifier.ge_o_cutoff:.3f} Å\n\n")

            f.write("Environment distribution (by Ge CN):\n")
            f.write("-" * 70 + "\n")
            for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
                f.write(f"  {env_type:12s}: {frac:6.2f}%  (count={stats['counts'][env_type]})\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED Ge SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")
            for ge_idx, data in classifications.items():
                c = self.structure[ge_idx].coords
                f.write(
                    f"Index {ge_idx:6d}: {data['label']:24s} "
                    f"CN={data['cn']} "
                    f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                )
                f.write(f"    O neighbors: {data['neighbors']['O']}\n")


def main():
    parser = argparse.ArgumentParser(description="GeO2 environment + constraint generator (no structure modification).")
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="geo2", help="Output prefix")
    parser.add_argument("--cutoff", type=float, default=2.1, help="Ge-O cutoff distance (Å)")
    args = parser.parse_args()

    structure = Structure.from_file(args.input)

    classifier = GeEnvironmentClassifier(structure, ge_o_cutoff=args.cutoff)
    classifications = classifier.classify_all_ge()
    stats = classifier.get_statistics(classifications)

    writer = GeO2ConstraintWriter(structure, classifier)
    writer.write_outputs(args.output, classifications, stats)


if __name__ == "__main__":
    main()
