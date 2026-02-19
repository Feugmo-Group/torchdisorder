import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

from pymatgen.core import Structure


class SiEnvironmentClassifier:
    """
    Classify Si environments in SiO2 by Si-O coordination number (CN).
    Primary target: SiO4 tetrahedra (CN=4).
    """

    def __init__(self, structure: Structure, si_o_cutoff: float = 2.2):
        self.structure = structure
        self.si_o_cutoff = si_o_cutoff

    def get_o_neighbors_of_si(self, si_index: int) -> List[int]:
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
        o_neigh = self.get_o_neighbors_of_si(si_index)
        cn = len(o_neigh)

        # Environment label by CN (extend if you want)
        if cn == 4:
            env_type = "Si4"
            label = "SiO4 (tetrahedral)"
        elif cn == 3:
            env_type = "Si3"
            label = "SiO3 (undercoordinated)"
        elif cn == 5:
            env_type = "Si5"
            label = "SiO5 (overcoordinated)"
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
        out = {}
        for i, site in enumerate(self.structure):
            if site.specie.symbol == "Si":
                out[i] = self.classify_si_site(i)
        return out

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
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
    Writes constraints and environment summaries for Si sites.
    Only enforces tetrahedral ('tet') order parameter constraint on Si.
    """

    # Keep only the "tet" constraint as requested
    ENVIRONMENT_ORDER_PARAMETERS = {
        "Si4": {
            "order_parameters": {
                "tet": {
                    "target": 0.7,
                    "min": 0.6,
                    "max": 1.0,
                    "weight": 2.0,
                    "description": "Tetrahedrality around Si (SiO4)",
                }
            },
            "element_filter": [8, 14],  # O=8, Si=14
            "cutoff": 2.2,
        }
    }

    def __init__(self, structure: Structure, classifier: SiEnvironmentClassifier):
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
            "cutoff": self.classifier.si_o_cutoff,
            "element_filter": [8, 14],
            "atom_constraints": {},
            "global_constraints": {
                "description": "Per-Si tetrahedral order parameter constraints for SiO2",
                "total_si_atoms": stats["total_si"],
                "environment_fractions": stats["fractions"],
            },
            "metadata": {
                "structure_type": "sio2",
                "total_atoms": self.structure.num_sites,
                "composition": str(self.structure.composition),
                "order_parameter_types": ["tet"],
                "notes": "Constraints intended for augmented Lagrangian / TorchSimOrderParameters-style workflows",
            },
        }

        for si_idx, data in classifications.items():
            env_type = data["type"]
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue  # only constrain tetrahedral Si

            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]
            atom_constraint = {
                "atom_index": si_idx,
                "element": "Si",
                "environment_type": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "order_parameters": dict(env_params["order_parameters"]),
                "o_neighbor_indices": data["neighbors"]["O"],
                "cn": data["cn"],
            }
            constraints["atom_constraints"][str(si_idx)] = atom_constraint

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
        with open(f"{output_prefix}_Si_environments.json", "w") as f:
            json.dump(env_json, f, indent=2)

        # 3) human-readable text summary
        with open(f"{output_prefix}_Si_environments.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Si ENVIRONMENT SUMMARY FOR SiO2\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total Si atoms: {stats['total_si']}\n")
            f.write(f"Cutoff (Si-O): {self.classifier.si_o_cutoff:.3f} Å\n\n")

            f.write("Environment distribution (by Si CN):\n")
            f.write("-" * 70 + "\n")
            for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
                f.write(f"  {env_type:12s}: {frac:6.2f}%  (count={stats['counts'][env_type]})\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED Si SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")
            for si_idx, data in classifications.items():
                c = self.structure[si_idx].coords
                f.write(
                    f"Index {si_idx:6d}: {data['label']:24s} "
                    f"CN={data['cn']} "
                    f"xyz=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\n"
                )
                f.write(f"    O neighbors: {data['neighbors']['O']}\n")


def main():
    parser = argparse.ArgumentParser(description="SiO2 environment + constraint generator (no structure modification).")
    parser.add_argument("--input", required=True, help="Input structure file (CIF/POSCAR/etc.)")
    parser.add_argument("--output", default="sio2", help="Output prefix")
    parser.add_argument("--cutoff", type=float, default=2.2, help="Si-O cutoff distance (Å)")
    args = parser.parse_args()

    structure = Structure.from_file(args.input)

    classifier = SiEnvironmentClassifier(structure, si_o_cutoff=args.cutoff)
    classifications = classifier.classify_all_si()
    stats = classifier.get_statistics(classifications)

    writer = SiO2ConstraintWriter(structure, classifier)
    writer.write_outputs(args.output, classifications, stats)


if __name__ == "__main__":
    main()
