"""
Li₂HfCl₆₋ₓFₓ Halide Constraint Generator
==========================================

Generates v6-compatible constraint files for Li₂HfCl₆ and its
fluorine-substituted variants Li₂HfCl₆₋ₓFₓ (x = 0, 1, 2, 3).

Crystal chemistry
-----------------
Li₂HfCl₆ is a hexagonal layered halide electrolyte (P-3m1, #164).
Hf⁴⁺ sits in an isolated HfCl₆²⁻ octahedron; Li⁺ occupies two
crystallographically distinct octahedral sites in the van-der-Waals gaps.
In the amorphous phase the octahedral motif is retained but with a
distribution of CN=5 (defect), CN=6 (dominant) and CN=7 (over-coordinated).

When F⁻ partially replaces Cl⁻ the HfX₆ octahedron becomes mixed-anion
(HfCl₆₋ₙFₙ, n=0..6).  F⁻ is smaller (1.33 Å) and more electronegative
than Cl⁻ (1.81 Å), so it preferentially shortens one or two Hf-X bonds and
breaks local inversion symmetry — detectable via F_IS analysis.

Bond distances (from literature + DFT):
    Hf-Cl: 2.48–2.56 Å  →  cutoff 2.70 Å (default)
    Hf-F : 1.96–2.04 Å  →  cutoff 2.20 Å (default)
    Li-Cl: 2.47–2.88 Å  →  cutoff 3.00 Å
    Li-F : 1.85–2.10 Å  →  included in Li-Cl cutoff 3.00 Å

Environment types
-----------------
Pure chloride (x=0):
    Hf6   — HfCl₆   octahedral, CN=6           (dominant)
    Hf5   — HfCl₅   five-coordinate, defect
    Hf7   — HfCl₇   seven-coordinate, over-coordinated

Mixed-anion (x>0), classified by number of F neighbours within Hf-F cutoff:
    Hf6F0 — HfCl₆   (same as Hf6, n_F=0)
    Hf6F1 — HfCl₅F  (n_F=1)
    Hf6F2 — HfCl₄F₂ (n_F=2)
    Hf6F3 — HfCl₃F₃ (n_F=3, mer or fac)
    Hf6F4 — HfCl₂F₄ (n_F=4)
    Hf6F5 — HfClF₅  (n_F=5)
    Hf6F6 — HfF₆    (n_F=6)

F_IS scientific context
-----------------------
Pure HfCl₆ octahedra: F_IS ≈ +1.0 (high inversion symmetry).
Each Cl→F substitution distorts bond lengths and breaks the O_h symmetry
→ F_IS decreases monotonically with n_F.
Warren-Cowley SRO α(Hf,F) < 0 indicates F prefers Hf neighbours (ordered
substitution); α ≈ 0 indicates random mixing.

Usage
-----
    # Li₂HfCl₆ baseline (no F)
    python -m torchdisorder.constraints.li2hfcl6_generator \\
        --input Li2HfCl6.cif --output li2hfcl6_glass

    # Li₂HfCl₄F₂  (x=2)
    python -m torchdisorder.constraints.li2hfcl6_generator \\
        --input Li2HfCl4F2.cif --output li2hfcl4f2_glass --mode mixed_anion

    # Supercell ~800 atoms
    python -m torchdisorder.constraints.li2hfcl6_generator \\
        --input Li2HfCl6.cif --supercell 800 --output li2hfcl6_sc800

Output files
------------
    {output}_constraints.json       — v6-format constraints
    {output}_Hf_environments.json   — machine-readable classification data
    {output}_Hf_environments.txt    — human-readable summary

Atomic numbers: Hf=72, Cl=17, F=9, Li=3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


# ---------------------------------------------------------------------------
# Supercell helper (shared pattern across all generators)
# ---------------------------------------------------------------------------

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
        best_diff = abs(n_unit * n_rep ** 3 - target_atoms)
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


# ---------------------------------------------------------------------------
# Environment classifier
# ---------------------------------------------------------------------------

class HfEnvironmentClassifier:
    """Classify Hf environments in Li₂HfCl₆₋ₓFₓ by CN and F-count.

    Two distance cutoffs are used:
        hf_halide_cutoff — distance within which ALL halide neighbours (Cl + F)
                           are counted for the total CN.  Default 2.70 Å.
        hf_f_cutoff      — stricter cutoff for counting just F neighbours.
                           Default 2.20 Å (Hf-F bonds are ~1.96–2.04 Å).

    The combined cutoff must satisfy:
        hf_f_cutoff < hf_halide_cutoff   (F bonds are always shorter than Cl bonds)
    """

    def __init__(
        self,
        structure: Structure,
        hf_halide_cutoff: float = 2.70,
        hf_f_cutoff: float = 2.20,
        mode: str = "auto",
    ):
        """
        Args:
            structure:          Pymatgen Structure.
            hf_halide_cutoff:   Max Hf–(Cl or F) distance for total CN (Å).
            hf_f_cutoff:        Max Hf–F distance for F-neighbour count (Å).
            mode:               'pure_cl'      — treat all halogens as Cl (ignore F).
                                'mixed_anion'  — subclassify by n_F.
                                'auto'         — detect from composition.
        """
        self.structure = structure
        self.hf_halide_cutoff = hf_halide_cutoff
        self.hf_f_cutoff = hf_f_cutoff

        if mode == "auto":
            has_f = any(s.specie.symbol == "F" for s in structure)
            self.mode = "mixed_anion" if has_f else "pure_cl"
        else:
            self.mode = mode

        self.has_fluorine = any(s.specie.symbol == "F" for s in structure)

    def _get_halide_neighbors(self, hf_idx: int) -> Tuple[List[int], List[int]]:
        """Return (cl_indices, f_indices) within their respective cutoffs."""
        site = self.structure[hf_idx]
        cl_neigh, f_neigh = [], []
        for j, other in enumerate(self.structure):
            if j == hf_idx:
                continue
            sym = other.specie.symbol
            d = site.distance(other)
            if sym == "Cl" and d <= self.hf_halide_cutoff:
                cl_neigh.append(j)
            elif sym == "F" and d <= self.hf_halide_cutoff:
                # Count F in total CN using halide cutoff (Hf-F is shorter, so always inside)
                f_neigh.append(j)
        return cl_neigh, f_neigh

    def classify_hf_site(self, hf_idx: int) -> Dict:
        cl_neigh, f_neigh = self._get_halide_neighbors(hf_idx)
        cn_cl = len(cl_neigh)
        cn_f = len(f_neigh)
        cn_total = cn_cl + cn_f

        if self.mode == "mixed_anion" and self.has_fluorine:
            # Subclassify CN=6 by F count
            if cn_total == 6:
                env_type = f"Hf6F{cn_f}"
                label = f"HfCl{cn_cl}F{cn_f} (octahedral, CN=6, {cn_f} F)"
            elif cn_total == 5:
                env_type = "Hf5"
                label = f"HfX5 (five-coordinate defect, {cn_f} F)"
            elif cn_total == 7:
                env_type = "Hf7"
                label = f"HfX7 (seven-coordinate, {cn_f} F)"
            else:
                env_type = "Hf_unknown"
                label = f"Unknown HfX{cn_total}"
        else:
            # Pure-Cl mode or no F present
            if cn_total == 6:
                env_type, label = "Hf6", "HfCl6 (octahedral, CN=6)"
            elif cn_total == 5:
                env_type, label = "Hf5", "HfCl5 (five-coordinate, defect)"
            elif cn_total == 7:
                env_type, label = "Hf7", "HfCl7 (seven-coordinate, over-coordinated)"
            else:
                env_type, label = "Hf_unknown", f"Unknown HfX{cn_total}"

        return {
            "type": env_type,
            "label": label,
            "cn": cn_total,
            "cn_cl": cn_cl,
            "cn_f": cn_f,
            "neighbors": {"Cl": cl_neigh, "F": f_neigh},
        }

    def classify_all_hf(self) -> Dict[int, Dict]:
        return {
            i: self.classify_hf_site(i)
            for i, site in enumerate(self.structure)
            if site.specie.symbol == "Hf"
        }

    def get_statistics(self, classifications: Dict[int, Dict]) -> Dict:
        counts: Dict[str, int] = defaultdict(int)
        for d in classifications.values():
            counts[d["type"]] += 1
        total = len(classifications)
        return {
            "counts": dict(counts),
            "fractions": {k: 100.0 * v / total for k, v in counts.items()} if total else {},
            "total_hf": total,
            "mode": self.mode,
            "has_fluorine": self.has_fluorine,
        }


# ---------------------------------------------------------------------------
# Order parameter specs per environment type
# ---------------------------------------------------------------------------

def _make_env_op_specs(env_type: str, cn_f: int = 0) -> Dict:
    """Return order-parameter constraints for a given Hf environment.

    q4 target:
        Ideal OH octahedron: q4 = 0.764
        Mixed HfCl₆₋ₙFₙ: same target but wider tolerance as n_F increases,
        because Hf-F bond contraction distorts the octahedron slightly.
    """
    # q4 tolerance widens with F substitution (each F adds ~0.02 Å bond asymmetry)
    q4_tol_extra = min(cn_f * 0.03, 0.15)

    if env_type in ("Hf6",) or env_type.startswith("Hf6F"):
        return {
            "cn": {
                "target": 6.0,
                "tolerance": 0.5,
                "weight": 2.0,
                "description": "Hf total halide coordination number (octahedral)",
            },
            "q4": {
                "target": 0.764,
                "min": max(0.40, 0.55 - q4_tol_extra),
                "max": 1.0,
                "weight": 1.5,
                "description": "Octahedral bond-angle order (q4=0.764 ideal O_h)",
            },
        }
    elif env_type == "Hf5":
        return {
            "cn": {
                "target": 5.0,
                "tolerance": 0.5,
                "weight": 1.0,
                "description": "Hf five-coordinate (defect)",
            },
        }
    elif env_type == "Hf7":
        return {
            "cn": {
                "target": 7.0,
                "tolerance": 0.5,
                "weight": 1.0,
                "description": "Hf seven-coordinate (over-coordinated)",
            },
        }
    return {}


def _env_priority(env_type: str, cn_f: int = 0) -> float:
    """Priority weights for environment-grouped Cooper constraints.

    HfCl₆ (x=0 reference): highest priority.
    Mixed-anion environments: priority tracks scientific interest.
        - n_F=0 and n_F=6 are symmetric endpoints → 2.5
        - n_F=1..5 are the interesting asymmetric environments → 2.0–1.8
    Defect environments (Hf5, Hf7): 1.0.
    """
    if env_type in ("Hf6", "Hf6F0", "Hf6F6"):
        return 2.5
    if env_type.startswith("Hf6F"):
        # n_F=1..5 → linear interpolation 2.0 → 1.8
        return round(2.0 - 0.04 * (cn_f - 1), 2)
    return 1.0  # Hf5, Hf7, unknown


# ---------------------------------------------------------------------------
# Constraint writer
# ---------------------------------------------------------------------------

class Li2HfCl6ConstraintWriter:
    """Write v6-format constraints for Li₂HfCl₆₋ₓFₓ structures.

    The primary scientific motif is the HfX₆ octahedron.  Constraints enforce:
      - Total halide CN = 6 (with tolerance 0.5)
      - Octahedral bond-order q4 ≥ 0.55 (wider for mixed-anion environments)

    The F-subclassified environments (Hf6F0..Hf6F6) allow the optimizer to
    treat Cl-rich and F-rich Hf sites differently, with environment_priorities
    tracking the scientific relevance of each mixed-anion motif.
    """

    def __init__(
        self,
        structure: Structure,
        classifier: HfEnvironmentClassifier,
        include_environments: Optional[List[str]] = None,
    ):
        self.structure = structure
        self.classifier = classifier
        if include_environments is None:
            self.include_environments = None  # include all
        else:
            self.include_environments = set(include_environments)

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_constraints(
        self, classifications: Dict[int, Dict], stats: Dict
    ) -> Dict:
        # Element filter depends on whether F is present
        if self.classifier.has_fluorine:
            element_filter = [9, 17, 72]   # F=9, Cl=17, Hf=72
        else:
            element_filter = [17, 72]       # Cl=17, Hf=72

        constraints: Dict = {
            "cutoff": self.classifier.hf_halide_cutoff,
            "hf_f_cutoff": self.classifier.hf_f_cutoff,
            "element_filter": element_filter,
            "atom_constraints": {},
            "environment_priorities": {},
        }

        present_envs: set = set()

        for hf_idx, data in classifications.items():
            env_type = data["type"]

            if self.include_environments is not None and env_type not in self.include_environments:
                continue
            if env_type == "Hf_unknown":
                continue

            present_envs.add(env_type)
            cn_f = data["cn_f"]
            op_specs = _make_env_op_specs(env_type, cn_f)

            constraints["atom_constraints"][str(hf_idx)] = {
                "atom_index": hf_idx,
                "element": "Hf",
                "environment": env_type,
                "environment_label": data["label"],
                "target_coordination": data["cn"],
                "cn_cl": data["cn_cl"],
                "cn_f": data["cn_f"],
                "order_parameters": op_specs,
                "cl_neighbor_indices": data["neighbors"]["Cl"],
                "f_neighbor_indices": data["neighbors"]["F"],
                "cn": data["cn"],
            }

        for env in present_envs:
            # Extract cn_f from env_type name if mixed-anion
            cn_f = 0
            if env.startswith("Hf6F") and env != "Hf6F":
                try:
                    cn_f = int(env[4:])
                except ValueError:
                    pass
            constraints["environment_priorities"][env] = _env_priority(env, cn_f)

        # Composition string for metadata
        comp = str(self.structure.composition.reduced_formula)

        constraints["global_constraints"] = {
            "description": f"Per-Hf constraints for Li₂HfCl₆₋ₓFₓ glass ({comp})",
            "total_hf_atoms": stats["total_hf"],
            "environment_fractions": stats["fractions"],
            "has_fluorine": stats["has_fluorine"],
            "mode": stats["mode"],
        }

        constraints["metadata"] = {
            "version": "v6",
            "structure_type": "li2hfcl6",
            "composition": comp,
            "total_atoms": self.structure.num_sites,
            "has_fluorine": self.classifier.has_fluorine,
            "mode": self.classifier.mode,
            "hf_halide_cutoff_ang": self.classifier.hf_halide_cutoff,
            "hf_f_cutoff_ang": self.classifier.hf_f_cutoff,
            "order_parameter_types": ["cn", "q4"],
            "environment_types": {
                "Hf6":   "HfCl₆ octahedral (CN=6, pure Cl)",
                "Hf6F0": "HfCl₆ octahedral (CN=6, 0 F) — mixed-anion mode",
                "Hf6F1": "HfCl₅F  (CN=6, 1 F)",
                "Hf6F2": "HfCl₄F₂ (CN=6, 2 F)",
                "Hf6F3": "HfCl₃F₃ (CN=6, 3 F — mer or fac)",
                "Hf6F4": "HfCl₂F₄ (CN=6, 4 F)",
                "Hf6F5": "HfClF₅  (CN=6, 5 F)",
                "Hf6F6": "HfF₆    (CN=6, pure F)",
                "Hf5":   "HfX₅ five-coordinate (defect)",
                "Hf7":   "HfX₇ seven-coordinate (over-coordinated)",
            },
            "notes": (
                "v6 constraints for EnvironmentConstrainedOptimizer. "
                "Primary motif: HfX₆ octahedra (X = Cl and/or F). "
                "q4 target 0.764 = ideal O_h value; tolerance widens with n_F "
                "because Hf-F bond contraction (~1.98 Å vs Hf-Cl ~2.53 Å) "
                "distorts the octahedron. "
                "Use Warren-Cowley SRO α(Hf,F) to test random vs ordered F substitution. "
                "Use fis_spatial_autocorrelation to measure symmetry correlation length vs x."
            ),
        }
        return self._to_jsonable(constraints)

    def write_outputs(
        self,
        output_prefix: str,
        classifications: Dict[int, Dict],
        stats: Dict,
    ) -> None:
        # Write supercell CIF
        cif_path = f"{output_prefix}.cif"
        CifWriter(self.structure).write_file(cif_path)
        print(f"  Wrote structure : {cif_path}")

        # Write v6 constraints JSON
        constraints = self.generate_constraints(classifications, stats)
        json_path = f"{output_prefix}_constraints.json"
        with open(json_path, "w") as fh:
            json.dump(constraints, fh, indent=2)
        n_constrained = len(constraints["atom_constraints"])
        print(f"  Wrote constraints: {json_path}  ({n_constrained} Hf atoms)")

        # Machine-readable environment JSON
        env_json = {
            "statistics": self._to_jsonable(stats),
            "classifications": {
                str(k): {
                    "type": v["type"],
                    "label": v["label"],
                    "cn": int(v["cn"]),
                    "cn_cl": int(v["cn_cl"]),
                    "cn_f": int(v["cn_f"]),
                    "coords": [float(x) for x in self.structure[k].coords],
                    "cl_neighbors": [int(i) for i in v["neighbors"]["Cl"]],
                    "f_neighbors": [int(i) for i in v["neighbors"]["F"]],
                }
                for k, v in classifications.items()
            },
        }
        env_json_path = f"{output_prefix}_Hf_environments.json"
        with open(env_json_path, "w") as fh:
            json.dump(env_json, fh, indent=2)
        print(f"  Wrote env data  : {env_json_path}")

        # Human-readable summary
        txt_path = f"{output_prefix}_Hf_environments.txt"
        with open(txt_path, "w") as fh:
            fh.write("=" * 70 + "\n")
            fh.write(f"Hf ENVIRONMENT SUMMARY — {self.structure.composition.reduced_formula}\n")
            fh.write("=" * 70 + "\n\n")
            fh.write(f"Total atoms : {self.structure.num_sites}\n")
            fh.write(f"Total Hf    : {stats['total_hf']}\n")
            fh.write(f"Mode        : {stats['mode']}\n")
            fh.write(f"Has F       : {stats['has_fluorine']}\n")
            fh.write(f"Hf-halide cutoff : {self.classifier.hf_halide_cutoff:.3f} Å\n")
            fh.write(f"Hf-F cutoff      : {self.classifier.hf_f_cutoff:.3f} Å\n\n")

            fh.write("Environment distribution:\n")
            fh.write("-" * 70 + "\n")
            for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
                cnt = stats["counts"][env_type]
                pri = _env_priority(env_type)
                fh.write(
                    f"  {env_type:12s}: {frac:6.2f}%"
                    f"  (n={cnt:4d}, priority={pri:.1f})\n"
                )

            fh.write("\n")
            by_type: Dict = defaultdict(list)
            for hf_idx, data in classifications.items():
                by_type[data["type"]].append((hf_idx, data))

            # Print in canonical order
            canonical = [
                "Hf6", "Hf6F0", "Hf6F1", "Hf6F2", "Hf6F3",
                "Hf6F4", "Hf6F5", "Hf6F6", "Hf5", "Hf7", "Hf_unknown",
            ]
            for env_type in canonical:
                if env_type not in by_type:
                    continue
                sites = by_type[env_type]
                fh.write(f"\n{env_type} ({len(sites)} atoms):\n")
                fh.write("-" * 70 + "\n")
                for hf_idx, data in sites[:20]:
                    c = self.structure[hf_idx].coords
                    fh.write(
                        f"  [{hf_idx:6d}] {data['label']:46s}"
                        f" xyz=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})\n"
                    )
                if len(sites) > 20:
                    fh.write(f"  ... ({len(sites) - 20} more)\n")

        print(f"  Wrote summary   : {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Li₂HfCl₆₋ₓFₓ environment + v6 constraint generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Li₂HfCl₆ baseline (no fluorine)
  python -m torchdisorder.constraints.li2hfcl6_generator \\
      --input Li2HfCl6.cif --output li2hfcl6_glass

  # Li₂HfCl₄F₂  (x=2, mixed-anion mode)
  python -m torchdisorder.constraints.li2hfcl6_generator \\
      --input Li2HfCl4F2.cif --mode mixed_anion --output li2hfcl4f2_glass

  # Supercell ~800 atoms
  python -m torchdisorder.constraints.li2hfcl6_generator \\
      --input Li2HfCl6.cif --supercell 800 --output li2hfcl6_sc800

  # Custom cutoffs (distorted amorphous glass)
  python -m torchdisorder.constraints.li2hfcl6_generator \\
      --input Li2HfCl6.cif --cutoff 2.80 --f_cutoff 2.25 --output li2hfcl6_loose

Cutoff guide (Hf-halide)
------------------------
  2.70 Å — default; clean gap between 1st shell (≤2.56) and 2nd shell (≥4.9)
  2.80 Å — permissive for heavily distorted amorphous structures
  2.20 Å — default Hf-F cutoff (Hf-F bonds are 1.96–2.04 Å)

Environment modes
-----------------
  auto         — detect from composition (default)
  pure_cl      — ignore F, treat all halogens as Cl
  mixed_anion  — subclassify by n_F: Hf6F0, Hf6F1, ..., Hf6F6
        """,
    )
    parser.add_argument("--input", required=True, help="Path to CIF file")
    parser.add_argument("--output", default="li2hfcl6_glass", help="Output prefix")
    parser.add_argument(
        "--cutoff", type=float, default=2.70,
        help="Hf–(Cl or F) first-shell cutoff in Å (default: 2.70)",
    )
    parser.add_argument(
        "--f_cutoff", type=float, default=2.20,
        help="Hf–F specific cutoff in Å (default: 2.20)",
    )
    parser.add_argument(
        "--mode", choices=["auto", "pure_cl", "mixed_anion"], default="auto",
        help="Classification mode (default: auto-detect from composition)",
    )
    parser.add_argument(
        "--supercell", type=int, default=None, metavar="N",
        help="Target supercell size in atoms",
    )
    parser.add_argument(
        "--replicate", type=int, nargs=3, default=None, metavar=("NA", "NB", "NC"),
        help="Explicit supercell replication (e.g. --replicate 2 2 3)",
    )
    parser.add_argument(
        "--environments", nargs="+", default=None,
        help="Restrict to specific environment types (e.g. --environments Hf6 Hf6F1)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("Li₂HfCl₆₋ₓFₓ Constraint Generator  (v6 format)")
    print(f"{'=' * 70}\n")

    structure = Structure.from_file(args.input)
    print(f"Loaded  : {structure.composition.reduced_formula}  ({structure.num_sites} atoms)")
    print(f"Species : {sorted(set(s.specie.symbol for s in structure))}")

    if args.supercell is not None or args.replicate is not None:
        structure, rep = create_supercell(structure, args.supercell, args.replicate)
        print(f"Supercell → {rep}  ({structure.num_sites} atoms)")

    classifier = HfEnvironmentClassifier(
        structure,
        hf_halide_cutoff=args.cutoff,
        hf_f_cutoff=args.f_cutoff,
        mode=args.mode,
    )
    print(f"\nClassification mode : {classifier.mode}")
    print(f"Hf-halide cutoff    : {args.cutoff} Å")
    if classifier.has_fluorine:
        print(f"Hf-F cutoff         : {args.f_cutoff} Å")

    classifications = classifier.classify_all_hf()
    stats = classifier.get_statistics(classifications)

    print(f"\nHf environment distribution:")
    print(f"  {'Environment':12s}  {'%':>6}  {'Count':>6}  {'Priority':>8}")
    print(f"  {'-' * 40}")
    for env_type, frac in sorted(stats["fractions"].items(), key=lambda x: -x[1]):
        cnt = stats["counts"][env_type]
        cn_f = 0
        if env_type.startswith("Hf6F"):
            try:
                cn_f = int(env_type[4:])
            except ValueError:
                pass
        pri = _env_priority(env_type, cn_f)
        print(f"  {env_type:12s}  {frac:6.1f}  {cnt:6d}  {pri:8.1f}")

    writer = Li2HfCl6ConstraintWriter(structure, classifier, args.environments)
    print(f"\nWriting outputs with prefix: {args.output}")
    writer.write_outputs(args.output, classifications, stats)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}\n")
    print("Next steps for Syed:")
    print("  1. Add configs/data/Li2HfCl6.yaml  (copy NaTaCl6.yaml, update paths)")
    print("  2. Add configs/structure/Li2HfCl6.yaml  (point to the CIF + this JSON)")
    print("  3. python scripts/train.py data=Li2HfCl6 structure=Li2HfCl6")
    print("  4. python scripts/analyze_fis_properties.py  \\")
    print("         --run_dir outputs/Li2HfCl6_*/...  \\")
    print("         --system Li2HfCl6  --central Hf  --neighbor Cl  --cutoff 2.7  \\")
    print("         --central_z 72  --neighbor_z 17")
    print()


if __name__ == "__main__":
    main()
