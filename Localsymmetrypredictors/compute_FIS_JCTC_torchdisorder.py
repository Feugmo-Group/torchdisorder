#!/usr/bin/env python3
"""
Compute the inversion-symmetry order parameter F_IS for atomistic
TorchDisorder-style network-glass structures from the JCTC 2026 paper.

This version is tailored to systems such as:
  - SiO2: central Si, neighbors O, cutoff 2.2 Angstrom
  - GeO2: central Ge, neighbors O, cutoff 2.4 Angstrom
  - Li2S-P2S5: central P, neighbors S and optionally P, cutoffs P-S 3.5 A, P-P 2.8 A

It implements the variable-bond-length generalization appropriate for
multicomponent network glasses:

    Xi_i = sum_j R_ij n_ij (n_ij^mu n_ij^nu)

    F_IS = 1 - sum_i |Xi_i|^2 / sum_i sum_j R_ij^2 (n_ij^mu n_ij^nu)^2

The numerator and denominator are both DIRECTED neighbor sums, i.e. each
bond contributes once to each central node for which it is considered.

For comparison only, --mode milkus2016 omits the R_ij factors, corresponding
to the constant-R0 PRB-2016 limit where all springs share the same bond length.
For the JCTC systems, --mode variable_R is the recommended default.

Input formats:
  .xyz       standard or extended XYZ. Extended XYZ Lattice="..." is read.
  .csv       columns x,y,z plus optional element/species/type/atom
  .dat/.txt  numeric columns x y z; species are unavailable unless using XYZ/CSV

Examples:
  python compute_FIS_JCTC_torchdisorder.py --input sio2_refined.xyz --preset sio2
  python compute_FIS_JCTC_torchdisorder.py --input geo2_refined.xyz --preset geo2
  python compute_FIS_JCTC_torchdisorder.py --input lps_refined.xyz --preset lps_ps
  python compute_FIS_JCTC_torchdisorder.py --input lps_refined.xyz --preset lps_ps_pp

Outputs:
  <prefix>_summary.txt
  <prefix>_local_FIS.csv
  <prefix>_FIS_hist.png, if --plot is passed
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ----------------------------- presets -----------------------------

@dataclass
class Preset:
    central: str
    neighbors: str
    cutoff: float
    pair_cutoffs: Dict[str, float]
    note: str


PRESETS: Dict[str, Preset] = {
    "sio2": Preset(
        central="Si",
        neighbors="O",
        cutoff=2.2,
        pair_cutoffs={"Si-O": 2.2},
        note="Si-centered SiO4 tetrahedra; cutoff from TorchDisorder JCTC text.",
    ),
    "geo2": Preset(
        central="Ge",
        neighbors="O",
        cutoff=2.4,
        pair_cutoffs={"Ge-O": 2.4},
        note="Ge-centered GeO4 tetrahedra; cutoff from TorchDisorder JCTC text.",
    ),
    "lps_ps": Preset(
        central="P",
        neighbors="S",
        cutoff=3.5,
        pair_cutoffs={"P-S": 3.5},
        note="P-centered sulfur coordination in Li2S-P2S5; ignores P-P dumbbell bonds.",
    ),
    "lps_ps_pp": Preset(
        central="P",
        neighbors="S,P",
        cutoff=3.5,
        pair_cutoffs={"P-S": 3.5, "P-P": 2.8},
        note="P-centered Li2S-P2S5 environments including P-P dumbbell bonds.",
    ),
}


# ----------------------------- IO -----------------------------

def parse_cell_from_xyz_comment(comment: str) -> Optional[np.ndarray]:
    """Parse orthorhombic cell from common XYZ comment-line formats."""
    m = re.search(r"Lattice\s*=\s*\"([^\"]+)\"", comment)
    if m:
        vals = np.array([float(x) for x in m.group(1).split()], dtype=float)
        if vals.size == 9:
            mat = vals.reshape(3, 3)
            # Extended XYZ stores the three cell vectors. Use their norms.
            return np.linalg.norm(mat, axis=1)

    m = re.search(r"cell\s*=\s*\"?([0-9eE+\-., ]+)\"?", comment, flags=re.I)
    if m:
        vals = [float(x) for x in re.split(r"[, ]+", m.group(1).strip()) if x]
        if len(vals) == 1:
            return np.array([vals[0], vals[0], vals[0]], dtype=float)
        if len(vals) >= 3:
            return np.array(vals[:3], dtype=float)
    return None


def load_structure(path: str) -> Tuple[np.ndarray, Optional[List[str]], Optional[np.ndarray]]:
    """Return positions (N,3), species list if available, and cell if available."""
    ext = os.path.splitext(path)[1].lower()
    species = None
    cell = None

    if ext == ".xyz":
        with open(path, "r", encoding="utf-8") as fh:
            raw = [line.rstrip() for line in fh if line.strip()]
        n = int(raw[0].split()[0])
        comment = raw[1] if len(raw) > 1 else ""
        cell = parse_cell_from_xyz_comment(comment)
        species = []
        positions = []
        for line in raw[2:2 + n]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ line: {line}")
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(positions, dtype=float), species, cell

    if ext == ".csv":
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        sp_col = next((c for c in ["element", "species", "type", "atom"] if c in df.columns), None)
        if sp_col is not None:
            species = df[sp_col].astype(str).tolist()
        if {"x", "y", "z"}.issubset(df.columns):
            positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        else:
            nums = df.select_dtypes(include=[np.number])
            if nums.shape[1] < 3:
                raise ValueError("CSV needs x,y,z columns or at least three numeric columns for JCTC 3D systems.")
            positions = nums.iloc[:, :3].to_numpy(dtype=float)
        return positions, species, cell

    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError("DAT/TXT input needs at least three numeric columns x y z for JCTC 3D systems.")
    return np.asarray(arr[:, :3], dtype=float), species, cell


# ----------------------------- neighbors -----------------------------

def parse_set(s: Optional[str]) -> Optional[set]:
    if s is None or s.strip() == "":
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def parse_pair_cutoffs(s: Optional[str]) -> Dict[Tuple[str, str], float]:
    if not s:
        return {}
    raw = json.loads(s)
    out: Dict[Tuple[str, str], float] = {}
    for key, value in raw.items():
        a, b = [x.strip() for x in key.replace("_", "-").split("-")]
        out[(a, b)] = float(value)
        out[(b, a)] = float(value)
    return out


def minimum_image(r: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    if cell is None:
        return r
    return r - cell * np.round(r / cell)


def build_directed_cutoff_neighbors(
    points: np.ndarray,
    species: Optional[List[str]],
    cell: Optional[np.ndarray],
    cutoff: float,
    central_species: Optional[set],
    neighbor_species: Optional[set],
    pair_cutoffs: Dict[Tuple[str, str], float],
) -> List[List[Tuple[int, np.ndarray]]]:
    """Directed neighbor list with optional element filters and PBC."""
    n = len(points)
    neigh: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(n)]

    max_cutoff = max([cutoff] + list(pair_cutoffs.values()))

    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        wrapped = points - cell * np.floor(points / cell)
        tree = cKDTree(wrapped, boxsize=cell)
        cand = tree.query_ball_point(wrapped, r=max_cutoff)
        coords = wrapped
    else:
        tree = cKDTree(points)
        cand = tree.query_ball_point(points, r=max_cutoff)
        coords = points

    for i in range(n):
        si = species[i] if species is not None else None
        if central_species is not None and si not in central_species:
            continue
        for j in cand[i]:
            if i == j:
                continue
            sj = species[j] if species is not None else None
            if neighbor_species is not None and sj not in neighbor_species:
                continue

            cij = cutoff
            if species is not None and (si, sj) in pair_cutoffs:
                cij = pair_cutoffs[(si, sj)]

            rij_wrapped = minimum_image(coords[j] - coords[i], cell)
            R = float(np.linalg.norm(rij_wrapped))
            if 0.0 < R < cij:
                # shift applied to original points[j] to reproduce minimum-image vector
                shift = rij_wrapped - (points[j] - points[i])
                neigh[i].append((j, shift))
    return neigh


# ----------------------------- F_IS -----------------------------

def compute_fis(
    points: np.ndarray,
    neighbors: List[List[Tuple[int, np.ndarray]]],
    cell: Optional[np.ndarray],
    shear: str,
    mode: str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute global F_IS and local F_IS_i using directed neighbor sums."""
    comp = {"x": 0, "y": 1, "z": 2}
    mu, nu = comp[shear[0]], comp[shear[1]]

    n = len(points)
    Xi = np.zeros((n, 3), dtype=float)
    denom_i = np.zeros(n, dtype=float)

    for i in range(n):
        for j, shift in neighbors[i]:
            rij = points[j] + shift - points[i]
            rij = minimum_image(rij, cell)
            R = float(np.linalg.norm(rij))
            if R == 0.0:
                continue
            nhat = rij / R
            orient = nhat[mu] * nhat[nu]
            if mode == "variable_R":
                weight = R
            elif mode == "milkus2016":
                weight = 1.0
            else:
                raise ValueError("mode must be variable_R or milkus2016")

            Xi[i] += weight * orient * nhat
            denom_i[i] += (weight * orient) ** 2

    num_i = np.sum(Xi * Xi, axis=1)
    valid = denom_i > 0.0
    local = np.full(n, np.nan)
    local[valid] = 1.0 - num_i[valid] / denom_i[valid]
    if not np.any(valid):
        raise RuntimeError("No valid F_IS sites. Check species filters, cutoffs, and cell.")
    global_fis = 1.0 - np.sum(num_i[valid]) / np.sum(denom_i[valid])
    return float(global_fis), local, Xi, num_i, denom_i


def compute_all_shears(points, neighbors, cell, mode):
    out = {}
    for shear in ["xy", "xz", "yz"]:
        out[shear] = compute_fis(points, neighbors, cell, shear, mode)
    global_mean = float(np.mean([out[s][0] for s in out]))
    return out, global_mean


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="F_IS for TorchDisorder/JCTC network-glass structures.")
    parser.add_argument("--input", required=True, help="Input XYZ/CSV/DAT/TXT structure file.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="System preset: sio2, geo2, lps_ps, or lps_ps_pp.")
    parser.add_argument("--cell", nargs="+", type=float, default=None,
                        help="Orthorhombic periodic cell: L or Lx Ly Lz. Overrides XYZ cell.")
    parser.add_argument("--no_pbc", action="store_true", help="Disable periodic boundaries.")
    parser.add_argument("--central", default=None, help="Comma-separated central species. Overrides preset.")
    parser.add_argument("--neighbors", default=None, help="Comma-separated neighbor species. Overrides preset.")
    parser.add_argument("--cutoff", type=float, default=None, help="Default cutoff. Overrides preset.")
    parser.add_argument("--pair_cutoffs", default=None,
                        help='JSON pair cutoffs, e.g. \'{"Si-O":2.2,"Ge-O":2.4,"P-S":3.5,"P-P":2.8}\'. Overrides preset.')
    parser.add_argument("--mode", choices=["variable_R", "milkus2016"], default="variable_R",
                        help="Recommended for JCTC: variable_R. milkus2016 is constant-R0 comparison.")
    parser.add_argument("--shear", choices=["xy", "xz", "yz", "all"], default="all",
                        help="Shear plane. Use all for isotropic average of xy,xz,yz.")
    parser.add_argument("--prefix", default=None, help="Output prefix. Default derived from input/preset.")
    parser.add_argument("--plot", action="store_true", help="Save histogram PNG of local F_IS values.")
    args = parser.parse_args()

    points, species, file_cell = load_structure(args.input)

    # Apply preset defaults, then command-line overrides.
    preset = PRESETS.get(args.preset) if args.preset else None
    central = args.central if args.central is not None else (preset.central if preset else None)
    neigh_species = args.neighbors if args.neighbors is not None else (preset.neighbors if preset else None)
    cutoff = args.cutoff if args.cutoff is not None else (preset.cutoff if preset else 2.5)
    pair_cutoff_dict = preset.pair_cutoffs.copy() if preset else {}
    if args.pair_cutoffs is not None:
        pair_cutoff_dict = json.loads(args.pair_cutoffs)
    pair_cutoffs = parse_pair_cutoffs(json.dumps(pair_cutoff_dict)) if pair_cutoff_dict else {}

    if species is None and (central is not None or neigh_species is not None or pair_cutoffs):
        raise RuntimeError("Species filters/pair cutoffs require XYZ or CSV input with species labels.")

    if args.no_pbc:
        cell = None
    elif args.cell is not None:
        vals = args.cell
        if len(vals) == 1:
            cell = np.array([vals[0], vals[0], vals[0]], dtype=float)
        elif len(vals) >= 3:
            cell = np.array(vals[:3], dtype=float)
        else:
            raise ValueError("--cell requires L or Lx Ly Lz.")
    else:
        cell = file_cell

    neighbors = build_directed_cutoff_neighbors(
        points=points,
        species=species,
        cell=cell,
        cutoff=cutoff,
        central_species=parse_set(central),
        neighbor_species=parse_set(neigh_species),
        pair_cutoffs=pair_cutoffs,
    )

    prefix = args.prefix
    if prefix is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        suffix = args.preset if args.preset else "custom"
        prefix = f"FIS_{stem}_{suffix}_{args.mode}"

    z = np.array([len(x) for x in neighbors], dtype=int)
    active = z > 0

    if args.shear == "all":
        shear_results, mean_global = compute_all_shears(points, neighbors, cell, args.mode)
        # Save local values for each shear and their local average.
        local_cols = {}
        for shear, (fg, local, Xi, num_i, denom_i) in shear_results.items():
            local_cols[f"FIS_local_{shear}"] = local
            local_cols[f"Xi2_local_{shear}"] = num_i
            local_cols[f"denom_local_{shear}"] = denom_i
        local_stack = np.vstack([local_cols[f"FIS_local_{s}"] for s in ["xy", "xz", "yz"]])
        local_mean = np.nanmean(local_stack, axis=0)
        globals_text = "\n".join([f"global F_IS {s} = {shear_results[s][0]:.10f}" for s in ["xy", "xz", "yz"]])
        global_report = mean_global
    else:
        fg, local, Xi, num_i, denom_i = compute_fis(points, neighbors, cell, args.shear, args.mode)
        local_cols = {f"FIS_local_{args.shear}": local, f"Xi2_local_{args.shear}": num_i, f"denom_local_{args.shear}": denom_i}
        local_mean = local
        globals_text = f"global F_IS {args.shear} = {fg:.10f}"
        global_report = fg

    # Save local data.
    df = pd.DataFrame({
        "species": species if species is not None else ["X"] * len(points),
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "coordination_directed": z,
        "FIS_local_mean_over_shears": local_mean,
        **local_cols,
    })
    df.to_csv(prefix + "_local_FIS.csv", index=False)

    summary = []
    summary.append(f"input = {args.input}")
    summary.append(f"preset = {args.preset}")
    summary.append(f"preset note = {preset.note if preset else 'custom'}")
    summary.append(f"N total = {len(points)}")
    summary.append(f"N active central sites = {int(np.sum(active))}")
    summary.append(f"mode = {args.mode}")
    summary.append(f"shear = {args.shear}")
    summary.append(f"central species = {central}")
    summary.append(f"neighbor species = {neigh_species}")
    summary.append(f"default cutoff = {cutoff}")
    summary.append(f"pair cutoffs = {pair_cutoff_dict}")
    summary.append(f"PBC cell = {None if cell is None else cell.tolist()}")
    summary.append(f"mean directed coordination over all sites = {np.mean(z):.8f}")
    summary.append(f"mean directed coordination over active sites = {np.mean(z[active]):.8f}")
    summary.append(globals_text)
    summary.append(f"reported/global mean F_IS = {global_report:.10f}")
    summary.append(f"mean local F_IS = {np.nanmean(local_mean):.10f}")
    summary.append(f"median local F_IS = {np.nanmedian(local_mean):.10f}")
    summary_text = "\n".join(summary) + "\n"
    with open(prefix + "_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    print(summary_text)
    print(f"Saved: {prefix}_summary.txt")
    print(f"Saved: {prefix}_local_FIS.csv")

    if args.plot:
        import matplotlib.pyplot as plt
        vals = local_mean[np.isfinite(local_mean)]
        plt.figure(figsize=(6.5, 4.8))
        plt.hist(vals, bins=80, density=True, histtype="step", linewidth=2)
        plt.xlabel(r"$F_{IS,i}$")
        plt.ylabel("Probability density")
        title = f"Local $F_{{IS,i}}$ distribution ({args.preset or 'custom'}, {args.mode})"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(prefix + "_FIS_hist.png", dpi=250)
        print(f"Saved: {prefix}_FIS_hist.png")


if __name__ == "__main__":
    main()
