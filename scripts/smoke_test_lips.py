"""
Smoke test for all 6 LiPS structure variants.

Checks (without running full training):
  1. CIF file exists and parses correctly
  2. JSON constraints file is valid and has expected keys
  3. Experimental data files load (Q-range, array shape)
  4. One forward pass of XRDModel succeeds on CPU

Run from project root:
    conda activate torchdisorder
    python scripts/smoke_test_lips.py
"""

import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VARIANTS = [
    {
        "name": "67Li2S-33P2S5  noLi",
        "cif":  DATA / "crystal-structures/glass_67Li2S_noLi.cif",
        "json": DATA / "json/glass_67Li2S_noLi_constraints.json",
        "has_li": False,
    },
    {
        "name": "67Li2S-33P2S5  withLi",
        "cif":  DATA / "crystal-structures/glass_67Li2S_withLi.cif",
        "json": DATA / "json/glass_67Li2S_withLi_constraints.json",
        "has_li": True,
    },
    {
        "name": "70Li2S-30P2S5  noLi",
        "cif":  DATA / "crystal-structures/glass_70Li2S_noLi.cif",
        "json": DATA / "json/glass_70Li2S_noLi_constraints.json",
        "has_li": False,
    },
    {
        "name": "70Li2S-30P2S5  withLi",
        "cif":  DATA / "crystal-structures/glass_70Li2S_withLi.cif",
        "json": DATA / "json/glass_70Li2S_withLi_constraints.json",
        "has_li": True,
    },
    {
        "name": "75Li2S-25P2S5  noLi",
        "cif":  DATA / "crystal-structures/glass_75Li2S_noLi.cif",
        "json": DATA / "json/glass_75Li2S_noLi_constraints.json",
        "has_li": False,
    },
    {
        "name": "75Li2S-25P2S5  withLi",
        "cif":  DATA / "crystal-structures/glass_75Li2S_withLi.cif",
        "json": DATA / "json/glass_75Li2S_withLi_constraints.json",
        "has_li": True,
    },
]

SQ_DATA  = DATA / "xrd_measurements/Li3PS4/S_of_Q.csv"
GR_DATA  = DATA / "xrd_measurements/Li3PS4/g_of_r.csv"

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(label, ok, detail=""):
    symbol = PASS if ok else FAIL
    print(f"    [{symbol} ] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def test_variant(v):
    name     = v["name"]
    cif_path = v["cif"]
    json_path = v["json"]
    has_li   = v["has_li"]
    all_ok   = True

    print(f"\n  {'=' * 60}")
    print(f"  {name}")
    print(f"  {'=' * 60}")

    # ── 1. CIF ──────────────────────────────────────────────────
    try:
        from pymatgen.core import Structure
        s = Structure.from_file(str(cif_path))
        elem = dict(s.composition.as_dict())
        n = s.num_sites
        has_li_in_cif = "Li+" in elem or "Li" in elem
        ok = cif_path.exists() and (has_li == has_li_in_cif)
        all_ok &= check("CIF loads", ok,
                        f"{n} atoms  {elem}")
    except Exception as e:
        all_ok &= check("CIF loads", False, str(e))
        s = None

    # ── 2. JSON constraints ──────────────────────────────────────
    try:
        with open(json_path) as f:
            cdata = json.load(f)
        n_p = len(cdata.get("atom_constraints", {}))
        li_block = cdata.get("li_constraints", None)
        ef = cdata.get("element_filter", [])
        ok_ef = (ef == [15, 16])
        ok_li = (has_li == (li_block is not None))
        ok = ok_ef and ok_li
        detail = f"{n_p} P constraints  element_filter={ef}"
        if has_li and li_block:
            detail += f"  li_atoms={li_block['n_li_atoms']}"
        all_ok &= check("JSON constraints", ok, detail)
        if not ok_ef:
            print(f"           ⚠ element_filter should be [15,16], got {ef}")
        if not ok_li:
            print(f"           ⚠ li_constraints present={li_block is not None}, expected has_li={has_li}")
    except Exception as e:
        all_ok &= check("JSON constraints", False, str(e))

    # ── 3. Experimental data ─────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(SQ_DATA, encoding="utf-8-sig")
        # Normalise column names: strip whitespace, upper-case
        df.columns = [c.strip() for c in df.columns]
        q_col = next((c for c in df.columns if c.upper() == "Q"), None)
        s_col = next((c for c in df.columns
                      if c.upper() in ("F", "S", "SQ", "S(Q)")), None)
        ok = SQ_DATA.exists() and q_col and s_col
        if ok:
            df = df.dropna(subset=[q_col, s_col])
            q_min, q_max = df[q_col].min(), df[q_col].max()
            detail = f"{len(df)} bins  Q=[{q_min:.2f},{q_max:.2f}]  col='{s_col}'"
        else:
            detail = f"columns={list(df.columns)}"
        all_ok &= check("S(Q) data file", bool(ok), detail)
    except Exception as e:
        all_ok &= check("S(Q) data file", False, str(e))

    # ── 4. Forward pass on small random subset ───────────────────
    try:
        import torch
        from torch_sim.io import atoms_to_state
        from ase import Atoms as AseAtoms
        from torchdisorder.model.xrd import XRDModel

        if s is None:
            all_ok &= check("Forward pass", False, "CIF failed to load")
            return all_ok

        # Subsample to ≤200 atoms for speed on CPU
        n_sub = min(200, s.num_sites)
        idx = np.random.choice(s.num_sites, n_sub, replace=False)
        sites = [s[i] for i in idx]

        symbols  = [site.specie.symbol for site in sites]
        positions = np.array([site.coords for site in sites])
        cell = np.array(s.lattice.matrix)

        atoms = AseAtoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        state = atoms_to_state([atoms], device="cpu", dtype=torch.float32)
        state.positions.requires_grad_(True)

        # Q bins from experimental data (subsampled)
        import pandas as pd
        df = pd.read_csv(SQ_DATA, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        q_col = next(c for c in df.columns if c.upper() == "Q")
        s_col = next(c for c in df.columns if c.upper() in ("F","S","SQ","S(Q)"))
        df = df.dropna(subset=[q_col, s_col])
        q_arr = torch.tensor(df[q_col].to_numpy(dtype="float32")[::4])
        q_arr = q_arr[(q_arr >= 0.5) & (q_arr <= 17.5)]

        r_bins = torch.linspace(0.01, 50.0, 200)

        unique_sym = list(dict.fromkeys(symbols))
        model_cfg = {
            "kernel_width": 0.03,
            "scattering_type": "xray",
            "neutron_scattering_lengths": {"Li": -1.90, "P": 5.13, "S": 2.847},
            "xray_form_factor_params": {
                "Li": {"a":[1.1282,0.7508,0.6175,0.4653],
                       "b":[3.9546,1.0524,85.3905,168.261], "c":[0.0377]},
                "P":  {"a":[6.4345,4.1791,1.78,1.4908],
                       "b":[1.9067,27.157,0.526,68.1645], "c":[1.1149]},
                "S":  {"a":[6.9053,5.2034,1.4379,1.5863],
                       "b":[1.4679,22.2151,0.2536,56.172], "c":[0.8669]},
            },
        }
        model = XRDModel(symbols=unique_sym, config=model_cfg,
                         r_bins=r_bins, q_bins=q_arr, device="cpu")
        results = model(state)
        s_q = results.get("S_Q")
        ok = s_q is not None and not torch.isnan(s_q).any()
        all_ok &= check("Forward pass (200-atom subset)",
                        ok, f"S(Q) shape={s_q.shape} range=[{s_q.min():.3f},{s_q.max():.3f}]"
                        if ok else "NaN or missing output")
    except Exception as e:
        all_ok &= check("Forward pass (200-atom subset)", False, str(e)[:120])

    return all_ok


def main():
    print("\n" + "=" * 64)
    print("  TorchDisorder — LiPS Structure Smoke Tests")
    print("=" * 64)

    results = {}
    for v in VARIANTS:
        results[v["name"]] = test_variant(v)

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    all_passed = True
    for name, ok in results.items():
        symbol = PASS if ok else FAIL
        print(f"  [{symbol} ] {name}")
        all_passed &= ok

    print()
    if all_passed:
        print("  All tests passed — safe to transfer to GPU cluster.")
    else:
        print("  Some tests failed — fix issues before transferring.")
    print()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
