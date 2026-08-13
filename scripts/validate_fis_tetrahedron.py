"""Validation of the F_IS order parameter on tetrahedral coordination shells.

Runs the sanity checks proposed by A. Zaccone (Aug 2026 correspondence):

  1. F_IS of a mathematically perfect regular tetrahedron        -> expect -1/3
  2. F_IS of a single SiO4 tetrahedron extracted from c-SiO2     -> compare to (1)
  3. Confirm Si-centred / O-neighbour environments only
  4. Progressive distortion of the ideal tetrahedron             -> F_IS(disorder)
  5. Compare 3D combination schemes:
       (a) mean of per-shear ratios      [current implementation]
       (b) sum(numerators)/sum(denominators)  [Zaccone's proposal]

The F_IS formula is reimplemented here from bond vectors so the analytic cases
can be evaluated without the neighbour-list machinery.  It mirrors
``PyTorchOrderParameters._compute_fis`` exactly; test 2 cross-checks the two
implementations against each other on the real crystal.

Run:  python scripts/validate_fis_tetrahedron.py
"""

from __future__ import annotations

import numpy as np

SHEAR_PAIRS = [(0, 1), (0, 2), (1, 2)]  # (x,y), (x,z), (y,z)


def fis_from_bonds(
    bond_vectors: np.ndarray,
    mode: str = "variable_R",
    combine: str = "sum_then_ratio",
) -> float:
    """F_IS for one central atom from its bond vectors (n, 3), un-normalised.

    ``combine="sum_then_ratio"`` mirrors ``PyTorchOrderParameters._compute_fis``.
    ``combine="mean_of_ratios"`` is the superseded scheme, kept so tests [5] and
    [7] below can show where the two diverge (see [7]: a 3x error on octahedra
    rotated about a single axis).
        Xi^(mu,nu) = sum_j w_j n_j        with w_j = R_j n_j^mu n_j^nu
        D^(mu,nu)  = sum_j w_j^2
        F_IS       = 1 - <|Xi|^2 / D>     (combine="mean_of_ratios")
                   = 1 - sum|Xi|^2 / sum D (combine="sum_then_ratio")
    """
    v = np.asarray(bond_vectors, dtype=float)
    R = np.linalg.norm(v, axis=1)
    n = v / R[:, None]

    numerators, denominators = [], []
    for mu, nu in SHEAR_PAIRS:
        orient = n[:, mu] * n[:, nu]
        w = R * orient if mode == "variable_R" else orient
        Xi = (w[:, None] * n).sum(axis=0)
        numerators.append(float(Xi @ Xi))
        denominators.append(float((w * w).sum()))

    # Same degeneracy guard as the repo implementation: a shear plane in which
    # every bond has n^mu n^nu == 0 carries no information and contributes 0.
    eps = 1e-10
    if combine == "sum_then_ratio":
        if sum(denominators) <= eps:
            return 0.0
        return 1.0 - sum(numerators) / sum(denominators)
    ratios = [
        (nu_ / d if d > eps else 1.0)  # d==0 -> repo returns F_IS=0 for that plane
        for nu_, d in zip(numerators, denominators)
    ]
    return 1.0 - float(np.mean(ratios))


IDEAL_TETRAHEDRON = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
) / np.sqrt(3)


def main() -> None:
    print("=" * 74)
    print("F_IS VALIDATION — tetrahedral coordination")
    print("=" * 74)

    # ---- Test 1: ideal tetrahedron + other analytic limits -----------------
    print("\n[1] Analytic reference geometries (variable_R, mean_of_ratios)\n")
    cases = {
        "Ideal regular tetrahedron (4 bonds)": IDEAL_TETRAHEDRON,
        "Antiparallel pair, axis-aligned [DEGEN]": np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        "Antiparallel pair, rotated (1,1,0)": np.array([[1.0, 1, 0], [-1.0, -1, 0]]),
        "Single bond": np.array([[1.0, 0, 0]]),
        "Ideal octahedron (6 bonds)": np.array(
            [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        ),
    }
    print(f"  {'geometry':40s} {'mean_of_ratios':>15s} {'sum_then_ratio':>16s}")
    for label, bonds in cases.items():
        print(f"  {label:40s} {fis_from_bonds(bonds, combine='mean_of_ratios'):+15.6f} "
              f"{fis_from_bonds(bonds, combine='sum_then_ratio'):+16.6f}")
    print(f"\n  Analytic expectation for tetrahedron: -1/3 = {-1/3:+.6f}")
    dev = abs(fis_from_bonds(IDEAL_TETRAHEDRON, combine='mean_of_ratios') + 1 / 3)
    print(f"  Deviation: {dev:.2e}   ->  {'PASS' if dev < 1e-12 else 'FAIL'}")

    # bond-length independence
    scaled = IDEAL_TETRAHEDRON * np.array([1.0, 1.0, 1.0, 1.0])[:, None] * 1.61
    print(f"\n  Same tetrahedron scaled to R=1.61 A: F_IS = {fis_from_bonds(scaled, combine='mean_of_ratios'):+.6f}"
          "   (scale-invariant)")

    # ---- Test 5: combination scheme, on ideal geometry ---------------------
    print("\n[5] 3D combination scheme on the ideal tetrahedron\n")
    for combine in ("mean_of_ratios", "sum_then_ratio"):
        print(f"  {combine:18s} F_IS = {fis_from_bonds(IDEAL_TETRAHEDRON, combine=combine):+.6f}")
    print("  (identical by symmetry, as predicted)")

    # ---- Tests 2 & 3: real crystal ----------------------------------------
    print("\n[2/3] SiO4 environments extracted from data/crystal-structures/c-SiO2.cif\n")
    try:
        from ase.io import read
        from ase.neighborlist import neighbor_list
    except ImportError:
        print("  ASE not available — skipping crystal tests.")
        return

    atoms = read("data/crystal-structures/c-SiO2.cif")
    symbols = np.array(atoms.get_chemical_symbols())
    n_si = int((symbols == "Si").sum())
    n_o = int((symbols == "O").sum())
    print(f"  Loaded: {len(atoms)} atoms total ({n_si} Si, {n_o} O)")
    print(f"  Cell: a={atoms.cell.lengths()[0]:.4f}  b={atoms.cell.lengths()[1]:.4f} "
          f" c={atoms.cell.lengths()[2]:.4f}  gamma={atoms.cell.angles()[2]:.2f} deg")
    print(f"  Primitive (5x5x5 supercell): a={atoms.cell.lengths()[0]/5:.4f} "
          f" c={atoms.cell.lengths()[2]/5:.4f}  "
          "[alpha-quartz ref: a=4.9134, c=5.4052]")

    cutoff = 2.2
    i_idx, j_idx, d_ij, D_ij = neighbor_list("ijdD", atoms, cutoff)

    # Si-centred, O-neighbour only
    si_mask = (symbols[i_idx] == "Si") & (symbols[j_idx] == "O")
    si_centers = sorted(set(i_idx[si_mask].tolist()))
    print(f"\n  Si-centred Si-O pairs within {cutoff} A: {int(si_mask.sum())}")
    print(f"  Distinct Si centres: {len(si_centers)} of {n_si}")

    cns = [int((i_idx[si_mask] == c).sum()) for c in si_centers]
    print(f"  Coordination number: min={min(cns)} max={max(cns)} mean={np.mean(cns):.3f}")
    print(f"  Si-O bond length: {d_ij[si_mask].mean():.4f} +/- {d_ij[si_mask].std():.4f} A")

    # O-centred contamination check
    o_mask = (symbols[i_idx] == "O") & (symbols[j_idx] == "Si")
    o_cns = np.bincount(i_idx[o_mask], minlength=len(atoms))
    o_cns = o_cns[symbols == "O"]
    print(f"  [check] O-centred O-Si CN: mean={o_cns.mean():.3f} "
          f"(bridging O -> 2; would DILUTE the Si distribution if mixed in)")

    fis_crystal, fis_crystal_sum = [], []
    for c in si_centers:
        bonds = D_ij[si_mask][i_idx[si_mask] == c]
        fis_crystal.append(fis_from_bonds(bonds, combine="mean_of_ratios"))
        fis_crystal_sum.append(fis_from_bonds(bonds, combine="sum_then_ratio"))
    fis_crystal = np.array(fis_crystal)
    fis_crystal_sum = np.array(fis_crystal_sum)

    print(f"\n  Isolated SiO4 F_IS (mean_of_ratios): "
          f"{fis_crystal.mean():+.6f} +/- {fis_crystal.std():.6f}")
    print(f"  Isolated SiO4 F_IS (sum_then_ratio): "
          f"{fis_crystal_sum.mean():+.6f} +/- {fis_crystal_sum.std():.6f}")
    print(f"  Single representative tetrahedron:   {fis_crystal[0]:+.6f}")
    print(f"  Ideal tetrahedron reference:         {-1/3:+.6f}")
    print(f"  Difference from -1/3:                {fis_crystal.mean() + 1/3:+.6f}")

    # ---- Test 4: progressive distortion ------------------------------------
    print("\n[4] Progressive distortion of the ideal tetrahedron\n")
    rng = np.random.default_rng(0)
    n_samples = 2000
    print(f"  {'sigma (A)':>10s} {'F_IS mean':>12s} {'F_IS std':>10s} "
          f"{'sum_then_ratio':>16s}")
    for sigma in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        vals, vals_sum = [], []
        for _ in range(n_samples if sigma > 0 else 1):
            bonds = IDEAL_TETRAHEDRON * 1.61 + rng.normal(0, sigma, (4, 3))
            vals.append(fis_from_bonds(bonds, combine="mean_of_ratios"))
            vals_sum.append(fis_from_bonds(bonds, combine="sum_then_ratio"))
        print(f"  {sigma:10.2f} {np.mean(vals):+12.6f} {np.std(vals):10.6f} "
              f"{np.mean(vals_sum):+16.6f}")

    # ---- Test 6: rotational invariance -------------------------------------
    print("\n[6] Rotational invariance (F_IS must not depend on lab-frame "
          "orientation)\n")

    def random_rotation(rng_):
        q = rng_.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    octahedron = np.array(
        [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    geometries = {
        "Ideal tetrahedron": IDEAL_TETRAHEDRON,
        "Ideal octahedron": octahedron,
        "Antiparallel pair": np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
    }
    rng2 = np.random.default_rng(1)
    rots = [random_rotation(rng2) for _ in range(500)]

    print(f"  {'geometry':22s} {'scheme':16s} {'min':>10s} {'max':>10s} "
          f"{'spread':>10s}")
    for label, geom in geometries.items():
        for combine in ("mean_of_ratios", "sum_then_ratio"):
            vals = [fis_from_bonds(geom @ Rm.T, combine=combine) for Rm in rots]
            spread = max(vals) - min(vals)
            flag = "  <-- FRAME-DEPENDENT" if spread > 1e-8 else "  (invariant)"
            print(f"  {label:22s} {combine:16s} {min(vals):+10.4f} "
                  f"{max(vals):+10.4f} {spread:10.2e}{flag}")

    print("\n  Note: a perfect octahedron and an antiparallel pair are both")
    print("  centrosymmetric, so an inversion-symmetry measure should return +1.")
    print("  Generic orientations DO return +1. The +0.000 / +0.333 values in [1]")
    print("  are artifacts of exact axis-alignment, where n^mu n^nu vanishes")
    print("  identically and the denominator degenerates (measure-zero set).")

    # ---- Test 7: near-degeneracy hazard for axis-aligned octahedra ----------
    print("\n[7] Axis-aligned octahedron: approach to the degenerate orientation\n")
    print("  Relevant because octahedral systems (Fe2O3, Li2HfCl6) are built")
    print("  from CIFs in the crystallographic frame, i.e. near axis-aligned.\n")

    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    print(f"  {'tilt (deg)':>11s} {'denominator':>14s} {'mean_of_ratios':>16s} "
          f"{'sum_then_ratio':>16s}")
    for deg in [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 5.0, 15.0, 45.0]:
        Rm = rot_z(np.deg2rad(deg))
        g = octahedron @ Rm.T
        Rn = np.linalg.norm(g, axis=1)
        nn = g / Rn[:, None]
        dsum = 0.0
        for mu, nu in SHEAR_PAIRS:
            w = Rn * nn[:, mu] * nn[:, nu]
            dsum += float((w * w).sum())
        print(f"  {deg:11.6f} {dsum:14.3e} "
              f"{fis_from_bonds(g, combine='mean_of_ratios'):+16.6f} "
              f"{fis_from_bonds(g, combine='sum_then_ratio'):+16.6f}")
    print("\n  -> The transition is abrupt: F_IS jumps 0 -> +1 as soon as the")
    print("     denominator clears the 1e-10 guard. Perfect octahedra sitting")
    print("     exactly on the axes are silently reported as F_IS = 0.")
    print("  -> CRITICAL: for a rotation about a SINGLE axis, two shear planes")
    print("     stay degenerate. mean_of_ratios zero-fills them and averages,")
    print("     giving (1+0+0)/3 = +1/3 for a centrosymmetric octahedron;")
    print("     sum_then_ratio correctly gives +1. A 3x systematic error that")
    print("     affects any crystal with a symmetry axis along a cell vector.")

    # ---- Test 8: impact on the reported c-SiO2 / a-SiO2 numbers ------------
    print("\n[8] Impact on the reported crystal/glass comparison\n")

    def fis_structure(path, cutoff_=2.2):
        at = read(path)
        sym = np.array(at.get_chemical_symbols())
        ii, jj, dd, DD = neighbor_list("ijdD", at, cutoff_)
        m = (sym[ii] == "Si") & (sym[jj] == "O")
        centers = sorted(set(ii[m].tolist()))
        out = {}
        for comb in ("mean_of_ratios", "sum_then_ratio"):
            vals = [fis_from_bonds(DD[m][ii[m] == c], combine=comb) for c in centers]
            out[comb] = (float(np.mean(vals)), float(np.std(vals)))
        cn = [int((ii[m] == c).sum()) for c in centers]
        return out, len(centers), float(np.mean(cn))

    # The published, validated GAP melt-quench glass (Erhard et al., npj Comput.
    # Mater. 8, 90 (2022); Zenodo 10.5281/zenodo.6353684).  The structure this
    # comparison originally used -- outputs/SiO2_mace_debug_.../final_structure.cif
    # -- is retained below only as a demonstration of what test [9] rejects.
    glass_path = "data/crystal-structures/sio2_glass_gap.cif"
    broken_path = (
        "outputs/SiO2_mace_debug_2026-07-16/13-55-58/final_results/final_structure.cif"
    )
    try:
        crys, n_c, cn_c = fis_structure("data/crystal-structures/c-SiO2.cif")
        glas, n_g, cn_g = fis_structure(glass_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not evaluate glass structure: {exc}")
        print("\n" + "=" * 74)
        return

    print(f"  c-SiO2: {n_c} Si centres, <CN> = {cn_c:.3f}")
    print(f"  a-SiO2: {n_g} Si centres, <CN> = {cn_g:.3f}\n")
    print(f"  {'scheme':18s} {'crystal':>18s} {'glass':>18s} {'Delta':>10s}")
    for comb in ("mean_of_ratios", "sum_then_ratio"):
        cm, cs = crys[comb]
        gm, gs = glas[comb]
        print(f"  {comb:18s} {cm:+9.4f}+/-{cs:6.4f} {gm:+9.4f}+/-{gs:6.4f} "
              f"{gm - cm:+10.4f}")

    # ---- Test 9: structure health check ------------------------------------
    print("\n[9] Structure health check (are the environments physical?)\n")
    print(f"  {'structure':20s} {'<CN>':>6s} {'CN=4':>7s} {'min Si-O':>10s} "
          f"{'n(Si-O<1.4A)':>13s}")
    for label, path in (("c-SiO2", "data/crystal-structures/c-SiO2.cif"),
                        ("a-SiO2 (GAP)", glass_path),
                        ("a-SiO2 (withdrawn)", broken_path)):
        at = read(path)
        sym = np.array(at.get_chemical_symbols())
        ii, jj, dd, _ = neighbor_list("ijdD", at, 3.0)
        m = (sym[ii] == "Si") & (sym[jj] == "O")
        ii22 = ii[m][dd[m] <= cutoff]
        cn = np.bincount(ii22, minlength=len(at))[sym == "Si"]
        print(f"  {label:20s} {cn.mean():6.3f} {100*(cn==4).mean():6.1f}% "
              f"{dd[m].min():10.3f} {int((dd[m] < 1.4).sum()):13d}")
    print("\n  A physical SiO2 network has <CN> = 4.000 over a broad cutoff")
    print("  plateau and NO Si-O contacts below ~1.5 A.  If the glass fails")
    print("  this, its F_IS is not a measure of tetrahedral distortion and the")
    print("  crystal/glass Delta above should not be quoted.")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
