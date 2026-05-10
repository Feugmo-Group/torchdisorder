"""
Faber-Ziman Partial Weight Calculation for Li2S-P2S5 Glass Systems
===================================================================

Computes the Faber-Ziman partial structure factor weights:

    w_αβ = (2 - δ_αβ) * c_α * c_β * b_α * b_β / <b>²

where:
    c_α  = atomic fraction of species α
    b_α  = coherent neutron scattering length (fm)
    <b>  = Σ_α c_α b_α  (mean scattering length)
    δ_αβ = Kronecker delta (1 for like pairs, 0 for unlike)

Positive w_αβ → pair contributes positively to S(Q).
Negative w_αβ → pair contributes negatively (cancellation / contrast inversion).

Compositions are taken directly from the CIF files used in the paper:
    67% Li2S: glass_67Li2S_WithLi.cif  → Li2800 P1200 S3624
    70% Li2S: derived from molar mixing  → Li14 P6 S22 (Li7P3S11 formula)
    75% Li2S: glass_75Li2S.cif          → Li3240 P1080 S3886

Outputs:
    - Console table of weights per composition
    - fz_weights_table.csv  (for inclusion in supplementary data)
    - fz_weights.pdf / fz_weights.png  (publication-quality bar chart)

Usage:
    python scripts/calc_fz_weights.py
"""

import os
import csv
import math
import itertools

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "fz_weights")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Coherent neutron scattering lengths (fm), NIST values ────────────────────
B = {"Li": -1.90, "P": 5.13, "S": 2.847}

# ── Atomic compositions (from CIF files, verified by smoke_test_lips.py) ─────
#   67% Li2S: glass_67Li2S_withLi.cif  → Li2800 P1200 S3854
#   70% Li2S: glass_70Li2S_withLi.cif  → Li2800 P1200 S3865
#   75% Li2S: glass_75Li2S_withLi.cif  → Li3240 P1080 S3886
SYSTEMS = {
    "67% Li$_2$S": {"Li": 2800, "P": 1200, "S": 3854},
    "70% Li$_2$S": {"Li": 2800, "P": 1200, "S": 3865},
    "75% Li$_2$S": {"Li": 3240, "P": 1080, "S": 3886},
}

PAIRS = [("Li", "Li"), ("Li", "P"), ("Li", "S"), ("P", "P"), ("P", "S"), ("S", "S")]

# ── Computation ───────────────────────────────────────────────────────────────
def compute_weights(counts: dict[str, int]) -> dict:
    total = sum(counts.values())
    c = {el: n / total for el, n in counts.items()}

    b_mean = sum(c[el] * B[el] for el in c)
    b_mean_sq = b_mean ** 2

    weights = {}
    for a1, a2 in PAIRS:
        mult = 1 if a1 == a2 else 2
        w = mult * c[a1] * c[a2] * B[a1] * B[a2] / b_mean_sq
        weights[(a1, a2)] = w

    total_w = sum(weights.values())
    return {"c": c, "b_mean": b_mean, "b_mean_sq": b_mean_sq,
            "weights": weights, "total_check": total_w}


results = {}
for name, counts in SYSTEMS.items():
    results[name] = compute_weights(counts)

# ── Console output ────────────────────────────────────────────────────────────
PAIR_LABELS = {("Li", "Li"): "Li–Li", ("Li", "P"): "Li–P",
               ("Li", "S"): "Li–S",  ("P", "P"): "P–P",
               ("P", "S"): "P–S",   ("S", "S"): "S–S"}

print("\n" + "=" * 72)
print("  Faber-Ziman Partial Weights  w_αβ = (2-δ_αβ) cα cβ bα bβ / <b>²")
print("=" * 72)
print(f"\n  Scattering lengths: b_Li = {B['Li']:.2f} fm, "
      f"b_P = {B['P']:.2f} fm, b_S = {B['S']:.3f} fm\n")

col = 18
header = f"  {'Pair':<8}" + "".join(f"{n:>{col}}" for n in results)
print(header)
print("  " + "-" * (8 + col * len(results)))

for pair in PAIRS:
    label = PAIR_LABELS[pair]
    row = f"  {label:<8}"
    for name, res in results.items():
        w = res["weights"][pair]
        row += f"{w:>{col}.5f}"
    print(row)

print("  " + "-" * (8 + col * len(results)))
# Li contributions summed
for name, res in results.items():
    w = res["weights"]
    li_total = w[("Li", "Li")] + w[("Li", "P")] + w[("Li", "S")]
    ps_total = w[("P", "P")] + w[("P", "S")] + w[("S", "S")]
    print(f"\n  [{name}]")
    print(f"    <b>      = {res['b_mean']:+.4f} fm")
    print(f"    Li-X sum = {li_total:+.5f}  (Li–Li + Li–P + Li–S)")
    print(f"    P,S  sum = {ps_total:+.5f}  (P–P + P–S + S–S)")
    print(f"    Total    = {res['total_check']:.5f}  (should equal 1.000)")

print()

# ── CSV output ────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "fz_weights_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    names = list(results.keys())
    writer.writerow(["Pair"] + names)
    for pair in PAIRS:
        row = [PAIR_LABELS[pair]] + [f"{results[n]['weights'][pair]:.5f}" for n in names]
        writer.writerow(row)
    writer.writerow([])
    writer.writerow(["<b> (fm)"] + [f"{results[n]['b_mean']:.4f}" for n in names])
    writer.writerow(["Li-X sum"] + [
        f"{results[n]['weights'][('Li','Li')] + results[n]['weights'][('Li','P')] + results[n]['weights'][('Li','S')]:.5f}"
        for n in names])
    writer.writerow(["P,S sum"] + [
        f"{results[n]['weights'][('P','P')] + results[n]['weights'][('P','S')] + results[n]['weights'][('S','S')]:.5f}"
        for n in names])

print(f"  CSV saved: {csv_path}")

# ── Publication-quality figure ────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ACS journal style parameters
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    # Color palette: colorblind-safe, distinguishable
    # Wong (2011) colorblind palette
    COLORS = {
        "67% Li$_2$S": "#0072B2",   # blue
        "70% Li$_2$S": "#E69F00",   # orange
        "75% Li$_2$S": "#009E73",   # green
    }

    pair_labels = [PAIR_LABELS[p] for p in PAIRS]
    system_names = list(results.keys())
    n_pairs = len(PAIRS)
    n_sys = len(system_names)

    x = np.arange(n_pairs)
    width = 0.25
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width

    # ACS single-column width = 3.3 in; double-column = 7.0 in
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    bars_list = []
    for i, (name, offset) in enumerate(zip(system_names, offsets)):
        w_vals = [results[name]["weights"][p] for p in PAIRS]
        color = COLORS[name]
        bars = ax.bar(x + offset, w_vals, width,
                      color=color, edgecolor="white",
                      linewidth=0.4, zorder=3, label=name)
        bars_list.append(bars)

    # Reference line at zero
    ax.axhline(0, color="black", linewidth=0.6, zorder=2)

    # Shade Li-X region background
    ax.axvspan(-0.5, 2.5, color="#f0f0f0", zorder=1)
    ax.text(1.0, ax.get_ylim()[0] if ax.get_ylim()[0] > -0.15 else -0.14,
            "Li–X pairs", ha="center", va="bottom", fontsize=6,
            color="#888888", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=0)
    ax.set_xlim(-0.5, n_pairs - 0.5)
    ax.set_ylabel(r"Faber-Ziman weight $w_{\alpha\beta}$")
    ax.set_xlabel("Pair (α–β)")

    # Grid on y only
    ax.yaxis.grid(True, linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
                       loc="upper right", borderpad=0.4, handlelength=1.0,
                       handleheight=0.8)

    ax.tick_params(which="both", top=True, right=True)

    fig.tight_layout(pad=0.3)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fz_weights.{ext}")
        fig.savefig(path)
        print(f"  Figure saved: {path}")

    plt.close(fig)

except ImportError as e:
    print(f"\n  WARNING: matplotlib not available ({e}). Skipping figure.")

print("\nDone.\n")
