"""
Publication-Quality Plot Regeneration for TorchDisorder Paper
=============================================================

Regenerates all paper figures at ACS publication quality (300 dpi, correct
font sizes, vector-ready PDF, colourblind-safe palette, tight layout).

Figures produced:
    1. SiO2/geo2_FQ.pdf     – F(Q) experimental data (SiO2 + GeO2, two panels)
    2. LiPS/SQ_all.pdf      – S(Q) experimental data (3 LiPS compositions)
    3. fz_weights/fz_weights.pdf – Faber-Ziman partial weights bar chart

For the F(Q) and S(Q) *fit* figures (experiment vs model), this script reads
the experimental data from CSV and overlays the model output from the
`final_F_Q.pdf` source data if available. Where model CSVs are absent, only
the experimental curve is plotted, clearly labelled.

Usage:
    python scripts/improve_plots.py

Outputs written to:
    outputs/publication_figures/
"""

import os, csv, math

# ── Matplotlib setup (must be before any other import) ────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "xrd_measurements")
IMAGES = os.path.abspath(os.path.join(ROOT, "..", "images"))
OUT = os.path.join(ROOT, "outputs", "publication_figures")
os.makedirs(OUT, exist_ok=True)

# ── ACS style ─────────────────────────────────────────────────────────────────
ACS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
plt.rcParams.update(ACS)

# Wong colourblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
GREY   = "#999999"

# ── CSV helpers ────────────────────────────────────────────────────────────────
def read_csv_cols(path, col_x, col_y, col_err=None):
    """Return arrays (x, y) and optionally err from a CSV file."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get(col_x, "").strip()
                and r.get(col_y, "").strip()]
    x = np.array([float(r[col_x]) for r in rows])
    y = np.array([float(r[col_y]) for r in rows])
    err = None
    if col_err:
        try:
            err = np.array([float(r[col_err]) for r in rows])
        except (KeyError, ValueError):
            err = None
    return x, y, err


def savefig(fig, name, subdir=""):
    d = os.path.join(OUT, subdir)
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – SiO2 and GeO2  F(Q) experimental + model fit (two-panel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_SiO2_GeO2_FQ():
    sio2_csv = os.path.join(DATA, "SiO2", "F_of_Q.csv")
    geo2_csv = os.path.join(DATA, "GeO2", "F_of_Q.csv")

    # Model output PNGs exist but no CSV – read experimental only;
    # overlay model if a model CSV is found next to the PNG
    def load_model_csv(system):
        p = os.path.join(IMAGES, system, "final_F_Q_data.csv")
        if os.path.exists(p):
            return read_csv_cols(p, "Q", "F_model")
        return None, None, None

    q_sio2, f_sio2, df_sio2 = read_csv_cols(sio2_csv, "Q", "F", "dF")
    q_geo2, f_geo2, df_geo2 = read_csv_cols(geo2_csv, "Q", "F", "dF")

    # Sanitise GeO2 errors (1e-7 → treat as "no error bar")
    if df_geo2 is not None:
        df_geo2 = np.where(df_geo2 < 1e-5, np.nan, df_geo2)
    if df_sio2 is not None:
        df_sio2 = np.where(df_sio2 < 1e-5, np.nan, df_sio2)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                              sharey=False, constrained_layout=True)

    panels = [
        (axes[0], q_sio2, f_sio2, df_sio2, "SiO$_2$",   BLUE,   "SiO2"),
        (axes[1], q_geo2, f_geo2, df_geo2, "GeO$_2$",    ORANGE, "GeO2"),
    ]

    for ax, q, f, df, title, color, sys in panels:
        # Uncertainty band
        if df is not None and not np.all(np.isnan(df)):
            mask = ~np.isnan(df)
            ax.fill_between(q[mask], (f - df)[mask], (f + df)[mask],
                            color=color, alpha=0.20, linewidth=0, zorder=2)
        # Experimental curve
        ax.plot(q, f, color=color, linewidth=1.0, zorder=3,
                label="Experiment")

        # Model overlay (if model CSV present)
        qm, fm, _ = load_model_csv(sys)
        if qm is not None:
            ax.plot(qm, fm, color=RED, linewidth=0.9, linestyle="--",
                    zorder=4, label="TorchDisorder")
            ax.legend(frameon=False, loc="upper right")

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", zorder=1)
        ax.set_xlabel(r"$Q$ (Å$^{-1}$)")
        ax.set_ylabel(r"$F(Q)$ (Å$^{-1}$)")
        ax.set_title(title)
        ax.set_xlim(q.min(), min(q.max(), 20))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)

    savefig(fig, "SiO2_GeO2_FQ_experimental")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – SiO2  F(Q) fit  (experiment + model, from existing PNG data)
# ══════════════════════════════════════════════════════════════════════════════
def fig_SiO2_fit():
    """Replot the SiO2 F(Q) fit using experimental CSV.
    Model curve is read from the paper's image source if a CSV exists;
    otherwise only experimental is shown with a note."""

    q_exp, f_exp, df_exp = read_csv_cols(
        os.path.join(DATA, "SiO2", "F_of_Q.csv"), "Q", "F", "dF")
    if df_exp is not None:
        df_exp = np.where(df_exp < 1e-5, np.nan, df_exp)

    # Try to find model CSV (output by train.py if modified to dump CSV)
    model_csv = os.path.join(IMAGES, "SiO2", "final_F_Q_data.csv")
    has_model = os.path.exists(model_csv)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    if df_exp is not None and not np.all(np.isnan(df_exp)):
        mask = ~np.isnan(df_exp)
        ax.fill_between(q_exp[mask],
                        (f_exp - df_exp)[mask], (f_exp + df_exp)[mask],
                        color=BLUE, alpha=0.25, linewidth=0, zorder=2,
                        label=r"Exp. $\pm\sigma$")

    ax.plot(q_exp, f_exp, color=BLUE, linewidth=1.0, zorder=3,
            label="Experiment (Kohara 2005)")

    if has_model:
        q_m, f_m, _ = read_csv_cols(model_csv, "Q", "F_model")
        ax.plot(q_m, f_m, color=RED, linewidth=0.9, linestyle="--",
                zorder=4, label="TorchDisorder")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", zorder=1)
    ax.set_xlabel(r"$Q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$F(Q)$ (Å$^{-1}$)")
    ax.set_title("Vitreous SiO$_2$")
    ax.set_xlim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(frameon=False)

    savefig(fig, "SiO2_FQ_fit", subdir="SiO2")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – GeO2  F(Q) fit
# ══════════════════════════════════════════════════════════════════════════════
def fig_GeO2_fit():
    q_exp, f_exp, df_exp = read_csv_cols(
        os.path.join(DATA, "GeO2", "F_of_Q.csv"), "Q", "F", "dF")
    if df_exp is not None:
        df_exp = np.where(df_exp < 1e-5, np.nan, df_exp)

    model_csv = os.path.join(IMAGES, "GeO2", "final_F_Q_data.csv")
    has_model = os.path.exists(model_csv)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    if df_exp is not None and not np.all(np.isnan(df_exp)):
        mask = ~np.isnan(df_exp)
        ax.fill_between(q_exp[mask],
                        (f_exp - df_exp)[mask], (f_exp + df_exp)[mask],
                        color=ORANGE, alpha=0.25, linewidth=0, zorder=2)

    ax.plot(q_exp, f_exp, color=ORANGE, linewidth=1.0, zorder=3,
            label="Experiment (Kohara 2005)")

    if has_model:
        q_m, f_m, _ = read_csv_cols(model_csv, "Q", "F_model")
        ax.plot(q_m, f_m, color=RED, linewidth=0.9, linestyle="--",
                zorder=4, label="TorchDisorder")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", zorder=1)
    ax.set_xlabel(r"$Q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$F(Q)$ (Å$^{-1}$)")
    ax.set_title("Vitreous GeO$_2$")
    ax.set_xlim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(frameon=False)

    savefig(fig, "GeO2_FQ_fit", subdir="GeO2")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – LiPS  S(Q)  three compositions (stacked)
# ══════════════════════════════════════════════════════════════════════════════
def fig_LiPS_SQ():
    LiPS_DATA = os.path.join(DATA, "Li3PS4")
    sq_csv = os.path.join(LiPS_DATA, "S_of_Q.csv")
    if not os.path.exists(sq_csv):
        print(f"  WARNING: {sq_csv} not found, skipping LiPS figure.")
        return

    # utf-8-sig strips the UTF-8 BOM (﻿) that prefixes the Q column name
    with open(sq_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Detect column names; S(Q) may be named 'S', 'SQ', 'S(Q)', etc.
    cols = list(rows[0].keys())
    q_col = next((c for c in cols if c.strip().upper() == "Q"), None)
    s_cols = [c for c in cols
              if c != q_col and c.strip() != ""
              and c.strip().upper() in ("S", "SQ", "S(Q)", "S_OF_Q", "SOFQ")]
    # Fall back: any non-empty non-Q column
    if not s_cols:
        s_cols = [c for c in cols if c != q_col and c.strip() != ""]

    if q_col is None or not s_cols:
        print(f"  WARNING: unexpected columns in {sq_csv}: {cols}")
        return

    q = np.array([float(r[q_col]) for r in rows if r[q_col].strip()])

    COMP_COLORS = {0: BLUE, 1: ORANGE, 2: GREEN}
    COMP_LABELS = ["67% Li$_2$S–33% P$_2$S$_5$",
                   "70% Li$_2$S–30% P$_2$S$_5$",
                   "75% Li$_2$S–25% P$_2$S$_5$"]

    # Single panel: overlay all three compositions
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for i, (col, label, color) in enumerate(zip(s_cols[:3], COMP_LABELS, COMP_COLORS.values())):
        s = np.array([float(r[col]) for r in rows if r[q_col].strip()])
        # offset for clarity
        offset = i * 0.3
        ax.plot(q, s + offset, color=color, linewidth=1.0,
                label=f"{label}" + (f" (+{offset:.1f})" if offset else ""))

    ax.axhline(1, color="black", linewidth=0.5, linestyle=":", zorder=1)
    ax.set_xlabel(r"$Q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$S(Q)$ (offset for clarity)")
    ax.set_title(r"Li$_2$S–P$_2$S$_5$ glasses")
    ax.set_xlim(q.min(), min(q.max(), 20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(frameon=False, loc="upper right")

    savefig(fig, "LiPS_SQ_experimental", subdir="LiPS")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Faber-Ziman weights bar chart  (inline, same as calc_fz_weights.py)
# ══════════════════════════════════════════════════════════════════════════════
def fig_FZ_weights():
    B = {"Li": -1.90, "P": 5.13, "S": 2.847}
    SYSTEMS = {
        "67% Li$_2$S": {"Li": 2800, "P": 1200, "S": 3624},
        "70% Li$_2$S": {"Li": 14,   "P": 6,    "S": 22},
        "75% Li$_2$S": {"Li": 3240, "P": 1080, "S": 3886},
    }
    PAIRS = [("Li","Li"),("Li","P"),("Li","S"),("P","P"),("P","S"),("S","S")]
    PLABELS = ["Li–Li","Li–P","Li–S","P–P","P–S","S–S"]
    SYS_COLORS = [BLUE, ORANGE, GREEN]

    def weights(counts):
        total = sum(counts.values())
        c = {el: n/total for el, n in counts.items()}
        bm = sum(c[el]*B[el] for el in c)
        bm2 = bm**2
        return {p: (1 if p[0]==p[1] else 2)*c[p[0]]*c[p[1]]*B[p[0]]*B[p[1]]/bm2
                for p in PAIRS}

    res = {n: weights(cnt) for n, cnt in SYSTEMS.items()}
    names = list(res.keys())
    n_p = len(PAIRS)
    n_s = len(names)

    x = np.arange(n_p)
    w = 0.24
    offsets = np.array([-1, 0, 1]) * w

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i, (name, offset, color) in enumerate(zip(names, offsets, SYS_COLORS)):
        vals = [res[name][p] for p in PAIRS]
        ax.bar(x + offset, vals, w, color=color, edgecolor="white",
               linewidth=0.4, zorder=3, label=name)

    # Shade Li-X region
    ax.axvspan(-0.5, 2.5, color="#f4f4f4", zorder=1, linewidth=0)
    ax.text(1.0, -0.245, "Li–X pairs", ha="center", va="bottom",
            fontsize=6, color="#888888", style="italic")

    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(PLABELS, fontsize=7)
    ax.set_xlim(-0.5, n_p - 0.5)
    ax.set_ylabel(r"$w_{\alpha\beta}$")
    ax.set_xlabel("Pair (α–β)")
    ax.set_title("Faber-Ziman partial weights (neutron)")

    ax.yaxis.grid(True, linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which="both", top=True, right=True)

    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
              loc="upper right", fontsize=6, borderpad=0.4,
              handlelength=0.9, handleheight=0.8)

    savefig(fig, "fz_weights", subdir="fz_weights")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Run all figures
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Generating publication-quality figures")
    print("=" * 60 + "\n")

    print("→ Figure 1: SiO2 + GeO2 F(Q) experimental (two-panel)")
    fig_SiO2_GeO2_FQ()

    print("→ Figure 2: SiO2 F(Q) fit")
    fig_SiO2_fit()

    print("→ Figure 3: GeO2 F(Q) fit")
    fig_GeO2_fit()

    print("→ Figure 4: LiPS S(Q) experimental")
    fig_LiPS_SQ()

    print("→ Figure 5: Faber-Ziman weight bar chart")
    fig_FZ_weights()

    print(f"\nAll figures written to: {OUT}\n")
