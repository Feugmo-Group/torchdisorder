"""
Analysis Plots
==============

Publication-quality matplotlib figures for post-training structure analysis.

All functions return a ``plt.Figure``.  Pass ``save_path`` to also write the
figure to disk at 300 DPI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

# Use a clean whitegrid style (fall back gracefully)
for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

_DPI = 300
_FIGSIZE_SINGLE = (6, 4)
_FIGSIZE_WIDE = (8, 4)

__all__ = [
    "plot_rdf",
    "plot_sq",
    "plot_order_param_histogram",
    "plot_cn_distribution",
    "plot_bond_length_distribution",
    "plot_bond_angle_distribution",
    "plot_fis_q4_scatter",
    "plot_convergence",
    "plot_summary_panel",
    "plot_steinhardt_distributions",
    "plot_disorder_heatmap",
    "plot_q4_q6_trajectory",
    "plot_fis_by_cn",
    "plot_distortion_index",
    "plot_warren_cowley",
    "plot_fis_spatial_autocorrelation",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_path: Optional[str | Path]) -> plt.Figure:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=_DPI, bbox_inches="tight")
    return fig


def _kde_overlay(
    ax: plt.Axes,
    data: np.ndarray,
    x_min: float,
    x_max: float,
    n: int = 300,
    **kwargs,
) -> None:
    """Overlay a KDE curve on the given axes."""
    if len(data) < 2:
        return
    try:
        kde = gaussian_kde(data, bw_method="scott")
        xs = np.linspace(x_min, x_max, n)
        # Scale so that area ≈ histogram area when density=True
        ax.plot(xs, kde(xs), **kwargs)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_rdf(
    r: np.ndarray,
    g_r_computed: np.ndarray,
    r_exp: Optional[np.ndarray] = None,
    g_r_exp: Optional[np.ndarray] = None,
    pair_label: str = "",
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Overlay computed g(r) vs experimental data."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(r, g_r_computed, color="steelblue", linewidth=1.5, label="Computed")
    if r_exp is not None and g_r_exp is not None:
        ax.plot(r_exp, g_r_exp, color="black", linewidth=1.2,
                linestyle="--", label="Experiment")
    ax.set_xlabel(r"$r$ (Å)", fontsize=12)
    ax.set_ylabel(r"$g(r)$", fontsize=12)
    ax.set_title(title or f"g(r)  {pair_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_sq(
    q: np.ndarray,
    s_q_computed: np.ndarray,
    q_exp: Optional[np.ndarray] = None,
    s_q_exp: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Overlay computed S(Q) vs experimental data."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(q, s_q_computed, color="steelblue", linewidth=1.5, label="Computed")
    if q_exp is not None and s_q_exp is not None:
        ax.plot(q_exp, s_q_exp, color="black", linewidth=1.2,
                linestyle="--", label="Experiment")
    ax.set_xlabel(r"$Q$ (Å$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"$S(Q)$", fontsize=12)
    ax.set_title(title or "S(Q)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_order_param_histogram(
    values: np.ndarray,
    label: str,
    color: str = "steelblue",
    vline: Optional[float] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of a per-atom order parameter with optional KDE overlay."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    if len(values) > 0:
        ax.hist(values, bins=30, color=color, alpha=0.75, edgecolor="white",
                density=True, label=label)
        _kde_overlay(ax, values, float(values.min()), float(values.max()),
                     color=color, linewidth=2.0)
    if vline is not None:
        ax.axvline(vline, color="crimson", linewidth=1.5, linestyle="--",
                   label=f"Crystal ref: {vline:.3f}")
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of {label}", fontsize=13)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_cn_distribution(
    cn_values: np.ndarray,
    element: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart of coordination number distribution."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    if len(cn_values) > 0:
        unique, counts = np.unique(cn_values, return_counts=True)
        fractions = counts / counts.sum()
        ax.bar([str(u) for u in unique], fractions, color="steelblue",
               edgecolor="white", alpha=0.85)
        for x, y in zip([str(u) for u in unique], fractions):
            ax.text(x, y + 0.005, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Coordination number", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title(f"CN distribution — {element}", fontsize=13)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_bond_length_distribution(
    lengths: np.ndarray,
    pair_label: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """KDE + histogram of bond lengths."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    if len(lengths) > 0:
        ax.hist(lengths, bins=40, color="steelblue", alpha=0.6,
                edgecolor="white", density=True)
        _kde_overlay(ax, lengths, float(lengths.min()) * 0.95, float(lengths.max()) * 1.05,
                     color="navy", linewidth=2.0)
    ax.set_xlabel(r"Bond length (Å)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Bond length distribution — {pair_label}", fontsize=13)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_bond_angle_distribution(
    angles_deg: np.ndarray,
    central_label: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of bond angles with ideal angle markers."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    if len(angles_deg) > 0:
        ax.hist(angles_deg, bins=40, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
        _kde_overlay(ax, angles_deg, 0, 180, color="navy", linewidth=2.0)
    ax.axvline(109.47, color="crimson", linewidth=1.5, linestyle="--",
               label="Tetrahedral (109.47°)")
    ax.axvline(90.0, color="darkorange", linewidth=1.2, linestyle=":",
               label="Octahedral (90°)")
    ax.set_xlabel("Bond angle (°)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Bond angle distribution — {central_label}", fontsize=13)
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_fis_q4_scatter(
    fis: np.ndarray,
    q4: np.ndarray,
    env_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Scatter plot F_IS vs q4, optionally coloured by environment label."""
    fig, ax = plt.subplots(figsize=(6, 5))

    if env_labels is not None:
        unique_envs = sorted(set(env_labels))
        palette = cm.get_cmap("tab10", len(unique_envs))
        for idx, env in enumerate(unique_envs):
            mask = np.array([e == env for e in env_labels])
            ax.scatter(q4[mask], fis[mask], s=30, alpha=0.7,
                       color=palette(idx), label=env, edgecolors="none")
        ax.legend(fontsize=8, markerscale=1.2, bbox_to_anchor=(1.01, 1),
                  loc="upper left")
    else:
        ax.scatter(q4, fis, s=30, alpha=0.6, color="steelblue",
                   edgecolors="none")

    xmid = float(np.median(q4)) if len(q4) > 0 else 0.5
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(xmid, color="gray", linewidth=0.8, linestyle="--")

    ax.text(0.98, 0.98, "high BOO\nhigh inversion", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray")
    ax.text(0.02, 0.98, "low BOO\nhigh inversion", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="gray")
    ax.text(0.98, 0.02, "high BOO\nlow inversion", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="gray")
    ax.text(0.02, 0.02, "low BOO\nlow inversion", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel(r"$q_4$", fontsize=12)
    ax.set_ylabel(r"$F_\mathrm{IS}$", fontsize=12)
    ax.set_title(r"$F_\mathrm{IS}$ vs $q_4$", fontsize=13)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_convergence(
    log_file: str | Path,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Parse train.log and plot loss convergence (two panels)."""
    log_file = Path(log_file)
    steps: list[int] = []
    losses: list[float] = []
    pct_reductions: list[float] = []

    pattern = re.compile(
        r"Step\s+(\d+).*?Loss\s*=\s*([\d.eE+\-]+)\s*\(?([\d.]+)%\)?"
    )

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
                    pct_reductions.append(float(m.group(3)))
    except FileNotFoundError:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if steps:
        ax1.plot(steps, losses, color="steelblue", linewidth=1.5)
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training loss", fontsize=13)

        ax2.plot(steps, pct_reductions, color="darkorange", linewidth=1.5)
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("% reduction", fontsize=12)
        ax2.set_title("Loss reduction (%)", fontsize=13)
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)

    fig.tight_layout()
    return _save(fig, save_path)


def plot_summary_panel(
    descriptors: "StructureDescriptors",  # noqa: F821
    system_name: str,
    central: str,
    neighbor: str,
    cutoff: float,
    exp_data: Optional[dict] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    6-panel summary figure.

    Layout (row, col)
    -----------------
    (0,0) Bond length distribution
    (0,1) Bond angle distribution
    (1,0) CN distribution (bar chart)
    (1,1) q4 histogram
    (2,0) F_IS histogram
    (2,1) F_IS vs q4 scatter
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f"Structure Analysis — {system_name}", fontsize=15)

    # ---------- (0,0) Bond lengths ----------
    ax = axes[0, 0]
    lengths = descriptors.bond_length_distribution((central, neighbor), cutoff)
    if len(lengths) > 0:
        ax.hist(lengths, bins=40, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
        _kde_overlay(ax, lengths, float(lengths.min()) * 0.95, float(lengths.max()) * 1.05,
                     color="navy", linewidth=2.0)
    ax.set_xlabel(r"Bond length (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{central}-{neighbor} bond lengths", fontsize=12)

    # ---------- (0,1) Bond angles ----------
    ax = axes[0, 1]
    angles = descriptors.bond_angle_distribution(central, neighbor, cutoff)
    if len(angles) > 0:
        ax.hist(angles, bins=40, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
        _kde_overlay(ax, angles, 0, 180, color="navy", linewidth=2.0)
    ax.axvline(109.47, color="crimson", linewidth=1.3, linestyle="--", label="109.47°")
    ax.axvline(90.0, color="darkorange", linewidth=1.2, linestyle=":", label="90°")
    ax.set_xlabel("Angle (°)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{neighbor}-{central}-{neighbor} angles", fontsize=12)
    ax.legend(fontsize=8)

    # ---------- (1,0) CN distribution ----------
    ax = axes[1, 0]
    cn = descriptors.coordination_numbers(central, neighbor, cutoff)
    if len(cn) > 0:
        unique, counts = np.unique(cn, return_counts=True)
        fractions = counts / counts.sum()
        ax.bar([str(u) for u in unique], fractions, color="steelblue",
               edgecolor="white", alpha=0.85)
    ax.set_xlabel("Coordination number", fontsize=11)
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_title(f"CN distribution — {central}", fontsize=12)

    # ---------- order parameters ----------
    # Determine central_z and neighbor_z from atoms
    syms = descriptors.atoms.get_chemical_symbols()
    central_z = int(descriptors.atoms.numbers[next(
        (i for i, s in enumerate(syms) if s == central), 0
    )])
    neighbor_z = None
    nb_idx = next((i for i, s in enumerate(syms) if s == neighbor), None)
    if nb_idx is not None:
        neighbor_z = int(descriptors.atoms.numbers[nb_idx])

    op_data = descriptors.order_params_per_atom(
        central_z=central_z,
        neighbor_z=neighbor_z,
        cutoff=cutoff,
        compute=["q4", "fis"],
    )
    q4_vals = op_data.get("q4", np.array([]))
    fis_vals = op_data.get("fis", np.array([]))

    # ---------- (1,1) q4 histogram ----------
    ax = axes[1, 1]
    if len(q4_vals) > 0:
        ax.hist(q4_vals, bins=30, color="mediumseagreen", alpha=0.75,
                edgecolor="white", density=True)
        _kde_overlay(ax, q4_vals, float(q4_vals.min()), float(q4_vals.max()),
                     color="darkgreen", linewidth=2.0)
    ax.set_xlabel(r"$q_4$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(r"$q_4$ per-atom distribution", fontsize=12)

    # ---------- (2,0) F_IS histogram ----------
    ax = axes[2, 0]
    if len(fis_vals) > 0:
        ax.hist(fis_vals, bins=30, color="coral", alpha=0.75,
                edgecolor="white", density=True)
        _kde_overlay(ax, fis_vals, float(fis_vals.min()), float(fis_vals.max()),
                     color="darkred", linewidth=2.0)
    ax.set_xlabel(r"$F_\mathrm{IS}$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(r"$F_\mathrm{IS}$ per-atom distribution", fontsize=12)

    # ---------- (2,1) F_IS vs q4 scatter ----------
    ax = axes[2, 1]
    if len(fis_vals) > 0 and len(q4_vals) > 0:
        ax.scatter(q4_vals, fis_vals, s=20, alpha=0.5, color="steelblue",
                   edgecolors="none")
    ax.set_xlabel(r"$q_4$", fontsize=11)
    ax.set_ylabel(r"$F_\mathrm{IS}$", fontsize=11)
    ax.set_title(r"$F_\mathrm{IS}$ vs $q_4$", fontsize=12)

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Steinhardt parameter plots
# ---------------------------------------------------------------------------

# Crystalline reference values for Steinhardt parameters
_CRYSTAL_REFS_Q4 = {
    "FCC": 0.191,
    "BCC": 0.036,
    "HCP": 0.097,
    "Diamond": 0.509,
}
_CRYSTAL_REFS_Q6 = {
    "FCC": 0.575,
    "BCC": 0.511,
    "HCP": 0.485,
    "Diamond": 0.629,
}
_CRYSTAL_REFS_W4 = {
    "FCC": -0.159,
    "BCC": +0.013,
}
_CRYSTAL_REFS_W6 = {
    "FCC": -0.013,
    "BCC": +0.013,
}
_REF_COLORS = {
    "FCC": "crimson",
    "BCC": "darkorange",
    "HCP": "purple",
    "Diamond": "forestgreen",
}


def plot_steinhardt_distributions(
    q4: np.ndarray,
    q6: np.ndarray,
    w4: Optional[np.ndarray] = None,
    w6: Optional[np.ndarray] = None,
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Distribution plots for Steinhardt parameters q4, q6 and optionally w4, w6.

    Layout
    ------
    If w4/w6 provided: 2x2 grid (q4, q6, w4, w6).
    Otherwise: 1x2 grid (q4, q6).

    Crystal reference lines are overlaid on each panel.
    """
    has_w = w4 is not None and w6 is not None
    n_rows = 2 if has_w else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = "Steinhardt parameter distributions"
    if system_name:
        title += f" — {system_name}"
    fig.suptitle(title, fontsize=14)

    def _ql_panel(ax: plt.Axes, data: np.ndarray, label: str,
                  refs: dict, x_lim: tuple) -> None:
        if len(data) > 0:
            ax.hist(data, bins=40, color="steelblue", alpha=0.65,
                    edgecolor="white", density=True, range=x_lim)
            _kde_overlay(ax, data, x_lim[0], x_lim[1], color="navy", linewidth=2.0)
            q25 = float(np.percentile(data, 25))
            q75 = float(np.percentile(data, 75))
            if q75 > q25:
                ax.axvspan(q25, q75, alpha=0.12, color="steelblue", label="IQR (amorphous)")
        for name, val in refs.items():
            color = _REF_COLORS.get(name, "gray")
            ax.axvline(val, color=color, linewidth=1.4, linestyle="--",
                       label=f"{name}={val:.3f}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"P({label})", fontsize=12)
        ax.set_xlim(x_lim)
        ax.legend(fontsize=7, loc="upper right")

    def _wl_panel(ax: plt.Axes, data: np.ndarray, label: str,
                  refs: dict) -> None:
        x_lim = (-0.2, 0.2)
        if len(data) > 0:
            ax.hist(data, bins=40, color="coral", alpha=0.65,
                    edgecolor="white", density=True, range=x_lim)
            _kde_overlay(ax, data, x_lim[0], x_lim[1], color="darkred", linewidth=2.0)
        for name, val in refs.items():
            color = _REF_COLORS.get(name, "gray")
            ax.axvline(val, color=color, linewidth=1.4, linestyle="--",
                       label=f"{name}={val:+.3f}")
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"P({label})", fontsize=12)
        ax.set_xlim(x_lim)
        ax.legend(fontsize=7, loc="upper right")

    _ql_panel(axes[0, 0], q4, r"$q_4$", _CRYSTAL_REFS_Q4, (0.0, 0.8))
    _ql_panel(axes[0, 1], q6, r"$q_6$", _CRYSTAL_REFS_Q6, (0.0, 0.8))

    if has_w:
        _wl_panel(axes[1, 0], w4, r"$w_4$", _CRYSTAL_REFS_W4)
        _wl_panel(axes[1, 1], w6, r"$w_6$", _CRYSTAL_REFS_W6)

    fig.tight_layout()
    return _save(fig, save_path)


def plot_disorder_heatmap(
    q4: np.ndarray,
    q6: np.ndarray,
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    2D density map of (q4, q6) per atom — the key disorder diagnostic.

    A spreading cloud around (0, 0.5) indicates amorphous/disordered
    material; tight clusters near crystal reference points indicate order.

    Crystal reference structures are overlaid as labelled star markers.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    title = "Steinhardt disorder map"
    if system_name:
        title += f" — {system_name}"
    ax.set_title(title, fontsize=13)

    if len(q4) > 0 and len(q6) > 0:
        _h, _xedges, _yedges, img = ax.hist2d(
            q4, q6,
            bins=50,
            range=[[0.0, 0.8], [0.0, 0.8]],
            cmap="hot_r",
            density=True,
        )
        cbar = fig.colorbar(img, ax=ax, shrink=0.85)
        cbar.set_label("Atom density", fontsize=11)

    # Crystal reference markers
    crystal_refs_local = {
        "FCC": (0.191, 0.575),
        "BCC": (0.036, 0.511),
        "HCP": (0.097, 0.485),
        "Diamond": (0.509, 0.629),
        "Icos.": (0.0, 0.663),
    }
    ref_colors_local = {
        "FCC": "crimson",
        "BCC": "darkorange",
        "HCP": "purple",
        "Diamond": "forestgreen",
        "Icos.": "royalblue",
    }
    for name, (x, y) in crystal_refs_local.items():
        color = ref_colors_local.get(name, "gray")
        ax.plot(x, y, marker="*", markersize=14, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.annotate(
            name, xy=(x, y), xytext=(x + 0.015, y + 0.012),
            fontsize=8, color=color, fontweight="bold",
            zorder=6,
        )

    ax.set_xlabel(r"$q_4$", fontsize=13)
    ax.set_ylabel(r"$q_6$", fontsize=13)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.8)

    fig.tight_layout()
    return _save(fig, save_path)


def plot_q4_q6_trajectory(
    q4_history: list,
    q6_history: list,
    steps: list,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Show how the mean (q4, q6) evolves during training as an arrow path
    overlaid on the disorder map background.

    Parameters
    ----------
    q4_history : list of np.ndarray
        Per-atom q4 arrays at each saved step.
    q6_history : list of np.ndarray
        Per-atom q6 arrays at each saved step.
    steps : list of int
        Step indices (for colorbar labeling).
    save_path : str or Path, optional
        Where to save the figure.
    """
    mean_q4 = np.array([float(np.mean(q)) for q in q4_history])
    mean_q6 = np.array([float(np.mean(q)) for q in q6_history])
    n = len(steps)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Training trajectory in (q4, q6) space", fontsize=13)

    cmap_traj = cm.get_cmap("viridis", n)
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(mean_q4[i + 1], mean_q6[i + 1]),
            xytext=(mean_q4[i], mean_q6[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=cmap_traj(i / max(n - 1, 1)),
                lw=1.8,
            ),
        )

    sc = ax.scatter(mean_q4, mean_q6, c=steps, cmap="viridis",
                    s=40, zorder=5, edgecolors="white", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Training step", fontsize=11)

    crystal_refs_traj = {
        "FCC": (0.191, 0.575),
        "BCC": (0.036, 0.511),
        "HCP": (0.097, 0.485),
        "Diamond": (0.509, 0.629),
        "Icos.": (0.0, 0.663),
    }
    for name, (x, y) in crystal_refs_traj.items():
        ax.plot(x, y, marker="*", markersize=12, color="gray",
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.annotate(name, xy=(x, y), xytext=(x + 0.015, y + 0.012),
                    fontsize=7, color="gray", zorder=4)

    ax.set_xlabel(r"$q_4$", fontsize=13)
    ax.set_ylabel(r"$q_6$", fontsize=13)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.8)

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# F_IS grouped by coordination number
# ---------------------------------------------------------------------------

def plot_fis_by_cn(
    fis_by_cn: dict[int, np.ndarray],
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Violin plot of F_IS distribution per coordination number.

    Theoretical reference lines:
      CN=4 (tetrahedral)  → F_IS = −1/3
      CN=6 (octahedral)   → F_IS = +1
    """
    if not fis_by_cn:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, save_path)

    cn_vals = sorted(fis_by_cn.keys())
    data = [fis_by_cn[cn] for cn in cn_vals]
    labels = [f"CN={cn}\n(n={len(fis_by_cn[cn])})" for cn in cn_vals]

    fig, ax = plt.subplots(figsize=(max(5, len(cn_vals) * 1.5), 4.5))

    parts = ax.violinplot(data, positions=range(len(cn_vals)), showmedians=True,
                          showextrema=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("#C44E52")
    parts["cmedians"].set_linewidth(2)

    # Theoretical reference lines
    theo = {4: -1 / 3, 6: 1.0}
    colors_theo = {4: "#2ca02c", 6: "#ff7f0e"}
    for cn_ref, fis_ref in theo.items():
        if cn_ref in fis_by_cn:
            pos = cn_vals.index(cn_ref)
            ax.hlines(fis_ref, pos - 0.4, pos + 0.4,
                      colors=colors_theo[cn_ref], linewidths=2.5,
                      linestyles="--",
                      label=f"CN={cn_ref} theory ({fis_ref:.2f})")

    ax.set_xticks(range(len(cn_vals)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$F_{IS}$", fontsize=13)
    ax.set_xlabel("Coordination number", fontsize=12)
    title = f"$F_{{IS}}$ by CN — {system_name}" if system_name else r"$F_{IS}$ by CN"
    ax.set_title(title, fontsize=13)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    if any(cn in fis_by_cn for cn in theo):
        ax.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Polyhedral distortion index
# ---------------------------------------------------------------------------

def plot_distortion_index(
    di: np.ndarray,
    cn: np.ndarray | None = None,
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of the polyhedral bond-length distortion index DI.

    If ``cn`` is provided, draws one histogram per CN group.
    """
    if di is None or len(di) == 0:
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, save_path)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    bins = np.linspace(0, max(di.max() * 1.1, 0.1), 40)

    if cn is not None and len(cn) == len(di):
        cn_uniq = sorted(np.unique(cn))
        colors_di = cm.tab10(np.linspace(0, 0.8, len(cn_uniq)))
        for k, cn_val in enumerate(cn_uniq):
            mask = cn == cn_val
            ax.hist(di[mask], bins=bins, alpha=0.6, label=f"CN={int(cn_val)}",
                    color=colors_di[k], edgecolor="white", linewidth=0.4)
        ax.legend(fontsize=9)
    else:
        ax.hist(di, bins=bins, color="#4C72B0", alpha=0.75,
                edgecolor="white", linewidth=0.4)

    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Distortion index DI", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    title = (f"Polyhedral distortion — {system_name}"
             if system_name else "Polyhedral distortion")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Warren-Cowley short-range order
# ---------------------------------------------------------------------------

def plot_warren_cowley(
    alpha: np.ndarray,
    elements: list[str],
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of Warren-Cowley SRO parameters α_ij.

    α > 0 (red)  : like-atom clustering / avoidance of j by i
    α < 0 (blue) : chemical ordering / preference for j by i
    α = 0        : random mixing (white)
    """
    n = len(elements)
    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(3.5, n + 0.5)))

    vmax = max(0.3, np.abs(alpha).max())
    im = ax.imshow(alpha, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r"$\alpha_{ij}$", fontsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(elements, fontsize=11)
    ax.set_yticklabels(elements, fontsize=11)
    ax.set_xlabel("Neighbor species j", fontsize=11)
    ax.set_ylabel("Central species i", fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{alpha[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(alpha[i, j]) < vmax * 0.6 else "white")

    title = (f"Warren-Cowley SRO — {system_name}"
             if system_name else "Warren-Cowley SRO")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# F_IS spatial autocorrelation
# ---------------------------------------------------------------------------

def plot_fis_spatial_autocorrelation(
    r_centers: np.ndarray,
    C_r: np.ndarray,
    system_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Spatial autocorrelation C(r) of the F_IS field.

    C(r) = <δF_IS(0) δF_IS(r)> / <δF_IS²>

    C(r)≈1 : short-range same-symmetry clustering
    C(r)≈0 : uncorrelated at distance r
    The decay length characterises the spatial extent of local structural motifs.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(r_centers, C_r, color="#4C72B0", lw=2.0, marker="o",
            markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.fill_between(r_centers, 0, C_r, where=(C_r > 0),
                    alpha=0.15, color="#4C72B0")
    ax.fill_between(r_centers, 0, C_r, where=(C_r < 0),
                    alpha=0.15, color="#C44E52")

    # Estimate correlation length (first zero crossing)
    zero_crossings = np.where(np.diff(np.sign(C_r)))[0]
    if len(zero_crossings) > 0:
        r0 = r_centers[zero_crossings[0]]
        ax.axvline(r0, color="#C44E52", lw=1.2, ls=":", alpha=0.7,
                   label=f"First zero: {r0:.2f} Å")
        ax.legend(fontsize=9)

    ax.set_xlabel(r"$r$ (Å)", fontsize=12)
    ax.set_ylabel(r"$C(r)$", fontsize=12)
    title = (f"$F_{{IS}}$ spatial autocorrelation — {system_name}"
             if system_name else r"$F_{IS}$ spatial autocorrelation")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, save_path)
