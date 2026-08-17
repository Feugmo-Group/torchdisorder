"""Plot per-atom order-parameter distributions against a reference model.

The companion figure to ``compare_order_params.py``. That script reports means and
standard deviations; this one draws the distributions behind them, which matters
because a mean can sit on the reference while the distribution underneath is a
different shape entirely -- a partly crystalline melt-quench shows up as a distinct
second population in q4 long before its mean drifts by a full sigma.

One row per system, one panel per order parameter, reference filled and each test
structure overlaid. Every panel is annotated with the shift of the *first* test
structure from the reference, in units of the reference's own spread.

Each option is repeatable: give it once per row.

Usage:
    # default: the melt-quench crystal -> glass validation for both oxides
    poetry run python scripts/plot_order_param_distributions.py

    # a single system of your own
    poetry run python scripts/plot_order_param_distributions.py \
        --reference data/crystal-structures/sio2_glass_gap.cif \
        --test outputs/<run>/final_results/final_structure.cif \
        --labels refined --central Si --neighbour O --cutoff 2.2 \
        --out Tutorials/plots/my_run.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_order_params import OPS, measure

TITLES = {
    "cn": "coordination number",
    "tet": "tetrahedral order",
    "q4": "$q_4$",
    "q6": "$q_6$",
    "fis": "$F_{\\mathrm{IS}}$",
}

REF_COLOUR = "#1d3557"
# First entry is the structure the delta annotation refers to, so it leads.
TEST_COLOURS = ["#c1121f", "#8d99ae", "#2a9d8f", "#e09f3e"]

DATA = Path("data/crystal-structures")

# The melt-quench validation: SiO2 with its failed gamma=0.1 predecessor drawn
# alongside, since the q4 separation between the two is the whole result.
DEFAULT_ROWS = [
    dict(
        name="SiO$_2$", central="Si", neighbour="O", cutoff=2.2,
        reference=DATA / "sio2_glass_gap.cif", ref_label="published GAP model",
        tests=[DATA / "SiO2_mq_hot.cif", DATA / "SiO2_mq.cif"],
        labels=["melt-quench, $\\gamma$=1.0 (job 1210)",
                "melt-quench, $\\gamma$=0.1 (job 1203, failed)"],
    ),
    dict(
        name="GeO$_2$", central="Ge", neighbour="O", cutoff=2.4,
        reference=DATA / "geo2_glass_nnp.cif", ref_label="published NNP model",
        tests=[DATA / "GeO2_mq.cif"],
        labels=["melt-quench (job 1203)"],
    ),
]

DEFAULT_TITLE = ("Melt-quench crystal $\\rightarrow$ glass, scored against published "
                 "models on quantities the route never fits")


def draw(ax, values, label, colour, filled=False):
    """One distribution. Percentile limits keep a stray outlier from eating the axis."""
    lo, hi = np.percentile(values, [0.2, 99.8])
    if hi - lo < 1e-9:
        lo, hi = values.mean() - 0.05, values.mean() + 0.05
    bins = np.linspace(lo, hi, 45)
    if filled:
        ax.hist(values, bins=bins, density=True, color=colour, alpha=0.35,
                label=label, edgecolor="none")
    else:
        ax.hist(values, bins=bins, density=True, histtype="step", lw=1.9,
                color=colour, label=label)
    ax.axvline(values.mean(), color=colour, ls="--", lw=1.0, alpha=0.8)


def build_rows(args):
    if args.reference is None:
        return DEFAULT_ROWS

    n = len(args.reference)

    def per_row(values, what, default=None):
        if values is None:
            return [default] * n
        if len(values) != n:
            raise SystemExit(f"--{what} given {len(values)} times, --reference {n}")
        return values

    centrals = per_row(args.central, "central", "Si")
    neighbours = per_row(args.neighbour, "neighbour", "O")
    cutoffs = per_row(args.cutoff, "cutoff", 2.2)
    tests = per_row(args.test, "test")
    labels = per_row(args.labels, "labels")
    names = per_row(args.names, "names")

    rows = []
    for i in range(n):
        if tests[i] is None:
            raise SystemExit("--test is required once per --reference")
        lab = labels[i] or [Path(t).stem for t in tests[i]]
        if len(lab) != len(tests[i]):
            raise SystemExit("--labels must match the --test entries in its row")
        rows.append(dict(
            name=names[i] or f"{centrals[i]}-centred",
            central=centrals[i], neighbour=neighbours[i], cutoff=cutoffs[i],
            reference=Path(args.reference[i]), ref_label="reference",
            tests=[Path(t) for t in tests[i]], labels=lab,
        ))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", action="append",
                   help="published / trusted model; repeat once per row")
    p.add_argument("--test", action="append", nargs="+",
                   help="structures to overlay; repeat once per row")
    p.add_argument("--labels", action="append", nargs="+")
    p.add_argument("--names", action="append", help="row label, e.g. SiO2")
    p.add_argument("--central", action="append")
    p.add_argument("--neighbour", action="append")
    p.add_argument("--cutoff", action="append", type=float)
    p.add_argument("--max-neighbors", type=int, default=8)
    p.add_argument("--title", default=None)
    p.add_argument("--out", type=Path,
                   default=Path("outputs/melt_quench_validation.png"))
    args = p.parse_args()

    from ase.data import atomic_numbers

    rows = build_rows(args)
    fig, axes = plt.subplots(len(rows), len(OPS),
                             figsize=(19, 3.6 * len(rows)), squeeze=False)
    summary = []

    for r, row in enumerate(rows):
        cz = atomic_numbers[row["central"]]
        nz = atomic_numbers[row["neighbour"]]
        cut, mn = row["cutoff"], args.max_neighbors

        ref, n_ref = measure(row["reference"], cz, nz, cut, mn)
        print(f"{row['central']}: reference {Path(row['reference']).name} "
              f"({n_ref} centres, cutoff {cut} A)")

        tests = []
        for i, (label, path) in enumerate(zip(row["labels"], row["tests"], strict=True)):
            got, n = measure(path, cz, nz, cut, mn)
            tests.append((label, got, TEST_COLOURS[i % len(TEST_COLOURS)], n))
            print(f"   {label}: {n} centres")

        for c, op in enumerate(OPS):
            ax = axes[r][c]
            draw(ax, ref[op], f"{row['ref_label']} (n={n_ref})", REF_COLOUR, filled=True)
            for label, got, colour, n in tests:
                draw(ax, got[op], f"{label} (n={n})", colour)

            ax.set_title(TITLES[op], fontsize=11)
            ax.set_yticks([])
            ax.tick_params(labelsize=8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)

            d = tests[0][1][op].mean() - ref[op].mean()
            sigma = max(ref[op].std(), 1e-9)
            ax.text(0.03, 0.94, f"$\\Delta$ = {d:+.4f}\n({abs(d) / sigma:.2f}$\\sigma$)",
                    transform=ax.transAxes, va="top", fontsize=8.5,
                    color="#2a9d3f" if abs(d) <= sigma else "#c1121f",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
            summary.append((row["central"], op, ref[op].mean(), sigma,
                            tests[0][1][op].mean(), d, d / sigma))

        axes[r][0].set_ylabel(row["name"], fontsize=15, labelpad=12)
        # Under the delta annotation, clear of the sharp CN spike at 4.
        axes[r][0].legend(fontsize=7.6, loc="upper left",
                          bbox_to_anchor=(-0.02, 0.80), frameon=False)

    title = args.title or (DEFAULT_TITLE if args.reference is None else
                           "Order-parameter distributions vs reference")
    fig.suptitle(title, fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "Dashed lines mark distribution means. $\\Delta$ is the first test "
             "structure's mean minus the reference mean, in units of the reference's "
             "own $\\sigma$; green = inside the reference spread.",
             ha="center", fontsize=8.6, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"\nwrote {args.out}")
    print(f"wrote {args.out.with_suffix('.pdf')}")

    print("\n{:<4} {:<5} {:>12} {:>10} {:>12} {:>10} {:>8}".format(
        "el", "op", "reference", "sigma", "test", "delta", "n_sigma"))
    for el, op, rm, rs, tm, d, ns in summary:
        print(f"{el:<4} {op:<5} {rm:>+12.4f} {rs:>10.4f} {tm:>+12.4f} "
              f"{d:>+10.4f} {ns:>8.2f}")


if __name__ == "__main__":
    main()
