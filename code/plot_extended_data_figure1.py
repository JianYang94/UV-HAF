"""Redraw Extended Data Figure 1 from archived source data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}


def smooth11(values):
    values = np.asarray(values, float)
    return np.convolve(np.pad(values, (5, 5), mode="edge"),
                       np.ones(11) / 11, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = (args.output_dir or repo / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)

    trajectory_rows = list(csv.DictReader(
        (repo / "data" / "derived" / "Figure2_source_data.csv").open(encoding="utf-8-sig")))
    sensitivity_rows = list(csv.DictReader(
        (repo / "data" / "derived" / "Extended_Data_Figure_1_source_data.csv").open(encoding="utf-8-sig")))

    trajectories = {}
    for scenario in SCENARIOS:
        rows = sorted((r for r in trajectory_rows if r["scenario"] == scenario),
                      key=lambda r: int(r["year"]))
        trajectories[scenario] = {
            "year": np.array([int(r["year"]) for r in rows]),
            "ozone": smooth11([float(r["ozone_change_pct"]) for r in rows]),
            "uv": smooth11([float(r["uv_change_pct"]) for r in rows]),
        }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.4, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.7,
        "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.45))
    fig.subplots_adjust(left=.125, right=.985, top=.965, bottom=.105,
                        wspace=.39, hspace=.32)
    # Reading order is vertical within each column: a,b on the left; c,d on the right.
    axa, axc, axb, axd = axs.flat
    for ax in axs.flat:
        ax.tick_params(direction="out", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # a: ozone--UV response trajectories.
    for scenario in SCENARIOS:
        d = trajectories[scenario]
        axa.plot(d["ozone"], d["uv"], color=COLORS[scenario], lw=1.45,
                 label=LABELS[scenario])
        for year in (2050, 2100):
            idx = int(np.where(d["year"] == year)[0][0])
            axa.scatter(d["ozone"][idx], d["uv"][idx], s=18,
                        facecolor=COLORS[scenario], edgecolor="white",
                        linewidth=.45, zorder=4)
        # One arrow shows chronological direction without adding more text.
        i0, i1 = 54, 59
        axa.annotate("", xy=(d["ozone"][i1], d["uv"][i1]),
                     xytext=(d["ozone"][i0], d["uv"][i0]),
                     arrowprops=dict(arrowstyle="->", color=COLORS[scenario],
                                     lw=.85, mutation_scale=7))
    axa.axhline(0, color="#888", lw=.55); axa.axvline(0, color="#888", lw=.55)
    axa.set_xlabel("Estimated total-column ozone change (%)")
    axa.set_ylabel("Erythemal UV dose change (%)")
    axa.text(.40, 6.30, "Ozone alone does not determine\nsurface UV",
             ha="left", va="top", fontsize=6.2,
             linespacing=1.05)
    axa.legend(frameon=True, fancybox=False, ncol=3, loc="lower right",
               edgecolor="#333", framealpha=.96, borderpad=.35,
               handlelength=1.8, columnspacing=.8, handletextpad=.45)

    # c: sensitivity to UV weight (original panel c).
    weight_rows = [r for r in sensitivity_rows if r["panel"] == "weight"]
    for scenario in SCENARIOS:
        rows = sorted((r for r in weight_rows if r["scenario"] == scenario),
                      key=lambda r: float(r["x"]))
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["delta_haf"]) for r in rows])
        axc.plot(x, y, color=COLORS[scenario], lw=1.45, marker="o", ms=3,
                 label=LABELS[scenario])
    critic = max(float(r["x"]) for r in weight_rows)
    axc.axhline(0, color="#777", lw=.55)
    axc.axvline(critic, color="#666", lw=.7, ls=(0, (3, 2)))
    axc.text(critic-.004, -.035, "CRITIC", rotation=90, ha="right", va="top",
             fontsize=5.8, color="#555")
    axc.set_ylim(-.190, .008)
    axc.set_xlabel("UV weight"); axc.set_ylabel("Change in HAF")
    axc.text(.030, -.122, "HAF reduction increases\nwith UV weight",
             ha="left", va="bottom", fontsize=6.2,
             linespacing=1.05)
    axc.legend(frameon=True, fancybox=False, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .012),
               edgecolor="#333", framealpha=.96, borderpad=.35,
               handlelength=1.8, columnspacing=.8, handletextpad=.45)

    # b: controlled ablation (original panel b).
    ablation = [r for r in sensitivity_rows if r["panel"] == "fixed_chs7_uv_ablation"]
    ordered = [(period, scenario) for period in ("mid", "late") for scenario in SCENARIOS]
    row_lookup = {(r["period"], r["scenario"]): r for r in ablation}
    y_pos = np.arange(len(ordered))
    y_labels = []
    for y, (period, scenario) in zip(y_pos, ordered):
        row = row_lookup[(period, scenario)]
        removed = float(row["haf_without_direct_uv"])
        full = float(row["haf_full_chs7"])
        axb.plot([full, removed], [y, y], color="#999", lw=.9, zorder=1)
        axb.scatter(removed, y, s=26, facecolor="white", edgecolor=COLORS[scenario],
                    linewidth=.9, zorder=3)
        axb.scatter(full, y, s=26, facecolor=COLORS[scenario],
                    edgecolor=COLORS[scenario], linewidth=.6, zorder=4)
        y_labels.append(LABELS[scenario])
    axb.set_yticks(y_pos, y_labels); axb.invert_yaxis(); axb.set_xlim(0, 1)
    # Leave visible clearance around both grouping rectangles so that their
    # outer horizontal edges are not clipped by the axes boundary.
    axb.set_ylim(5.65, -.65)
    axb.set_xlabel("Habitable Area Fraction")
    # Group the three SSP comparisons in each time slice with restrained,
    # publication-style dashed rectangles.
    axb.text(.515, .95, "Mid-century\n2041–2060",
             ha="left", va="center", fontsize=6.0, color="#333",
             linespacing=1.05)
    axb.text(.22, 3.85, "Late-century\n2081–2100",
             ha="left", va="center", fontsize=6.0, color="#333",
             linespacing=1.05)
    from matplotlib.patches import Rectangle
    dash = (0, (3, 2))
    axb.add_patch(Rectangle((.50, -.36), .445, 2.72, fill=False,
                            ec="#222", lw=.75, ls=dash, zorder=5))
    axb.add_patch(Rectangle((.055, 2.64), .89, 2.72, fill=False,
                            ec="#222", lw=.75, ls=dash, zorder=5))
    axb.text(.10, 1.50, "Direct UV lowers HAF\nin every comparison",
             ha="left", va="center", fontsize=6.2,
             linespacing=1.05)
    axb.legend(handles=[
        Line2D([0], [0], marker="o", lw=0, mfc="white", mec="#444",
               markersize=np.sqrt(26), markeredgewidth=.9,
               label="UV term removed"),
        Line2D([0], [0], marker="o", lw=0, mfc="#444", mec="#444",
               markersize=np.sqrt(26), markeredgewidth=.6,
               label="Full CHS$_7$")],
        frameon=True, fancybox=False, loc="upper left", ncol=1,
        edgecolor="#333", bbox_to_anchor=(.05, .965),
        framealpha=.96, borderpad=.30, columnspacing=.7,
        handletextpad=.35)

    # d: threshold sensitivity.
    threshold_rows = [r for r in sensitivity_rows if r["panel"] == "threshold"]
    for scenario in SCENARIOS:
        rows = sorted((r for r in threshold_rows if r["scenario"] == scenario),
                      key=lambda r: float(r["x"]))
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["delta_haf"]) for r in rows])
        axd.plot(x, y, color=COLORS[scenario], lw=1.45, marker="o", ms=3,
                 label=LABELS[scenario])
    axd.axhline(0, color="#777", lw=.55)
    axd.axvline(90, color="#666", lw=.7, ls=(0, (3, 2)))
    axd.text(90, -.040, "90th percentile threshold", rotation=90,
             ha="right", va="top",
             fontsize=5.8, color="#555")
    axd.set_xlabel("Baseline threshold percentile"); axd.set_ylabel("Change in HAF")
    axd.text(80, -.125, "The direct UV effect remains\nnegative across thresholds",
             ha="left", va="bottom", fontsize=6.2,
             linespacing=1.05)
    axd.legend(frameon=True, fancybox=False, loc="upper left", ncol=3,
               bbox_to_anchor=(.01, .945),
               edgecolor="#333", framealpha=.96, borderpad=.35,
               handlelength=1.8, columnspacing=.8, handletextpad=.45)

    for letter, ax in (("a", axa), ("b", axb), ("c", axc), ("d", axd)):
        ax.text(-.13, 1.035, letter, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", va="bottom")

    stem = "Extended_Data_Figure_1"
    fig.savefig(out / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    print(out / f"{stem}.png")
    print(out / f"{stem}.pdf")


if __name__ == "__main__":
    main()
