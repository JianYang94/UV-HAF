"""Redraw Supplementary Figure S1 from its archived source data.

The panel order follows the final SI layout: a/b in the left column and c/d
in the right column.  The former panels b and c are deliberately exchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import transforms


VARS = ["gmst", "co2", "sst", "ohc", "ph", "omega", "uv_erythema"]
VLABELS = ["GMST", "CO$_2$", "SST", "OHC", "pH", "$\\Omega_{arag}$", "UV"]
METHODS = ("equal", "critic", "entropy", "pca")
MLABELS = {"equal": "Equal", "critic": "CRITIC", "entropy": "Entropy", "pca": "PCA"}
SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}
METHOD_COLORS = ["#777777", "#0072B2", "#E69F00", "#009E73"]


def smooth11(values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.convolve(np.pad(values, (5, 5), mode="edge"), np.ones(11) / 11, mode="valid")


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.4,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    })


def boxed_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    ax.tick_params(direction="out", top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / "data" / "derived" / "Supplementary_Figure_S1_source_data.json"
    output = (args.output or (root / "figures")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(source.read_text(encoding="utf-8"))

    set_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.45))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.09,
                        wspace=0.34, hspace=0.28)
    axa, axc, axb, axd = axs.flat  # visual order: left a-b; right c-d
    for ax in axs.flat:
        boxed_axes(ax)

    # a | Independently estimated CRITIC weights for CHS6 and CHS7.
    x = np.arange(len(VARS))
    width = 0.36
    w6 = data["six_variable_critic"]
    w7 = data["seven_variable_weights"]["critic"]
    axa.bar(x - width / 2, [w6.get(k, 0.0) for k in VARS], width,
            color="#777777", label="CHS$_6$")
    axa.bar(x + width / 2, [w7[k] for k in VARS], width,
            color="#0072B2", label="CHS$_7$")
    axa.set_xticks(x, VLABELS)
    axa.set_ylabel("Weight")
    axa.set_ylim(0, 0.48)
    axa.legend(frameon=True, fancybox=False, ncol=2, loc="upper left",
               edgecolor="black", framealpha=1.0, borderpad=0.35,
               handlelength=1.8, columnspacing=0.9)
    axa.text(-0.35, 0.300, "Including UV substantially\nreduces the OHC weight\nfrom 0.443 to 0.182",
             ha="left", va="top", fontsize=6.2, linespacing=1.05)

    # b | HAF in 2100 under SSP2-4.5 for each weighting scheme.
    vals = [data["haf2100_ssp245_by_method"][m] for m in METHODS]
    xb = np.arange(len(METHODS))
    axb.bar(xb, vals, width=0.62, color=METHOD_COLORS,
            label=[MLABELS[m] for m in METHODS])
    for i, value in enumerate(vals):
        axb.text(i, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=7.0)
    axb.set_xticks(xb, [MLABELS[m] for m in METHODS])
    axb.set_ylabel("HAF in 2100 (SSP2-4.5)")
    axb.set_ylim(0, 0.80)
    axb.legend(frameon=True, fancybox=False, ncol=2, loc="upper left",
               edgecolor="black", framealpha=1.0, borderpad=0.35,
               handlelength=1.8, columnspacing=0.9)
    axb.text(2.40, 0.735, "HAF ranges from 0.527 to 0.583\nacross weighting methods",
             ha="center", va="center", fontsize=6.2, linespacing=1.05)

    # c | CHS7 weights under four objective weighting schemes.
    width_c = 0.18
    for j, method in enumerate(METHODS):
        axc.bar(x + (j - 1.5) * width_c,
                [data["seven_variable_weights"][method][k] for k in VARS],
                width_c, color=METHOD_COLORS[j], label=MLABELS[method])
    axc.axhline(1 / 7, color="#666666", lw=0.75, ls=(0, (3, 2)))
    axc.text(-0.48, 1 / 7 + 0.004, "Equal weight",
             ha="left", va="bottom", fontsize=6.2,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=5)
    axc.set_xticks(x, VLABELS)
    axc.set_ylabel("Weight")
    axc.set_ylim(0, 0.35)
    axc.legend(frameon=True, fancybox=False, ncol=2, loc="upper left",
               edgecolor="black", framealpha=1.0, borderpad=0.35,
               handlelength=1.8, columnspacing=0.9)
    axc.text(0.00, 0.225, "CRITIC assigns the largest weight to UV (0.306)",
             ha="left", va="center", fontsize=6.2, linespacing=1.05)

    # d | Independently weighted CHS6 and CHS7 HAF trajectories.
    years = np.asarray(data["years"], dtype=int)
    phases = [
        (2015, 2025, "Reference period", "#E4E4E4"),
        (2041, 2060, "Mid-century", "#EAF1F4"),
        (2081, 2100, "Late-century", "#F6E9EC"),
    ]
    for start, end, title, color in phases:
        axd.axvspan(start, end, color=color, zorder=0)
        axd.axvline(start, color="#AAAAAA", lw=0.55, ls=(0, (2, 2)), zorder=1)
        axd.axvline(end, color="#AAAAAA", lw=0.55, ls=(0, (2, 2)), zorder=1)
        axd.text((start + end) / 2, 1.025, title,
                 transform=axd.get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=6.3, clip_on=False, linespacing=0.9)

    haf7_smoothed = {scenario: smooth11(data["haf7"][scenario]) for scenario in SCENARIOS}
    haf6_smoothed = {scenario: smooth11(data["haf6"][scenario]) for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        axd.plot(years, haf7_smoothed[scenario], color=COLORS[scenario],
                 lw=1.45, label=LABELS[scenario], zorder=3)
        axd.plot(years, haf6_smoothed[scenario], color=COLORS[scenario],
                 lw=1.0, ls=(0, (3, 2)), zorder=3)

    scenario_handles = [Line2D([0], [0], color=COLORS[s], lw=1.45, label=LABELS[s])
                        for s in SCENARIOS]
    scenario_legend = axd.legend(handles=scenario_handles, frameon=True, fancybox=False,
                                 ncol=3, loc="upper left", bbox_to_anchor=(0.012, 0.985),
                                 edgecolor="black", framealpha=1.0,
                                 borderaxespad=0, borderpad=0.35,
                                 handlelength=1.8, columnspacing=0.8)
    axd.add_artist(scenario_legend)
    style_handles = [
        Line2D([0], [0], color="#333333", lw=1.45, label="CHS$_7$"),
        Line2D([0], [0], color="#333333", lw=1.0, ls=(0, (3, 2)), label="CHS$_6$"),
    ]
    axd.legend(handles=style_handles, frameon=False, loc="lower left",
               bbox_to_anchor=(0.02, 0.02), borderaxespad=0, handlelength=2.0)
    axd.set_xlim(2015, 2100)
    # Reserve separate vertical bands for the legend and phase annotations.
    axd.set_ylim(0, 1.35)
    axd.set_yticks(np.arange(0, 1.01, 0.2))
    axd.set_xticks([2020, 2040, 2060, 2080, 2100])
    axd.set_xlabel("Year")
    axd.set_ylabel("Habitable Area Fraction")

    axd.text(2050.5, 1.03, "HAF continues\nto decline",
             fontsize=6.2, ha="center", va="center", linespacing=1.05, zorder=6)
    axd.text(2090.5, 1.03, "HAF declines\nfastest under\nSSP5-8.5",
             fontsize=6.2, ha="center", va="center", linespacing=1.05, zorder=6)
    axd.text(2070.5, 0.15, "Index design\nreverses the\nHAF difference\nunder SSP5-8.5",
             fontsize=6.2, ha="center", va="center", linespacing=1.05, zorder=6)

    # Match Figure 2; for SSP2-4.5, place the arrow at the largest CHS7-CHS6
    # separation within 2060-2080 rather than selecting the year manually.
    search_mask = (years >= 2060) & (years <= 2080)
    ssp245_gap = np.abs(haf7_smoothed["ssp245"] - haf6_smoothed["ssp245"])
    ssp245_year = int(years[search_mask][np.argmax(ssp245_gap[search_mask])])
    gap_years = {"ssp126": 2070, "ssp245": ssp245_year, "ssp585": 2075}
    for scenario in SCENARIOS:
        arrow_year = gap_years[scenario]
        y7 = float(np.interp(arrow_year, years, haf7_smoothed[scenario]))
        y6 = float(np.interp(arrow_year, years, haf6_smoothed[scenario]))
        arrow_options = dict(arrowstyle="<->", color=COLORS[scenario],
                             lw=0.9, shrinkA=0, shrinkB=0)
        axd.annotate("", xy=(arrow_year, y6), xytext=(arrow_year, y7),
                     zorder=7,
                     arrowprops=arrow_options)

    # Column-wise lettering: left column a-b, right column c-d.
    for letter, ax in zip("acbd", axs.flat):
        ax.text(-0.13, 1.035, letter, transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", va="bottom")

    stem = "Supplementary_Figure_S1"
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
