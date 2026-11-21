"""Redraw manuscript Figure 2 with data-derived phase and process annotations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}


def load_data(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    fields = ("ozone_change_pct", "uv_change_pct", "chs_anomaly",
              "haf_with_uv", "haf_without_uv")
    out = {}
    for scenario in SCENARIOS:
        subset = sorted((r for r in rows if r["scenario"] == scenario),
                        key=lambda r: int(r["year"]))
        out[scenario] = {"year": np.array([int(r["year"]) for r in subset])}
        for field in fields:
            out[scenario][field] = np.array([float(r[field]) for r in subset])
    return out


def smooth11(values):
    values = np.asarray(values, float)
    return np.convolve(np.pad(values, (5, 5), mode="edge"),
                       np.ones(11) / 11, mode="valid")


def y_at(series, years, year):
    return float(series[np.where(years == year)[0][0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = (args.output_dir or repo / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = repo / "data" / "derived" / "Figure2_source_data.csv"
    if not source.exists():
        source = repo / "figures" / "Figure2_source_data.csv"
    data = load_data(source)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.4, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 5.8,
        "axes.linewidth": 0.65, "xtick.major.width": 0.65,
        "ytick.major.width": 0.65, "xtick.major.size": 3,
        "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.55), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.875, bottom=0.095,
                        wspace=0.20, hspace=0.28)
    axa, axb, axc, axd = axs.flat

    phases = [
        (2015, 2025, "Reference period", "#E4E4E4"),
        (2041, 2060, "Mid-century", "#EAF1F4"),
        (2081, 2100, "Late-century", "#F6E9EC"),
    ]
    for ax in axs.flat:
        for x0, x1, _, color in phases:
            ax.axvspan(x0, x1, color=color, lw=0, zorder=0)
            ax.axvline(x0, color="#B0B0B0", lw=0.45,
                       ls=(0, (2.0, 2.0)), zorder=1)
            ax.axvline(x1, color="#B0B0B0", lw=0.45,
                       ls=(0, (2.0, 2.0)), zorder=1)
        ax.set_xlim(2015, 2100)
        ax.set_xticks([2020, 2040, 2060, 2080, 2100])
        ax.tick_params(direction="out", top=True, right=True, labelbottom=True)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Only manuscript-defined assessment windows are identified.
    for ax in (axa, axb):
        trans = ax.get_xaxis_transform()
        for x0, x1, label, _ in phases:
            ax.text((x0 + x1) / 2, 1.025, label, transform=trans,
                    ha="center", va="bottom", fontsize=6.3, color="black",
                    clip_on=False, linespacing=0.9)

    smoothed = {}
    for scenario in SCENARIOS:
        years = data[scenario]["year"]
        smoothed[scenario] = {}
        for field in ("ozone_change_pct", "uv_change_pct", "chs_anomaly",
                      "haf_with_uv", "haf_without_uv"):
            smoothed[scenario][field] = smooth11(data[scenario][field])
        axa.plot(years, smoothed[scenario]["ozone_change_pct"],
                 color=COLORS[scenario], lw=1.45)
        # Panel b is CHS and panel c is UV, matching the revised narrative order.
        axb.plot(years, smoothed[scenario]["chs_anomaly"],
                 color=COLORS[scenario], lw=1.45)
        axc.plot(years, smoothed[scenario]["uv_change_pct"],
                 color=COLORS[scenario], lw=1.45)
        axd.plot(years, smoothed[scenario]["haf_with_uv"],
                 color=COLORS[scenario], lw=1.45)
        axd.plot(years, smoothed[scenario]["haf_without_uv"],
                 color=COLORS[scenario], lw=0.95, ls=(0, (3, 2)), alpha=0.95)

    for ax in (axa, axb, axc):
        ax.axhline(0, color="#777777", lw=0.55, zorder=1)
    axa.set_ylabel("Estimated total-column ozone change (%)")
    axb.set_ylabel("Composite Hazard Score anomaly")
    axc.set_ylabel("Erythemal UV dose change (%)")
    axd.set_ylabel("Habitable Area Fraction")
    for ax in axs.flat:
        ax.set_xlabel("Year")

    years = data["ssp585"]["year"]
    ozone585 = smoothed["ssp585"]["ozone_change_pct"]
    uv585 = smoothed["ssp585"]["uv_change_pct"]
    # Reserve a clear top band for short, period-specific descriptions.
    axa.set_ylim(-0.65, 5.85)
    axb.set_ylim(-0.55, 7.85)
    axc.set_ylim(-1.30, 11.15)
    axd.set_ylim(0.08, 1.27)
    note_font = 6.2

    def fixed_note(ax, x0, x1, y_top, text, height=None):
        """Place an unboxed phase annotation centered within its phase."""
        if height is None:
            line_count = text.count("\n") + 1
            height = {1: 0.09, 2: 0.13, 3: 0.18}.get(line_count,
                                                        0.045 * line_count + 0.045)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((x0 + x1) / 2, y_top - height / 2, text, transform=trans,
                fontsize=note_font, ha="center", va="center",
                linespacing=1.05, zorder=6)

    # Description frames exactly match the horizontal extent of each phase.
    fixed_note(axa, 2041, 2060, 0.80, "Ozone increases\noverall")
    fixed_note(axa, 2081, 2100, 0.91, "Ozone remains\nelevated as\nUV rises")
    # Panels b and c have ample lower white space, so their mid-century boxes sit lower.
    fixed_note(axb, 2041, 2060, 0.68, "Pathway\ndifferences\nemerge")
    fixed_note(axb, 2081, 2100, 0.91, "Largest score\nincrease under\nSSP5-8.5")
    fixed_note(axc, 2041, 2060, 0.68, "UV trajectories\ndiffer from\nozone")
    fixed_note(axc, 2081, 2100, 0.91, "Largest UV\nincrease under\nSSP5-8.5")
    fixed_note(axd, 2041, 2060, 0.91, "Non-monotonic\nHAF ordering\nacross pathways")
    fixed_note(axd, 2081, 2100, 0.91, "Fastest HAF\ndecline under\nSSP5-8.5")
    # Mark each pathway's solid--dashed ablation gap with its line color.
    gap_years = {"ssp126": 2070, "ssp245": 2074, "ssp585": 2078}
    direct_box_top = 0.16
    fixed_note(axd, 2061, 2080, direct_box_top, "Direct UV term\nlowers HAF")
    for scenario, gap_year in gap_years.items():
        haf_with = smoothed[scenario]["haf_with_uv"]
        haf_without = smoothed[scenario]["haf_without_uv"]
        y1 = y_at(haf_with, years, gap_year)
        y2 = y_at(haf_without, years, gap_year)
        axd.annotate("", xy=(gap_year, y2), xytext=(gap_year, y1),
                     zorder=7,
                     arrowprops=dict(arrowstyle="<->", color=COLORS[scenario],
                                     lw=0.9, shrinkA=0, shrinkB=0))

    scenario_handles = [Line2D([0], [0], color=COLORS[s], lw=1.45,
                               label=LABELS[s]) for s in SCENARIOS]
    scenario_legends = []
    for ax in axs.flat:
        legend = ax.legend(handles=scenario_handles, frameon=True,
                           fancybox=False, ncol=3, loc="upper left",
                           bbox_to_anchor=(0.012, 0.988), borderaxespad=0,
                           handlelength=2.05, columnspacing=1.0,
                           handletextpad=0.5, fontsize=6.7,
                           edgecolor="#333333", framealpha=0.96,
                           borderpad=0.35)
        scenario_legends.append(legend)
    style_handles = [
        Line2D([0], [0], color="#333333", lw=1.45, label="With direct UV term"),
        Line2D([0], [0], color="#333333", lw=0.95, ls=(0, (3, 2)),
               label="Without direct UV term"),
    ]
    axd.add_artist(scenario_legends[-1])
    axd.legend(handles=style_handles, frameon=False, fancybox=False,
               loc="lower left", bbox_to_anchor=(0.02, 0.02), borderaxespad=0,
               borderpad=0.4)
    # Column-wise lettering: left column a-b, right column c-d.
    for letter, ax in zip("acbd", axs.flat):
        ax.text(-0.13, 1.045, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=10.5, va="bottom")

    stem = "Figure2"
    png, pdf = output / f"{stem}.png", output / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"source={source}")
    for scenario, gap_year in gap_years.items():
        gap = abs(y_at(smoothed[scenario]["haf_without_uv"], years, gap_year)
                  - y_at(smoothed[scenario]["haf_with_uv"], years, gap_year))
        print(f"HAF_gap_{scenario}_{gap_year}={gap:.6f}")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
