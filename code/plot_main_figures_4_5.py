"""Corrected main Figures 3–4 for the UV-extended CHS/HAF analysis."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
sys.path.insert(0, str(HERE))

import analysis_helpers as base
from haf_core import chs, emulator

YEARS = base.YEARS
SCENARIOS = base.SCENARIOS
LABELS = base.LABELS
PERIODS = {"mid": (2041, 2060), "late": (2081, 2100)}


def six_variable_setup(D):
    ally = np.arange(1750, 2301)
    params = {"ecs": 3.65, "gamma": 0.48}
    ref = emulator.run_emulator(params, ally, ssp="ssp245", chem=False)
    scales = chs.reference_scales_window(ref)
    # Independent same-window six-variable comparator. These weights are
    # estimated over 2015-2100 using the same SSP2-4.5 pathway, centred
    # 11-year smoothing, min-max rescaling and CRITIC calculation as CHS7.
    # They replace the earlier full-emulator six-variable weights.
    import json
    wd = json.loads(base.derived_or_output("critic_weights_2015_2100.json").read_text(encoding="utf-8"))
    weights = wd["critic6_11yr_smoothed_all_variables"]["weights"]
    idx = np.array([int(np.argmin(np.abs(ally - y))) for y in YEARS])
    climate = {}
    for scenario in SCENARIOS:
        out = emulator.run_emulator(params, ally, ssp=scenario, chem=False)
        st, uu = chs.tier_series(out, weights, scales)
        climate[scenario] = (st[idx], uu[idx])

    baseline = np.zeros_like(D["G"]["B2d"], float)
    baseline_disp = np.zeros_like(baseline)
    n = 0
    for scenario in SCENARIOS:
        st, uu = climate[scenario]
        for year in range(2015, 2026):
            j = year - 2015
            baseline += D["G"]["B2d"] + D["G"]["P2d"] * st[j] + uu[j]
            baseline_disp += D["bdisp"] + D["pdisp"] * st[j] + uu[j]
            n += 1
    baseline /= n
    baseline_disp /= n
    tau = base.weighted_percentile(baseline[D["land"]], D["area"][D["land"]], 90)
    return dict(weights=weights, climate=climate, baseline=baseline,
                baseline_disp=baseline_disp, tau=tau)


def compute(D, S):
    fields3, pure_uv, emergence6, emergence7 = {}, {}, {}, {}
    haf6, haf7 = {}, {}
    for scenario in SCENARIOS:
        st6, u6 = S["climate"][scenario]
        st7, u7 = D["climate_rel"][scenario]
        sums = {key: np.zeros_like(D["bdisp"], float) for key in PERIODS}
        uvsum = {key: np.zeros_like(D["bdisp"], float) for key in PERIODS}
        counts = {key: 0 for key in PERIODS}

        first6 = np.zeros(D["display_land"].shape, np.int16)
        first7 = np.zeros_like(first6)
        run6 = np.zeros_like(first6, np.uint8)
        run7 = np.zeros_like(first6, np.uint8)
        hab6 = D["display_land"] & (S["baseline_disp"] < S["tau"])
        hab7 = D["display_land"] & (D["baseline_disp"] < D["tau"])

        for j, year in enumerate(YEARS):
            uv = base.load_uv(scenario, int(year), D["G"])
            z = (uv - D["uv_ref"]) / D["uv_scale"]
            c6 = D["bdisp"] + D["pdisp"] * st6[j] + u6[j]
            c7 = (D["bdisp"] + D["pdisp"] * (D["six_total"] * st7[j])
                  + D["six_total"] * u7[j] + D["wuv"] * z)
            x6 = hab6 & (c6 >= S["tau"])
            x7 = hab7 & (c7 >= D["tau"])
            run6 = np.where(x6, np.minimum(run6 + 1, 5), 0).astype(np.uint8)
            run7 = np.where(x7, np.minimum(run7 + 1, 5), 0).astype(np.uint8)
            new6 = hab6 & (first6 == 0) & (run6 >= 5)
            new7 = hab7 & (first7 == 0) & (run7 >= 5)
            first6[new6] = int(year - 4)
            first7[new7] = int(year - 4)
            for key, (y0, y1) in PERIODS.items():
                if y0 <= year <= y1:
                    sums[key] += c7 - c6
                    uvsum[key] += D["wuv"] * z
                    counts[key] += 1
            if year == 2100:
                land = D["land"]
                area = D["area"]
                haf6[scenario] = float(area[land & (c6 < S["tau"])].sum() / area[land].sum())
                haf7[scenario] = float(area[land & (c7 < D["tau"])].sum() / area[land].sum())

        for key in PERIODS:
            fields3[(scenario, key)] = sums[key] / counts[key]
            pure_uv[(scenario, key)] = uvsum[key] / counts[key]
        e6 = np.full(first6.shape, np.nan)
        e7 = np.full(first7.shape, np.nan)
        e6[D["display_land"] & ~hab6] = 2014
        e7[D["display_land"] & ~hab7] = 2014
        e6[hab6 & (first6 == 0)] = 2101
        e7[hab7 & (first7 == 0)] = 2101
        e6[first6 > 0] = first6[first6 > 0]
        e7[first7 > 0] = first7[first7 > 0]
        emergence6[scenario], emergence7[scenario] = e6, e7
    return fields3, pure_uv, emergence6, emergence7, haf6, haf7


def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "Arial", "font.size": 6.5,
                         "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42,
                         "ps.fonttype": 42})


def figure3(D, fields, pure):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    style()
    vals = np.concatenate([np.abs(v[D["display_land"] & np.isfinite(v)]) for v in fields.values()])
    lim = float(np.nanpercentile(vals, 99))
    cmap = LinearSegmentedColormap.from_list("bwr_accessible", ["#2166ac", "#f7f7f7", "#b2182b"])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(2, 3, figsize=(7.205, 3.82))
    last = None
    for r, key in enumerate(("mid", "late")):
        for c, scenario in enumerate(SCENARIOS):
            ax = axes[r, c]
            f = np.where(D["display_land"], fields[(scenario, key)], np.nan)
            last = base.draw_robinson(ax, D["G"]["lon"], D["G"]["lat"], f,
                                      D["display_land"], cmap=cmap, norm=norm)
            valid = D["land"] & np.isfinite(f)
            mean = np.average(f[valid], weights=D["area"][valid])
            mean_label = f"{mean:+.3f}".replace("-", "−")
            ax.set_title(f"{LABELS[scenario]}\nmean $\\Delta$CHS = {mean_label}",
                         pad=2.5, weight="bold", linespacing=1.15)
            # Keep panel letters outside the map frame and clear of mapped colours.
            ax.text(-.018, 1.035, chr(97 + r * 3 + c), transform=ax.transAxes,
                    weight="bold", fontsize=8, ha="left", va="bottom", clip_on=False)
    fig.text(.019, .735, "Mid-century (2041–2060)", rotation=90,
             ha="center", va="center", weight="bold", fontsize=6.4)
    fig.text(.019, .275, "Late-century (2081–2100)", rotation=90,
             ha="center", va="center", weight="bold", fontsize=6.4)
    fig.subplots_adjust(left=.058, right=.915, bottom=.035, top=.945,
                        wspace=.040, hspace=.18)
    cax = fig.add_axes([.935, .18, .014, .65])
    cb = fig.colorbar(last, cax=cax, extend="both")
    cb.set_label(r"$\Delta CHS$", fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"Figure4_UV_total_CHS_increment.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    np.savez_compressed(FIG / "Figure4_source_fields_0p5deg.npz",
                        lat=D["G"]["lat"], lon=D["G"]["lon"],
                        **{f"delta_total_{s}_{p}": fields[(s, p)] for s in SCENARIOS for p in PERIODS},
                        **{f"pure_uv_{s}_{p}": pure[(s, p)] for s in SCENARIOS for p in PERIODS})


def figure4(D, S, em6, em7, haf6, haf7):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch
    style()
    scenarios = SCENARIOS
    # Crossing year is ordered data, so use a perceptually uniform sequential map.
    em_cmap = plt.get_cmap("cividis").copy()
    em_cmap.set_under("#3d3d3d")
    em_cmap.set_over("#d9d9d9")
    adv_cmap = LinearSegmentedColormap.from_list("advance", [
        "#2166ac", "#d8e8f2", "#f7f7f7", "#f6c4ad", "#b2182b"])
    adv = {}
    for s in scenarios:
        common = D["display_land"] & (S["baseline_disp"] < S["tau"]) & (D["baseline_disp"] < D["tau"])
        a = np.full(common.shape, np.nan)
        cross7 = common & (em7[s] >= 2015) & (em7[s] <= 2100)
        cross6 = cross7 & (em6[s] >= 2015) & (em6[s] <= 2100)
        never6 = cross7 & (em6[s] == 2101)
        a[cross6] = em6[s][cross6] - em7[s][cross6]
        a[never6] = 2101 - em7[s][never6]
        adv[s] = a
    fig, axes = plt.subplots(2, 3, figsize=(7.205, 3.82))
    for c, s in enumerate(scenarios):
        f = np.where(D["display_land"], em7[s], np.nan)
        im1 = base.draw_robinson(axes[0, c], D["G"]["lon"], D["G"]["lat"], f,
                                 D["display_land"], cmap=em_cmap, norm=Normalize(2020, 2100))
        axes[0, c].set_title(f"{LABELS[s]}\nHAF$_7$(2100) = {haf7[s]:.3f}",
                             pad=2.5, weight="bold", linespacing=1.1)
        im2 = base.draw_robinson(axes[1, c], D["G"]["lon"], D["G"]["lat"], adv[s],
                                 D["display_land"], cmap=adv_cmap,
                                 norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30))
        dh = haf7[s] - haf6[s]
        dh_label = f"{dh:+.3f}".replace("-", "−")
        axes[1, c].set_title(f"{LABELS[s]}\n$\\Delta$HAF(2100) = {dh_label}",
                             pad=2.5, weight="bold", linespacing=1.1)
        for r in range(2):
            axes[r, c].text(-.018, 1.035, chr(97 + r * 3 + c), transform=axes[r, c].transAxes,
                            weight="bold", fontsize=8, ha="left", va="bottom", clip_on=False)
    fig.text(.019, .735, "First persistent threshold crossing", rotation=90,
             ha="center", va="center", fontsize=6.3, weight="bold")
    fig.text(.019, .275, "UV extension effect on crossing year", rotation=90,
             ha="center", va="center", fontsize=6.3, weight="bold")
    fig.subplots_adjust(left=.058, right=.915, top=.945, bottom=.035,
                        wspace=.040, hspace=.18)
    cax1 = fig.add_axes([.935, .555, .014, .31])
    cb1 = fig.colorbar(im1, cax=cax1, extend="both")
    cb1.set_ticks([2020, 2040, 2060, 2080, 2100])
    cb1.set_label("Crossing year", fontsize=6.7, labelpad=3)
    cb1.ax.yaxis.set_label_position("left")
    fig.legend(handles=[Patch(facecolor="#3d3d3d", label="Above threshold in baseline"),
                        Patch(facecolor="#d9d9d9", label="No crossing by 2100")],
               loc="upper right", bbox_to_anchor=(.995, .965), frameon=False,
               fontsize=5.6, handlelength=1.1, borderaxespad=0)
    cax2 = fig.add_axes([.935, .135, .014, .31])
    cb2 = fig.colorbar(im2, cax=cax2, extend="both")
    crossing_ticks = [-30, -15, 0, 15, 30]
    cb2.set_ticks(crossing_ticks)
    cb2.ax.set_yticklabels([str(value).replace("-", "−")
                            for value in crossing_ticks])
    cb2.set_label("Shift in crossing year", fontsize=6.7, labelpad=3)
    cb2.ax.yaxis.set_label_position("left")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"Figure5_HAF_and_first_crossing.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    np.savez_compressed(FIG / "Figure5_source_fields_0p5deg.npz",
                        lat=D["G"]["lat"], lon=D["G"]["lon"],
                        **{f"emergence7_{s}": em7[s] for s in scenarios},
                        **{f"emergence6_{s}": em6[s] for s in scenarios},
                        **{f"shift_{s}": adv[s] for s in scenarios},
                        **{f"haf6_{s}": haf6[s] for s in scenarios},
                        **{f"haf7_{s}": haf7[s] for s in scenarios},
                        **{f"delta_haf_{s}": haf7[s] - haf6[s] for s in scenarios})


def main():
    if not (ROOT / "outputs" / "uv_ssp126_2015_2015.npz").exists():
        from source_data_reproduction import figure4, figure5
        figure4()
        figure5()
        print("created Figures 4 and 5 from archived source data")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    D = base.common_setup()
    S = six_variable_setup(D)
    fields, pure, em6, em7, haf6, haf7 = compute(D, S)
    figure3(D, fields, pure)
    figure4(D, S, em6, em7, haf6, haf7)
    print(f"tau6={S['tau']:.6f}; tau7={D['tau']:.6f}; wUV={D['wuv']:.6f}")
    print("HAF6", haf6)
    print("HAF7", haf7)


if __name__ == "__main__":
    main()
