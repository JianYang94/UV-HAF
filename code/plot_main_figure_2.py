"""Create publication-style Figure 2 for the seven-variable UV-HAF study.

The figure uses UKESM1-0-LL r1i1p1f2 monthly drivers reconstructed to annual
erythemal dose for three SSPs, the original HAF emulator for the six carried
variables, and the 2015-2100 11-year-smoothed CRITIC weights.  A fixed
2015-2025 modern baseline and its area-weighted 90th-percentile CHS threshold
are used for all scenarios.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
DATA = ROOT / "data" / "derived"
sys.path.insert(0, str(HERE))

import uv_reconstruction as uvbase
from haf_core import chs, emulator, grid, whi

YEARS = np.arange(2015, 2101)
SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)


def smooth11(x: np.ndarray) -> np.ndarray:
    return np.convolve(np.pad(np.asarray(x, float), (5, 5), mode="edge"),
                       np.ones(11) / 11.0, mode="valid")


def weighted_percentile(values, weights, q):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    return float(np.interp(q / 100.0, np.cumsum(w[order]) / w.sum(), v[order]))


def load_year(scenario: str, year: int):
    path = OUT / f"uv_{scenario}_{year}_{year}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        annual = np.asarray(z["annual"], float)
        ozone = np.average(np.asarray(z["ozone"], float), axis=0, weights=MONTH_DAYS)
        lat = np.asarray(z["lat"], float)
        lon = np.asarray(z["lon"], float)
    return annual, ozone, lat, lon


def land_mean(field, G):
    land = G["land2d"] & np.isfinite(field)
    return float(np.average(field[land], weights=G["area2d"][land]))


def main():
    if not (OUT / "uv_ssp126_2015_2015.npz").exists():
        from source_data_reproduction import figure2
        figure2()
        print("created Figure 2 from archived source data")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    G = grid.build()
    B, coverage, _ = whi.load_whi_field(G)
    grid.set_baseline(B, G)
    land, area = G["land2d"], G["area2d"]
    land_area = float(area[land].sum())

    # First pass: annual global-land ozone/UV means and the common local UV baseline.
    ozone_mean = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    uv_mean = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    uv_ref_sum = np.zeros_like(G["B2d"], dtype=np.float64)
    uv_ref_n = 0
    for s in SCENARIOS:
        for j, y in enumerate(YEARS):
            annual, ozone, lat, lon = load_year(s, int(y))
            uv05 = uvbase.regrid(annual, lat, lon, G["lat"], G["lon"])
            o305 = uvbase.regrid(ozone, lat, lon, G["lat"], G["lon"])
            uv_mean[s][j] = land_mean(uv05, G)
            ozone_mean[s][j] = land_mean(o305, G)
            if 2015 <= y <= 2025:
                uv_ref_sum += np.nan_to_num(uv05)
                uv_ref_n += 1
    uv_ref = uv_ref_sum / uv_ref_n

    # Match the original HAF standardisation logic: anchor on the modern
    # baseline and use the full SSP2-4.5 forced-series spread as UV scale.
    uv_scale = float(np.std(smooth11(uv_mean["ssp245"]), ddof=1))
    if not np.isfinite(uv_scale) or uv_scale <= 0:
        raise RuntimeError("invalid UV scale")

    weights_path = OUT / "critic_weights_2015_2100.json"
    if not weights_path.exists():
        weights_path = DATA / "critic_weights_2015_2100.json"
    weights_doc = json.loads(weights_path.read_text(encoding="utf-8"))
    weights7 = weights_doc["critic7_11yr_smoothed_all_variables"]["weights"]
    wuv = float(weights7["uv_erythema"])
    six_total = float(sum(weights7[k] for k in chs.CHS_VARS))
    six_relative = {k: float(weights7[k] / six_total) for k in chs.CHS_VARS}

    all_years = np.arange(1750, 2301)
    params = {"ecs": 3.65, "gamma": 0.48}
    refout = emulator.run_emulator(params, all_years, ssp="ssp245", chem=False)
    scales = chs.reference_scales_window(refout)
    idx = np.array([int(np.argmin(np.abs(all_years - y))) for y in YEARS])
    climate = {}
    for s in SCENARIOS:
        out = emulator.run_emulator(params, all_years, ssp=s, chem=False)
        st, uu = chs.tier_series(out, six_relative, scales)
        climate[s] = (six_total * st[idx], six_total * uu[idx])

    # One threshold for every year/scenario, calibrated on 2015-2025 CHS.
    baseline_chs = np.zeros_like(G["B2d"], dtype=np.float64)
    baseline_n = 0
    base_mask = (YEARS >= 2015) & (YEARS <= 2025)
    for s in SCENARIOS:
        st, uu = climate[s]
        for j in np.where(base_mask)[0]:
            baseline_chs += G["B2d"] + G["P2d"] * st[j] + uu[j]
            baseline_n += 1
    baseline_chs /= baseline_n
    tau = weighted_percentile(baseline_chs[land], area[land], 90)

    chs_mean = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    haf_with = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    haf_without = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    for s in SCENARIOS:
        st, uu = climate[s]
        for j, y in enumerate(YEARS):
            annual, _, lat, lon = load_year(s, int(y))
            uv05 = uvbase.regrid(annual, lat, lon, G["lat"], G["lon"])
            zuv = (uv05 - uv_ref) / uv_scale
            no_uv = G["B2d"] + G["P2d"] * st[j] + uu[j]
            total = no_uv + wuv * zuv
            valid = land & np.isfinite(total)
            den = area[valid].sum()
            chs_mean[s][j] = float(np.average(total[valid], weights=area[valid]))
            haf_with[s][j] = float(area[valid & (total < tau)].sum() / den)
            haf_without[s][j] = float(area[valid & (no_uv < tau)].sum() / den)

    # Baseline-relative plotting metrics.
    ozone_pct, uv_pct, chs_anom = {}, {}, {}
    for s in SCENARIOS:
        o0 = float(np.mean(ozone_mean[s][base_mask]))
        u0 = float(np.mean(uv_mean[s][base_mask]))
        c0 = float(np.mean(chs_mean[s][base_mask]))
        ozone_pct[s] = 100.0 * (ozone_mean[s] / o0 - 1.0)
        uv_pct[s] = 100.0 * (uv_mean[s] / u0 - 1.0)
        chs_anom[s] = chs_mean[s] - c0

    # Nature-style double-column figure (183 mm), Arial/Helvetica, no gridlines.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    })
    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.45), sharex=True,
                            constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.09,
                        wspace=0.19, hspace=0.26)
    axa, axb, axc, axd = axs.flat
    for ax in axs.flat:
        ax.axvspan(2015, 2025, color="#E6E6E6", lw=0, zorder=0)
        ax.set_xlim(2015, 2100)
        ax.set_xticks([2020, 2040, 2060, 2080, 2100])
        ax.tick_params(direction="out", top=False, right=False, labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for s in SCENARIOS:
        axa.plot(YEARS, smooth11(ozone_pct[s]), color=COLORS[s], lw=1.35,
                 label=LABELS[s])
        axb.plot(YEARS, smooth11(uv_pct[s]), color=COLORS[s], lw=1.35)
        axc.plot(YEARS, smooth11(chs_anom[s]), color=COLORS[s], lw=1.35)
        axd.plot(YEARS, smooth11(haf_with[s]), color=COLORS[s], lw=1.35)
        axd.plot(YEARS, smooth11(haf_without[s]), color=COLORS[s], lw=0.9,
                 ls=(0, (3, 2)), alpha=0.9)

    axa.axhline(0, color="#777777", lw=0.55)
    axb.axhline(0, color="#777777", lw=0.55)
    axc.axhline(0, color="#777777", lw=0.55)
    axa.set_ylabel("Estimated total-column ozone change (%)")
    axb.set_ylabel("Erythemal UV dose change (%)")
    axc.set_ylabel("Composite Hazard Score anomaly")
    axd.set_ylabel("Habitable Area Fraction")
    for ax in axs.flat:
        ax.set_xlabel("Year", fontweight="normal")

    scenario_handles = [
        Line2D([0], [0], color=COLORS[s], lw=1.35, label=LABELS[s])
        for s in SCENARIOS
    ]
    scenario_handles.append(Patch(facecolor="#E6E6E6", edgecolor="none",
                                  label="2015-2025 baseline"))
    fig.legend(handles=scenario_handles, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.53, 0.985),
               handlelength=2.2, columnspacing=1.5)

    style_handles = [
        Line2D([0], [0], color="#333333", lw=1.35, label="With UV term"),
        Line2D([0], [0], color="#333333", lw=0.9, ls=(0, (3, 2)),
               label="Without UV term"),
    ]
    axd.legend(handles=style_handles, frameon=False, loc="lower left",
               bbox_to_anchor=(0.14, 0.02), borderaxespad=0)
    for label, ax in zip("abcd", axs.flat):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold",
                fontsize=8, va="bottom")

    png = FIG / "Figure2_ozone_UV_CHS_HAF_2015_2100.png"
    pdf = FIG / "Figure2_ozone_UV_CHS_HAF_2015_2100.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    csv_path = FIG / "Figure2_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "scenario", "ozone_DU", "ozone_change_pct",
                         "erythemal_uv_dose", "uv_change_pct", "chs_mean",
                         "chs_anomaly", "haf_with_uv", "haf_without_uv"])
        for s in SCENARIOS:
            for j, y in enumerate(YEARS):
                writer.writerow([int(y), s, ozone_mean[s][j], ozone_pct[s][j],
                                 uv_mean[s][j], uv_pct[s][j], chs_mean[s][j],
                                 chs_anom[s][j], haf_with[s][j], haf_without[s][j]])

    meta = {
        "figure": "Figure 2",
        "years": [2015, 2100],
        "baseline": [2015, 2025],
        "model": "UKESM1-0-LL r1i1p1f2",
        "scenarios": list(SCENARIOS),
        "smoothing": "centred 11-year running mean with edge padding",
        "critic_weights": weights7,
        "uv_scale_J_m-2_d-1": uv_scale,
        "threshold_percentile": 90,
        "tau": tau,
        "whi_coverage": coverage,
        "comparison": "with versus without UV term; seven-variable weights held fixed",
    }
    (FIG / "Figure2_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "csv": str(csv_path),
                      "tau": tau, "uv_scale": uv_scale}, indent=2))


if __name__ == "__main__":
    main()
