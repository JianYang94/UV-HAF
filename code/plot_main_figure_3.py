"""Figure 3: global spatial CHS maps for three SSPs and two future periods.

Uses exactly the seven-variable CRITIC weights, UV standardisation, 2015-2025
baseline and HAF threshold used for Figure 2.  The map design follows the
original HAF paper: Robinson projection, Natural Earth coastlines, land-only
hazard raster, and a shared sequential hazard colour scale.
"""
from __future__ import annotations

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
PERIODS = {"mid": (2041, 2060), "late": (2081, 2100)}


def smooth11(x: np.ndarray) -> np.ndarray:
    return np.convolve(np.pad(np.asarray(x, float), (5, 5), mode="edge"),
                       np.ones(11) / 11.0, mode="valid")


def weighted_percentile(values, weights, q):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    return float(np.interp(q / 100.0, np.cumsum(w[order]) / w.sum(), v[order]))


def load_uv(scenario: str, year: int, G):
    p = OUT / f"uv_{scenario}_{year}_{year}.npz"
    with np.load(p) as z:
        annual = np.asarray(z["annual"], float)
        lat = np.asarray(z["lat"], float)
        lon = np.asarray(z["lon"], float)
    return uvbase.regrid(annual, lat, lon, G["lat"], G["lon"])


def main():
    if not (OUT / "uv_ssp126_2015_2015.npz").exists():
        from source_data_reproduction import figure3
        figure3()
        print("created Figure 3 from archived source data")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    G = grid.build()
    B, coverage, _ = whi.load_whi_field(G)
    grid.set_baseline(B, G)
    land, area = G["land2d"], G["area2d"]

    # The original HAF domain excludes the Antarctic ice sheet.  For the map,
    # nevertheless display Antarctic land using a transparent, documented
    # extrapolation: neutral baseline water hazard (B=0), the same forced
    # climate/UV terms, and the land-amplified temperature pattern. Antarctica
    # remains excluded from HAF and threshold calculations so Figure 3 stays
    # numerically consistent with Figure 2.
    released_mask = HERE / "haf_core" / "released" / "land_mask_0p5deg.npz"
    full_land = np.load(released_mask)["mask"].astype(bool)
    antarctica = full_land & ~land & (G["lat2d"] <= -60.0)
    raw_lat = 1.0 + 1.6 * (np.abs(G["lat2d"]) / 90.0) ** 2
    raw_existing = np.where(land, 1.4 * raw_lat, raw_lat)
    gmean_existing = float(np.sum(area * raw_existing) / np.sum(area))
    p_display = np.array(G["P2d"], copy=True)
    p_display[antarctica] = 1.4 * raw_lat[antarctica] / gmean_existing
    b_display = np.array(G["B2d"], copy=True)
    b_display[antarctica] = 0.0
    display_land = land | antarctica

    # Reproduce Figure 2's common spatial UV reference and temporal scale.
    uv_ref_sum = np.zeros_like(G["B2d"], dtype=np.float64)
    uv_ref_n = 0
    uv_global_245 = np.empty(len(YEARS), dtype=float)
    for s in SCENARIOS:
        for y in range(2015, 2026):
            uv_ref_sum += np.nan_to_num(load_uv(s, y, G))
            uv_ref_n += 1
    uv_ref = uv_ref_sum / uv_ref_n
    for j, y in enumerate(YEARS):
        uv = load_uv("ssp245", int(y), G)
        valid = land & np.isfinite(uv)
        uv_global_245[j] = np.average(uv[valid], weights=area[valid])
    uv_scale = float(np.std(smooth11(uv_global_245), ddof=1))

    weights_path = OUT / "critic_weights_2015_2100.json"
    if not weights_path.exists():
        weights_path = DATA / "critic_weights_2015_2100.json"
    wd = json.loads(weights_path.read_text(encoding="utf-8"))
    weights7 = wd["critic7_11yr_smoothed_all_variables"]["weights"]
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

    # Same fixed 2015-2025 threshold as Figure 2.
    baseline_chs = np.zeros_like(G["B2d"], dtype=np.float64)
    baseline_n = 0
    for s in SCENARIOS:
        st, uu = climate[s]
        for y in range(2015, 2026):
            j = y - 2015
            baseline_chs += G["B2d"] + G["P2d"] * st[j] + uu[j]
            baseline_n += 1
    baseline_chs /= baseline_n
    tau = weighted_percentile(baseline_chs[land], area[land], 90)

    maps = {}
    diagnostics = []
    for period, (y0, y1) in PERIODS.items():
        for s in SCENARIOS:
            acc = np.zeros_like(G["B2d"], dtype=np.float64)
            n = 0
            st, uu = climate[s]
            for y in range(y0, y1 + 1):
                j = y - 2015
                uv = load_uv(s, y, G)
                zuv = (uv - uv_ref) / uv_scale
                acc += b_display + p_display * st[j] + uu[j] + wuv * zuv
                n += 1
            field = np.where(display_land, acc / n, np.nan)
            maps[f"{s}_{period}"] = field.astype("float32")
            valid = land & np.isfinite(field)
            den = area[valid].sum()
            diagnostics.append({
                "scenario": s, "period": period, "years": [y0, y1],
                "area_weighted_mean_chs": float(np.average(field[valid], weights=area[valid])),
                "haf": float(area[valid & (field < tau)].sum() / den),
                "land_above_threshold": float(area[valid & (field >= tau)].sum() / den),
            })

    # Original-paper hazard palette.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import cartopy
    import cartopy.crs as ccrs
    cartopy.config["data_dir"] = str(HERE / "cartopy_data")

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.5,
        "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": False,
    })
    hazard = LinearSegmentedColormap.from_list("hazard", [
        "#f7f4ec", "#f6e2b3", "#f3c169", "#ea9a3c", "#d96b2c",
        "#b8342a", "#8c1d35", "#4d0f27"])
    pooled = np.concatenate([v[np.isfinite(v)] for v in maps.values()])
    vmin = float(np.floor(np.percentile(pooled, 0.5)))
    vmax = float(np.ceil(np.percentile(pooled, 99.5)))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(2, 3, figsize=(7.205, 3.82),
                            subplot_kw={"projection": ccrs.Robinson()},
                            constrained_layout=False)
    fig.subplots_adjust(left=0.048, right=0.915, top=0.965, bottom=0.035,
                        wspace=0.035, hspace=0.13)
    tr = ccrs.PlateCarree()
    diagnostic_lookup = {(d["scenario"], d["period"]): d for d in diagnostics}
    im = None
    for r, period in enumerate(("mid", "late")):
        for c, s in enumerate(SCENARIOS):
            ax = axs[r, c]
            field = maps[f"{s}_{period}"]
            ax.set_global()
            im = ax.pcolormesh(G["lon"], G["lat"], field, transform=tr,
                               cmap=hazard, norm=norm, shading="auto",
                               rasterized=True)
            ax.coastlines(resolution="110m", linewidth=0.28, color="#222222")
            ax.spines["geo"].set_linewidth(0.45)
            haf = diagnostic_lookup[(s, period)]["haf"]
            ax.set_title(f"{LABELS[s]}\nHAF = {haf:.3f}", pad=2.5,
                         fontweight="bold", linespacing=1.15)
            letter = "abcdef"[r * 3 + c]
            ax.text(0.01, 0.96, letter, transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", fontsize=8)

    fig.text(0.019, 0.735, "Mid-century (2041–2060)", rotation=90,
             ha="center", va="center", fontsize=6.4, fontweight="bold")
    fig.text(0.019, 0.275, "Late-century (2081–2100)", rotation=90,
             ha="center", va="center", fontsize=6.4, fontweight="bold")

    cax = fig.add_axes([0.935, 0.18, 0.014, 0.65])
    cb = fig.colorbar(im, cax=cax)
    ticks = np.arange(np.ceil(vmin / 2) * 2, vmax + 0.1, 2)
    cb.set_ticks(ticks)
    cb.set_label("Composite Hazard Score (CHS)", fontsize=7, labelpad=5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2.5)
    cb.outline.set_linewidth(0.45)
    png = FIG / "Figure3_global_CHS_maps_mid_late.png"
    pdf = FIG / "Figure3_global_CHS_maps_mid_late.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    np.savez_compressed(FIG / "Figure3_source_fields_0p5deg.npz",
                        lon=G["lon"], lat=G["lat"], tau=tau, **maps)
    meta = {
        "figure": "Figure 3", "model": "UKESM1-0-LL r1i1p1f2",
        "baseline": [2015, 2025], "periods": PERIODS,
        "scenarios": list(SCENARIOS), "projection": "Robinson",
        "map_resolution_degrees": 0.5, "tau": tau,
        "uv_scale_J_m-2_d-1": uv_scale, "critic_weights": weights7,
        "whi_coverage": coverage, "vmin": vmin, "vmax": vmax,
        "diagnostics": diagnostics,
        "antarctica_display": "CHS shown using neutral B=0 water-hazard fill and forced climate/UV terms; excluded from HAF and threshold calculations",
    }
    (FIG / "Figure3_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "tau": tau,
                      "vmin": vmin, "vmax": vmax, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
