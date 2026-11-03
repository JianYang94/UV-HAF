"""Create publication-style Figures 3 and 4 for the UV-extended HAF study.

Figure 3 maps first persistent threshold crossing and UV-induced advancement.
Figure 4 diagnoses ozone-UV coupling, UV attribution, and sensitivity to UV
weight and the habitability threshold.  Methods are consistent with Figures 1-2.
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

from haf_core import chs, emulator, grid, whi

YEARS = np.arange(2015, 2101)
SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}
PERIODS = {"mid": (2041, 2060), "late": (2081, 2100)}


def derived_or_output(name):
    """Return a generated output when present, otherwise the released derivative."""
    generated = OUT / name
    return generated if generated.exists() else DATA / name


def smooth11(x):
    return np.convolve(np.pad(np.asarray(x, float), (5, 5), mode="edge"),
                       np.ones(11) / 11.0, mode="valid")


def weighted_percentile(values, weights, q):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    return float(np.interp(q / 100.0, np.cumsum(w[order]) / w.sum(), v[order]))


def regrid(source, slat, slon, dlat, dlon):
    """Bilinear regridding with periodic longitude; no NetCDF dependency."""
    source = np.asarray(source, float)
    order = np.argsort(slat)
    slat, source = np.asarray(slat)[order], source[order]
    slon = np.mod(slon, 360)
    order = np.argsort(slon)
    slon, source = slon[order], source[:, order]
    ext_lon = np.r_[slon, slon[0] + 360]
    ext = np.concatenate([source, source[:, :1]], axis=1)
    tlon = np.mod(dlon, 360)
    tmp = np.empty((source.shape[0], tlon.size), float)
    for i in range(source.shape[0]):
        tmp[i] = np.interp(tlon, ext_lon, ext[i])
    out = np.empty((dlat.size, dlon.size), float)
    for j in range(tlon.size):
        out[:, j] = np.interp(dlat, slat, tmp[:, j])
    return out


def load_uv(scenario, year, G):
    p = OUT / f"uv_{scenario}_{year}_{year}.npz"
    with np.load(p) as z:
        annual = np.asarray(z["annual"], float)
        lat = np.asarray(z["lat"], float)
        lon = np.asarray(z["lon"], float)
    return regrid(annual, lat, lon, G["lat"], G["lon"])


def robinson_xy(lon, lat):
    """Robinson projection using the published 5-degree coefficient table."""
    xt = np.array([1.0000, .9986, .9954, .9900, .9822, .9730, .9600,
                   .9427, .9216, .8962, .8679, .8350, .7986, .7597,
                   .7186, .6732, .6213, .5722, .5322])
    yt = np.array([0.0000, .0620, .1240, .1860, .2480, .3100, .3720,
                   .4340, .4958, .5571, .6176, .6769, .7346, .7903,
                   .8435, .8936, .9394, .9761, 1.0000])
    lon, lat = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))
    a = np.abs(lat)
    xp = np.interp(a, np.arange(0, 91, 5), xt)
    yp = np.interp(a, np.arange(0, 91, 5), yt)
    x = 0.8487 * np.deg2rad(lon) * xp
    y = 1.3523 * np.sign(lat) * yp
    return x, y


def draw_robinson(ax, lon, lat, field, land_mask, **kwargs):
    lon2, lat2 = np.meshgrid(lon, lat)
    x, y = robinson_xy(lon2, lat2)
    im = ax.pcolormesh(x, y, field, shading="auto", rasterized=True, **kwargs)
    ax.contour(x, y, land_mask.astype(float), levels=[0.5], colors="#222222",
               linewidths=0.28)
    blats = np.linspace(-90, 90, 361)
    xl, yl = robinson_xy(np.full_like(blats, -180.0), blats)
    xr, yr = robinson_xy(np.full_like(blats, 180.0), blats)
    ax.plot(np.r_[xl, xr[::-1], xl[:1]], np.r_[yl, yr[::-1], yl[:1]],
            color="#222222", lw=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.75, 2.75); ax.set_ylim(-1.40, 1.40)
    ax.set_axis_off()
    return im


def common_setup():
    G = grid.build()
    B, coverage, _ = whi.load_whi_field(G)
    grid.set_baseline(B, G)
    land, area = G["land2d"], G["area2d"]

    # Display Antarctica using a transparent extrapolation, but retain the
    # original ice-free HAF domain for all fractions and thresholds.
    released = HERE / "haf_core" / "released" / "land_mask_0p5deg.npz"
    full_land = np.load(released)["mask"].astype(bool)
    antarctica = full_land & ~land & (G["lat2d"] <= -60.0)
    raw_lat = 1.0 + 1.6 * (np.abs(G["lat2d"]) / 90.0) ** 2
    raw_existing = np.where(land, 1.4 * raw_lat, raw_lat)
    gmean = float(np.sum(area * raw_existing) / np.sum(area))
    pdisp = np.array(G["P2d"], copy=True)
    pdisp[antarctica] = 1.4 * raw_lat[antarctica] / gmean
    bdisp = np.array(G["B2d"], copy=True)
    bdisp[antarctica] = 0.0
    display_land = land | antarctica

    uv_ref = np.zeros_like(G["B2d"], float)
    nref = 0
    for s in SCENARIOS:
        for y in range(2015, 2026):
            uv_ref += np.nan_to_num(load_uv(s, y, G))
            nref += 1
    uv_ref /= nref

    uv_mean_245 = []
    for y in YEARS:
        uv = load_uv("ssp245", int(y), G)
        valid = land & np.isfinite(uv)
        uv_mean_245.append(np.average(uv[valid], weights=area[valid]))
    uv_scale = float(np.std(smooth11(uv_mean_245), ddof=1))

    wd = json.loads(derived_or_output("critic_weights_2015_2100.json").read_text(encoding="utf-8"))
    weights7 = wd["critic7_11yr_smoothed_all_variables"]["weights"]
    wuv = float(weights7["uv_erythema"])
    six_total = float(sum(weights7[k] for k in chs.CHS_VARS))
    six_relative = {k: float(weights7[k] / six_total) for k in chs.CHS_VARS}

    ally = np.arange(1750, 2301)
    params = {"ecs": 3.65, "gamma": 0.48}
    refout = emulator.run_emulator(params, ally, ssp="ssp245", chem=False)
    scales = chs.reference_scales_window(refout)
    idx = np.array([int(np.argmin(np.abs(ally - y))) for y in YEARS])
    climate_rel = {}
    for s in SCENARIOS:
        out = emulator.run_emulator(params, ally, ssp=s, chem=False)
        st, uu = chs.tier_series(out, six_relative, scales)
        climate_rel[s] = (st[idx], uu[idx])

    # Fixed modern threshold and baseline state, exactly as in Figures 1-2.
    baseline = np.zeros_like(G["B2d"], float)
    baseline_disp = np.zeros_like(G["B2d"], float)
    n = 0
    for s in SCENARIOS:
        st, uu = climate_rel[s]
        for y in range(2015, 2026):
            j = y - 2015
            baseline += G["B2d"] + G["P2d"] * (six_total * st[j]) + six_total * uu[j]
            baseline_disp += bdisp + pdisp * (six_total * st[j]) + six_total * uu[j]
            n += 1
    baseline /= n
    baseline_disp /= n
    tau = weighted_percentile(baseline[land], area[land], 90)
    return dict(G=G, land=land, area=area, display_land=display_land,
                antarctica=antarctica, bdisp=bdisp, pdisp=pdisp,
                uv_ref=uv_ref, uv_scale=uv_scale, weights7=weights7,
                wuv=wuv, six_total=six_total, climate_rel=climate_rel,
                baseline=baseline, baseline_disp=baseline_disp, tau=tau,
                coverage=coverage)


def analyse(D):
    G, land, area = D["G"], D["land"], D["area"]
    display = D["display_land"]
    base_hab = display & (D["baseline_disp"] < D["tau"])
    already = display & ~base_hab
    emergence, emergence_no_uv = {}, {}
    period = {}
    haf2100 = {}

    for s in SCENARIOS:
        first = np.zeros(display.shape, dtype=np.int16)
        first0 = np.zeros(display.shape, dtype=np.int16)
        run = np.zeros(display.shape, dtype=np.uint8)
        run0 = np.zeros(display.shape, dtype=np.uint8)
        acc = {p: {"clim_st": 0.0, "clim_u": 0.0,
                   "z": np.zeros(display.shape, float), "n": 0}
               for p in PERIODS}
        st, uu = D["climate_rel"][s]
        for j, y in enumerate(YEARS):
            uv = load_uv(s, int(y), G)
            z = (uv - D["uv_ref"]) / D["uv_scale"]
            no_uv = D["bdisp"] + D["pdisp"] * (D["six_total"] * st[j]) + D["six_total"] * uu[j]
            total = no_uv + D["wuv"] * z
            cross = base_hab & (total >= D["tau"])
            cross0 = base_hab & (no_uv >= D["tau"])
            run = np.where(cross, np.minimum(run + 1, 5), 0).astype(np.uint8)
            run0 = np.where(cross0, np.minimum(run0 + 1, 5), 0).astype(np.uint8)
            new = base_hab & (first == 0) & (run >= 5)
            new0 = base_hab & (first0 == 0) & (run0 >= 5)
            first[new] = int(y - 4)
            first0[new0] = int(y - 4)
            if y == 2100:
                valid = land & np.isfinite(total)
                haf2100[s] = float(area[valid & (total < D["tau"])].sum() / area[valid].sum())
            for p, (y0, y1) in PERIODS.items():
                if y0 <= y <= y1:
                    acc[p]["clim_st"] += st[j]
                    acc[p]["clim_u"] += uu[j]
                    acc[p]["z"] += np.nan_to_num(z)
                    acc[p]["n"] += 1

        em = np.full(display.shape, np.nan, float)
        em0 = np.full(display.shape, np.nan, float)
        em[already] = 2014; em0[already] = 2014
        em[base_hab & (first == 0)] = 2101
        em0[base_hab & (first0 == 0)] = 2101
        em[first > 0] = first[first > 0]
        em0[first0 > 0] = first0[first0 > 0]
        emergence[s], emergence_no_uv[s] = em, em0

        for p in PERIODS:
            q = acc[p]; n = q["n"]
            period[(s, p)] = dict(st=q["clim_st"] / n, u=q["clim_u"] / n,
                                  z=q["z"] / n)

    return emergence, emergence_no_uv, period, haf2100


def figure3(D, emergence, emergence0, haf2100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.titlesize": 7,
        "legend.fontsize": 5.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    emerge_cmap = LinearSegmentedColormap.from_list("emerge", [
        "#9b1c1c", "#e65f1e", "#f3b52b", "#8bcf83", "#45a9ae", "#3567a9"])
    emerge_cmap.set_under("#3d3d3d")
    emerge_cmap.set_over("#d9d9d9")
    adv_cmap = LinearSegmentedColormap.from_list("advance", [
        "#2166ac", "#d8e8f2", "#f7f7f7", "#f6c4ad", "#b2182b"])

    fig, axs = plt.subplots(2, 3, figsize=(7.205, 3.82))
    fig.subplots_adjust(left=0.05, right=0.90, top=0.965, bottom=0.055,
                        wspace=0.035, hspace=0.14)
    top_im = bot_im = None
    advances = {}
    for c, s in enumerate(SCENARIOS):
        em, em0 = emergence[s], emergence0[s]
        adv = em0 - em
        # Already-uninhabitable and never-crossing cells have no defined advance.
        valid_adv = D["display_land"] & (em >= 2015) & (em <= 2100)
        adv = np.where(valid_adv, np.clip(adv, -30, 30), np.nan)
        advances[s] = adv.astype("float32")
        for r, field in enumerate((em, adv)):
            ax = axs[r, c]
            if r == 0:
                top_im = draw_robinson(ax, D["G"]["lon"], D["G"]["lat"], field,
                                       D["display_land"], cmap=emerge_cmap,
                                       norm=Normalize(2020, 2100))
                ax.set_title(f"{LABELS[s]}\nHAF$_{{2100}}$ = {haf2100[s]:.3f}",
                             pad=2.5, fontweight="bold", linespacing=1.15)
            else:
                bot_im = draw_robinson(ax, D["G"]["lon"], D["G"]["lat"], field,
                                       D["display_land"], cmap=adv_cmap,
                                       norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30))
                ax.set_title(LABELS[s], pad=2.5, fontweight="bold")
            ax.text(0.01, 0.96, "abcdef"[r * 3 + c], transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold")

    fig.text(0.019, 0.735, "First persistent threshold crossing", rotation=90,
             ha="center", va="center", fontsize=6.3, fontweight="bold")
    fig.text(0.019, 0.275, "UV-induced change in crossing year", rotation=90,
             ha="center", va="center", fontsize=6.3, fontweight="bold")
    cax1 = fig.add_axes([0.915, 0.555, 0.012, 0.31])
    cb1 = fig.colorbar(top_im, cax=cax1, extend="both")
    cb1.set_ticks([2020, 2040, 2060, 2080, 2100])
    cb1.set_label("First crossing year", fontsize=6.7, labelpad=4)
    cb1.ax.tick_params(labelsize=5.8, width=0.45, length=2.3)
    cb1.outline.set_linewidth(0.4)
    fig.legend(handles=[Patch(facecolor="#3d3d3d", label="Above threshold in baseline"),
                        Patch(facecolor="#d9d9d9", label="No crossing by 2100")],
               loc="upper right", bbox_to_anchor=(0.995, 0.965), frameon=False,
               handlelength=1.1, borderaxespad=0)
    cax2 = fig.add_axes([0.915, 0.135, 0.012, 0.31])
    cb2 = fig.colorbar(bot_im, cax=cax2, extend="both")
    cb2.set_ticks([-30, -15, 0, 15, 30])
    cb2.set_label("Earlier due to UV (years)", fontsize=6.7, labelpad=4)
    cb2.ax.tick_params(labelsize=5.8, width=0.45, length=2.3)
    cb2.outline.set_linewidth(0.4)
    fig.text(0.50, 0.018, "Crossing requires CHS above the threshold for 5 consecutive years; changes are capped at +/-30 years.",
             ha="center", va="center", fontsize=5.7, color="#333333")

    png = FIG / "Figure3_time_of_emergence_and_UV_advance.png"
    pdf = FIG / "Figure3_time_of_emergence_and_UV_advance.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    np.savez_compressed(FIG / "Figure3_source_fields_0p5deg.npz",
                        lon=D["G"]["lon"], lat=D["G"]["lat"], tau=D["tau"],
                        **{f"emergence_{s}": emergence[s] for s in SCENARIOS},
                        **{f"emergence_no_uv_{s}": emergence0[s] for s in SCENARIOS},
                        **{f"uv_advance_{s}": advances[s] for s in SCENARIOS})
    return png, pdf


def haf_for(D, s, p, w, pct, P):
    q = P[(s, p)]
    base = D["G"]["B2d"] + D["G"]["P2d"] * ((1 - w) * q["st"]) + (1 - w) * q["u"]
    total = base + w * q["z"]
    # Recalibrate threshold for the same weight/percentile on the modern baseline.
    baseline_clim = np.zeros_like(base)
    n = 0
    for ss in SCENARIOS:
        st, uu = D["climate_rel"][ss]
        for y in range(2015, 2026):
            j = y - 2015
            baseline_clim += D["G"]["P2d"] * st[j] + uu[j]
            n += 1
    baseline_clim /= n
    b0 = D["G"]["B2d"] + (1 - w) * baseline_clim
    tau = weighted_percentile(b0[D["land"]], D["area"][D["land"]], pct)
    valid = D["land"] & np.isfinite(total)
    den = D["area"][valid].sum()
    h0 = float(D["area"][valid & (base < tau)].sum() / den)
    h1 = float(D["area"][valid & (total < tau)].sum() / den)
    return h0, h1


def figure4(D, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 5.8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    rows = list(csv.DictReader((FIG / "Figure1_source_data.csv").open(encoding="utf-8-sig")))
    traj = {s: [] for s in SCENARIOS}
    for r in rows:
        traj[r["scenario"]].append((int(r["year"]), float(r["ozone_change_pct"]),
                                     float(r["uv_change_pct"])))

    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.09,
                        wspace=0.28, hspace=0.34)
    axa, axb, axc, axd = axs.flat
    for ax in axs.flat:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)

    # a: ozone-UV trajectories.
    for s in SCENARIOS:
        a = np.asarray(traj[s])
        axa.plot(smooth11(a[:, 1]), smooth11(a[:, 2]), color=COLORS[s], lw=1.25,
                 label=LABELS[s])
        for yr in (2050, 2100):
            j = int(np.where(a[:, 0] == yr)[0][0])
            axa.scatter(smooth11(a[:, 1])[j], smooth11(a[:, 2])[j], s=14,
                        color=COLORS[s], edgecolor="white", linewidth=0.35, zorder=3)
    axa.axhline(0, color="0.65", lw=0.5); axa.axvline(0, color="0.65", lw=0.5)
    axa.set_xlabel("Estimated total-column ozone change (%)")
    axa.set_ylabel("Erythemal UV dose change (%)")
    axa.legend(frameon=False, loc="upper left")
    axa.set_title("Ozone-UV response trajectories", loc="left", fontweight="bold")

    # b: paired HAF, current CRITIC weight and 90th-percentile threshold.
    ypos = np.arange(6)
    labels = []; h0s = []; h1s = []; col = []
    for p in ("mid", "late"):
        for s in SCENARIOS:
            h0, h1 = haf_for(D, s, p, D["wuv"], 90, P)
            h0s.append(h0); h1s.append(h1); col.append(COLORS[s])
            labels.append(f"{LABELS[s]}  {'2041-60' if p == 'mid' else '2081-2100'}")
    for i, (h0, h1, cc) in enumerate(zip(h0s, h1s, col)):
        axb.plot([h0, h1], [i, i], color="0.65", lw=0.8, zorder=1)
        axb.scatter(h0, i, s=20, facecolor="white", edgecolor=cc, linewidth=0.8, zorder=2)
        axb.scatter(h1, i, s=20, facecolor=cc, edgecolor=cc, linewidth=0.6, zorder=3)
    axb.set_yticks(ypos, labels); axb.invert_yaxis(); axb.set_xlim(0, 1)
    axb.set_xlabel("Habitable Area Fraction")
    axb.set_title("Net HAF effect of the UV term", loc="left", fontweight="bold")
    axb.legend(handles=[Line2D([0], [0], marker="o", lw=0, markerfacecolor="white",
                               markeredgecolor="0.25", label="Without UV"),
                        Line2D([0], [0], marker="o", lw=0, markerfacecolor="0.25",
                               markeredgecolor="0.25", label="With UV")],
               frameon=False, loc="lower left")

    # c: sensitivity to UV weight at fixed 90th-percentile threshold.
    weights = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, D["wuv"]])
    sensitivity_rows = []
    for s in SCENARIOS:
        vals = []
        for w in weights:
            h0, h1 = haf_for(D, s, "late", float(w), 90, P)
            vals.append(h1 - h0)
            sensitivity_rows.append(dict(scenario=s, period="late", uv_weight=float(w),
                                         threshold_percentile=90, haf_no_uv=h0,
                                         haf_with_uv=h1, delta_haf=h1-h0))
        axc.plot(weights, vals, color=COLORS[s], lw=1.2, marker="o", ms=2.8,
                 label=LABELS[s])
    axc.axhline(0, color="0.5", lw=0.55)
    axc.axvline(D["wuv"], color="0.35", lw=0.65, ls=(0, (3, 2)))
    axc.text(D["wuv"], axc.get_ylim()[0], " CRITIC", rotation=90,
             ha="right", va="bottom", fontsize=5.5, color="0.35")
    axc.set_xlabel("UV weight")
    axc.set_ylabel("UV-induced change in HAF")
    axc.set_title("Weight sensitivity, 2081-2100", loc="left", fontweight="bold")
    axc.legend(frameon=False, loc="lower left")

    # d: sensitivity to habitability threshold at the current CRITIC UV weight.
    pcts = np.array([80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5])
    for s in SCENARIOS:
        vals = []
        for pct in pcts:
            h0, h1 = haf_for(D, s, "late", D["wuv"], float(pct), P)
            vals.append(h1 - h0)
            sensitivity_rows.append(dict(scenario=s, period="late",
                                         uv_weight=D["wuv"],
                                         threshold_percentile=float(pct),
                                         haf_no_uv=h0, haf_with_uv=h1,
                                         delta_haf=h1-h0))
        axd.plot(pcts, vals, color=COLORS[s], lw=1.2, marker="o", ms=2.8,
                 label=LABELS[s])
    axd.axhline(0, color="0.5", lw=0.55); axd.axvline(90, color="0.35", lw=0.65, ls=(0, (3, 2)))
    axd.set_xlabel("Baseline threshold percentile")
    axd.set_ylabel("UV-induced change in HAF")
    axd.set_title("Threshold sensitivity, 2081-2100", loc="left", fontweight="bold")
    axd.legend(frameon=False, loc="lower left")
    for letter, ax in zip("abcd", axs.flat):
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")

    png = FIG / "Figure4_UV_attribution_and_robustness.png"
    pdf = FIG / "Figure4_UV_attribution_and_robustness.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    csvp = FIG / "Figure4_sensitivity_source_data.csv"
    with csvp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader(); writer.writerows(sensitivity_rows)
    return png, pdf


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    D = common_setup()
    emergence, emergence0, period, haf2100 = analyse(D)
    f3png, f3pdf = figure3(D, emergence, emergence0, haf2100)
    f4png, f4pdf = figure4(D, period)
    meta = {
        "model": "UKESM1-0-LL r1i1p1f2", "years": [2015, 2100],
        "baseline": [2015, 2025], "tau": D["tau"],
        "uv_weight": D["wuv"], "uv_scale_J_m-2_d-1": D["uv_scale"],
        "threshold_crossing_rule": "first of five consecutive years above tau",
        "antarctica": "displayed with B=0 and forced climate/UV; excluded from HAF and threshold",
        "figure3": {"png": str(f3png), "pdf": str(f3pdf)},
        "figure4": {"png": str(f4png), "pdf": str(f4pdf)},
    }
    (FIG / "Figure3_4_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
