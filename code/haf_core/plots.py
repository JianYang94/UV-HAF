"""Nature-family vector figures (per the user's Springer/Nature figure guide).

Vector PDF backend, sans-serif <=7pt, RGB, 1-col=88mm / 2-col=180mm widths.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
ONE_COL, TWO_COL = 88 * MM, 180 * MM


def plot_gmst_fit(years, ens_gmst, obs_years, obs_gmst, calib_window, path):
    """Posterior GMST fan vs HadCRUT5; shade calibration vs withheld validation."""
    fig, ax = plt.subplots(figsize=(ONE_COL, 60 * MM))
    lo, mid, hi = np.percentile(ens_gmst, [5, 50, 95], axis=0)
    ax.fill_between(years, lo, hi, color="#c6dbef", label="model 5-95%")
    ax.plot(years, mid, color="#2171b5", lw=0.8, label="model median")
    ax.plot(obs_years, obs_gmst, color="k", lw=0.6, label="HadCRUT5")
    ax.axvspan(calib_window[0], calib_window[1], color="0.9", zorder=0)
    ax.axvline(calib_window[1], color="0.5", lw=0.5, ls="--")
    ax.text(calib_window[1] + 2, ax.get_ylim()[0] + 0.1, "withheld ->", fontsize=5)
    ax.set_xlim(1850, 2025); ax.set_xlabel("Year")
    ax.set_ylabel("GMST anomaly (K, vs 1850-1900)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_ohc_fit(years, ens_ohc, obs_years, obs_ohc, obs_sigma, window, path):
    """Posterior OHC fan vs NOAA/NCEI 0-2000 m, both on the 2005-2014 baseline."""
    fig, ax = plt.subplots(figsize=(ONE_COL, 60 * MM))
    # clip to the display window so the y-axis is not dominated by the far-future
    # cumulative OHC of the full 1750-2300 trajectory
    m = (years >= 2000) & (years <= 2030)
    yv = years[m]
    lo, mid, hi = np.percentile(ens_ohc[:, m], [5, 50, 95], axis=0)
    ax.fill_between(yv, lo, hi, color="#fdd0a2", label="model 5-95%")
    ax.plot(yv, mid, color="#e6550d", lw=0.8, label="model median")
    ax.errorbar(obs_years, obs_ohc, yerr=obs_sigma, fmt="o", ms=2, lw=0.5,
                color="k", capsize=1, label="NOAA/NCEI 0-2000 m")
    ax.axvspan(window[0], window[1], color="0.9", zorder=0)
    ax.set_xlim(2000, 2030); ax.set_xlabel("Year")
    ax.set_ylabel("OHC anomaly (ZJ, vs 2005-2014)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_haf(years, ens_haf, path, pctile_sens=None):
    """Headline HAF trajectory 1750-2300 with posterior envelope."""
    fig, ax = plt.subplots(figsize=(ONE_COL, 60 * MM))
    lo, mid, hi = np.percentile(ens_haf, [5, 50, 95], axis=0)
    ax.fill_between(years, lo, hi, color="#c7e9c0", label="5-95% (posterior)")
    ax.plot(years, mid, color="#238b45", lw=1.0, label="median")
    if pctile_sens:
        for p, h in pctile_sens.items():
            ax.plot(years, h, lw=0.4, ls=":", color="0.5")
        ax.plot([], [], lw=0.4, ls=":", color="0.5", label="tau = 80-99th pct")
    ax.set_xlim(1750, 2300); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Year"); ax.set_ylabel("Habitable Area Fraction")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


_SSP_STYLE = {
    "ssp126": ("#1f78b4", "SSP1-2.6"), "ssp245": ("#33a02c", "SSP2-4.5"),
    "ssp370": ("#ff7f00", "SSP3-7.0"), "ssp585": ("#e31a1c", "SSP5-8.5"),
}


def plot_haf_scenarios(years, scen_haf, path):
    """HAF trajectory under each SSP at the posterior mean (scenario spread)."""
    fig, ax = plt.subplots(figsize=(ONE_COL, 60 * MM))
    for s, h in scen_haf.items():
        color, label = _SSP_STYLE.get(s, ("0.3", s))
        ax.plot(years, h, lw=1.0, color=color, label=label)
    ax.set_xlim(1750, 2300); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Year"); ax.set_ylabel("Habitable Area Fraction")
    ax.legend(frameon=False, loc="lower left", title="scenario")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_haf_weight_ensemble(years, haf_weights, haf_equal, path):
    """HAF envelope from a weight ENSEMBLE vs the equal-weight fallback.

    Demonstrates the proposal's Major-1 fix: rather than a single weight set, an
    ensemble of weightings (here a flat draw over the simplex of the carried
    variables, standing in for the not-yet-available learned RF/WHI weights) is
    propagated through the HAF, and the equal-weight fallback is shown for
    reference.
    """
    fig, ax = plt.subplots(figsize=(ONE_COL, 60 * MM))
    lo, mid, hi = np.percentile(haf_weights, [5, 50, 95], axis=0)
    q25, q75 = np.percentile(haf_weights, [25, 75], axis=0)
    ax.fill_between(years, lo, hi, color="#dadaeb", label="5-95% (weight ensemble)")
    ax.fill_between(years, q25, q75, color="#bcbddc", label="25-75%")
    ax.plot(years, mid, color="#6a51a3", lw=0.9, label="weight-ensemble median")
    ax.plot(years, haf_equal, color="k", lw=0.9, ls="--", label="equal-weight fallback")
    ax.set_xlim(1750, 2300); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Year"); ax.set_ylabel("Habitable Area Fraction")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_chs_map(field2d, G, year, path):
    """2-D Composite Hazard Score map at `year` on the real 0.5-deg grid.

    Land only (ocean blank). Uses a Robinson projection with coastlines if
    cartopy is importable, else a plain lon/lat pcolormesh.
    """
    lon, lat = G["lon"], G["lat"]
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(TWO_COL, 95 * MM))
        ax = plt.axes(projection=ccrs.Robinson())
        im = ax.pcolormesh(lon, lat, field2d, transform=ccrs.PlateCarree(),
                           cmap="YlOrRd", shading="auto", rasterized=True)
        ax.coastlines(linewidth=0.3)
        ax.set_global()
    except Exception:
        fig, ax = plt.subplots(figsize=(TWO_COL, 95 * MM))
        im = ax.pcolormesh(lon, lat, field2d, cmap="YlOrRd", shading="auto",
                           rasterized=True)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(f"Composite Hazard Score ({year}, SSP2-4.5)")
    # rasterise the 0.5-deg field at 300 dpi (text/coastlines stay vector)
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)


def plot_toe_map(toe2d, G, path, ssp_label="SSP2-4.5", percentile=90):
    """Time-of-Emergence map: first year each land cell crosses tau.

    Land that never crosses in 1750-2300 is drawn grey; emerging land is coloured
    by emergence year (earlier = more urgent). Robinson if cartopy is importable.
    """
    lon, lat = G["lon"], G["lat"]
    import copy as _copy
    from matplotlib.patches import Patch
    # Continuous scale spans the emergence years only (2050-2300); its dark-purple
    # top must NOT be reused for the "already lost" cells, or the two are confused.
    ALREADY_LOST = "#111111"   # near-black: already hazardous at start_year
    NEVER = "#d9d9d9"          # light grey: never crosses within the horizon
    cmap = _copy.copy(plt.get_cmap("plasma_r"))  # version-robust (mpl 3.3-3.9)
    cmap.set_bad(NEVER)                            # never-emerging land shows grey
    cmap.set_under(ALREADY_LOST)                   # already hazardous in start_year
    # Start the colour scale above start_year so the colourbar maps only to genuine
    # emergence years; anything <= start_year falls into the distinct under-colour.
    vmin, vmax = 2021, 2300
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(TWO_COL, 95 * MM))
        ax = plt.axes(projection=ccrs.Robinson())
        im = ax.pcolormesh(lon, lat, toe2d, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                           rasterized=True)
        ax.coastlines(linewidth=0.3)
        ax.set_global()
    except Exception:
        fig, ax = plt.subplots(figsize=(TWO_COL, 95 * MM))
        im = ax.pcolormesh(lon, lat, toe2d, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, extend="min")
    cb.set_label(f"Year CHS crosses habitability threshold ({ssp_label})")
    # Explicit legend for the two off-scale categories so neither is read off the
    # continuous colourbar (whose top is a distinct dark purple = year 2300).
    ax.legend(handles=[Patch(facecolor=ALREADY_LOST, edgecolor="none",
                             label="Already lost (≤2020)"),
                       Patch(facecolor=NEVER, edgecolor="none",
                             label="Never crosses by 2300")],
              loc="lower left", frameon=False, fontsize=5,
              handlelength=1.2, borderaxespad=0.2)
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)


def plot_weight_methods(methods, cvars, haf2100, path):
    """Objective-weighting agreement figure.

    Left: the weight vector each objective method assigns to the six carried
    variables (grouped bars). Right: HAF-2100 under each weighting. Demonstrates
    that CRITIC/entropy/PCA and the equal-weight reference give near-identical
    outcomes -- the collinearity result the paper reports.
    """
    order = list(cvars)
    labels = list(methods.keys())
    fig, (axw, axh) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    x = np.arange(len(order)); nb = len(labels); bw = 0.8 / nb
    palette = ["k", "#3182bd", "#e6550d", "#31a354", "#756bb1"]
    for j, m in enumerate(labels):
        w = [methods[m][k] for k in order]
        axw.bar(x + (j - (nb - 1) / 2) * bw, w, bw, label=m,
                color=palette[j % len(palette)])
    axw.axhline(1.0 / len(order), color="0.5", lw=0.5, ls=":")
    axw.set_xticks(x); axw.set_xticklabels([c.upper() for c in order], rotation=30,
                                           ha="right", fontsize=5)
    axw.set_ylabel("CHS weight"); axw.legend(frameon=False, ncol=2, fontsize=5)
    hy = [haf2100[m] for m in labels]
    axh.bar(range(len(labels)), hy, color=[palette[j % len(palette)]
                                           for j in range(len(labels))])
    axh.set_xticks(range(len(labels)))
    axh.set_xticklabels(labels, rotation=30, ha="right", fontsize=5)
    axh.set_ylabel("HAF 2100 (SSP2-4.5)")
    axh.set_ylim(0, max(hy) * 1.25 + 1e-3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_rf_importances(rf, path):
    """Permutation importances of the Component-1 Random Forest (WHI ~ predictors)."""
    names = rf["names"]; imp = rf["importance"]
    n = min(len(names), 14)
    names, imp = names[:n][::-1], imp[:n][::-1]
    fig, ax = plt.subplots(figsize=(ONE_COL, 75 * MM))
    ax.barh(range(n), imp, color="#3182bd")
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("permutation importance (normalised)")
    ax.set_title(f"Component-1 RF: WHI from independent predictors\n"
                 f"held-out $R^2$={rf['cv_r2_heldout']:.2f} (independent) vs "
                 f"{rf['cv_r2_with_constituents']:.2f} (+constituents); "
                 f"leakage={rf['leakage_r2']:.2f}", fontsize=5.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_stationarity(r, path):
    """Pattern-stationarity figure (Major-5 demonstration).

    Left: the observed reference warming pattern (local K per global K). Right: a
    Taylor diagram of the warming-level patterns vs the reference, plus the
    the assumed tier-(i) pattern, showing how stationary the pattern is
    across the observed warming range.
    """
    lat, lon, beta = r["lat"], r["lon"], r["beta_ref"]
    fig = plt.figure(figsize=(TWO_COL, 60 * MM))
    try:
        import cartopy.crs as ccrs
        axL = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
        im = axL.pcolormesh(lon, lat, beta, transform=ccrs.PlateCarree(),
                            cmap="RdYlBu_r", vmin=0, vmax=3, shading="auto", rasterized=True)
        axL.coastlines(linewidth=0.3); axL.set_global()
    except Exception:
        axL = fig.add_subplot(1, 2, 1)
        im = axL.pcolormesh(lon, lat, beta, cmap="RdYlBu_r", vmin=0, vmax=3,
                            shading="auto", rasterized=True)
        axL.set_xlabel("Longitude"); axL.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=axL, shrink=0.55, pad=0.02)
    cb.set_label("local warming per K global")
    axL.set_title("observed warming pattern $\\beta_{ref}$", fontsize=6)

    axR = fig.add_subplot(1, 2, 2)
    th = np.linspace(0, np.pi / 2, 120)
    for rr in (0.5, 1.0, 1.5):
        axR.plot(rr * np.cos(th), rr * np.sin(th), color="0.85", lw=0.5, zorder=0)
    for c in (0.9, 0.95, 0.99):
        ang = np.arccos(c)
        axR.plot([0, 1.65 * np.cos(ang)], [0, 1.65 * np.sin(ang)], color="0.85",
                 lw=0.4, ls=":", zorder=0)
        axR.text(1.68 * np.cos(ang), 1.68 * np.sin(ang), f"{c}", fontsize=4.5,
                 color="0.5", ha="left", va="center")
    axR.text(1.5, 0.05, "corr $\\rightarrow$", fontsize=4.5, color="0.5")
    axR.plot(1, 0, marker="*", ms=10, color="k", zorder=5, label="reference $\\beta_{ref}$")
    cmap = plt.get_cmap("viridis")
    el = r["warm_epochs_list"]
    for i, e in enumerate(el):
        c, sr, _ = r["stats"][e]
        ang = np.arccos(np.clip(c, -1, 1))
        axR.plot(sr * np.cos(ang), sr * np.sin(ang), marker="o", ms=4,
                 color=cmap(i / max(1, len(el) - 1)), zorder=5,
                 label=f"{r['warming'][e]:.1f} K (r={c:.3f})")
    pc, psr = r["proto_point"]
    ang = np.arccos(np.clip(pc, -1, 1))
    axR.plot(psr * np.cos(ang), psr * np.sin(ang), marker="D", ms=4.5,
             color="#cb181d", zorder=5, label=f"assumed pattern (r={pc:.2f})")
    axR.set_xlim(0, 1.8); axR.set_ylim(0, 1.0); axR.set_aspect("equal")
    axR.set_xlabel("normalised standard deviation")
    axR.set_title("Taylor diagram: pattern vs warming level", fontsize=6)
    axR.legend(frameon=False, fontsize=4.6, loc="upper right")
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)


def plot_niche_vs_haf(r, path, labels=("a", "b")):
    """HAF vs human-climate-niche validation (Major-3 demonstration).

    Left: near-unlivable land fraction (MAT > 29 degC, Xu 2020) over time under
    the four SSPs, with Xu's ~19%-by-2070 (RCP8.5) reference marked. Right: the
    headline HAF vs the independent niche-habitable fraction under SSP2-4.5.
    ``labels`` sets the two panel letters (default a,b for the standalone figure;
    the Perspective's stacked Fig. 6 uses c,d so the four sub-plots read a-d).
    """
    years = r["years"]; per = r["per_ssp"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM))
    for s in per:
        color, label = _SSP_STYLE.get(s, ("0.3", s))
        axL.plot(years, per[s]["unlivable"] * 100, lw=1.0, color=color, label=label)
    axL.plot([2070], [r["xu2020_rcp85_2070_unlivable"] * 100], marker="*",
             ms=9, color="k", ls="none", label="Xu 2020 (RCP8.5, 2070)")
    axL.set_xlim(1950, 2300); axL.set_ylim(0, 60)
    axL.set_xlabel("Year"); axL.set_ylabel("Near-unlivable land (MAT > 29$^\\circ$C, %)")
    axL.legend(frameon=False, loc="upper left", fontsize=5)

    axR.plot(years, r["haf245"], lw=1.1, color="#238b45", label="HAF (this model)")
    axR.plot(years, per["ssp245"]["niche_hab"], lw=1.1, color="#6a51a3", ls="--",
             label="niche-habitable (Xu niche)")
    axR.set_xlim(1850, 2300); axR.set_ylim(0, 1.02)
    axR.set_xlabel("Year"); axR.set_ylabel("Habitable land fraction (SSP2-4.5)")
    axR.legend(frameon=False, loc="lower left", fontsize=5)
    for ax, lab in zip((axL, axR), labels):
        ax.text(0.0, 1.04, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_structural(r, path, labels=("a", "b")):
    """Leave-one-reconstruction-out structural-error figure (Major-4 demonstration).

    Left: the three independent reconstructions + the single OHC-calibrated emulator
    GMST, with the 1880-1980 calibration and 1981-2020 withheld windows shaded.
    Right: per-reconstruction in-sample vs out-of-sample RMSE, with the derived
    sigma_struct and the value assumed in the SMC marked.
    """
    names = r["names"]; recon = r["recon"]; gm = r["gm"]; years = r["years"]
    colors = {"HadCRUT5": "#000000", "GISTEMP": "#2171b5", "Berkeley": "#cb181d"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    cw, ow = r["cal_window"], r["oos_window"]
    axL.axvspan(cw[0], cw[1], color="0.92", zorder=0)
    axL.axvspan(ow[0], ow[1], color="#fde0dd", zorder=0)
    for n in names:
        oy, og, _ = recon[n]
        held = r["per_reconstruction"][n]["held_out"]
        axL.plot(oy, og, lw=0.7, color=colors.get(n, "0.3"),
                 label=n + (" (held out)" if held else " (calib.)"))
    axL.plot(years, gm, lw=1.1, color="#238b45", label="calibrated emulator")
    axL.set_xlim(1860, 2020); axL.set_ylim(-0.6, 1.6)
    axL.set_xlabel("Year"); axL.set_ylabel("GMST anomaly (K, vs 1880-1920)")
    axL.text(cw[1] - 2, -0.5, "calibration", fontsize=5, color="0.4", ha="right")
    axL.text(ow[0] + 1, -0.5, "withheld", fontsize=5, color="#cb181d")
    axL.legend(frameon=False, loc="upper left", fontsize=5)

    x = np.arange(len(names)); bw = 0.38
    ins = [r["per_reconstruction"][n]["rmse_insample_K"] for n in names]
    oos = [r["per_reconstruction"][n]["rmse_oos_K"] for n in names]
    axR.bar(x - bw / 2, ins, bw, color="0.7", label="in-sample 1880-1980")
    axR.bar(x + bw / 2, oos, bw, color="#cb181d", label="OOS 1981-2020")
    axR.axhline(r["sigma_struct_assumed_in_smc_K"], color="#238b45", lw=0.9, ls="--",
                label=f"assumed $\\sigma_{{struct}}$={r['sigma_struct_assumed_in_smc_K']:.2f}")
    axR.set_xticks(x); axR.set_xticklabels(names, rotation=20, ha="right")
    axR.set_ylabel("GMST RMSE (K)")
    axR.legend(frameon=False, fontsize=5, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    for ax, lab in zip((axL, axR), labels):
        ax.text(0.0, 1.04, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_cropyield_validation(r, path):
    """CHS vs independent crop-yield instability (Major-3, impacts half).

    Left: present-day CHS binned into deciles vs the area-weighted mean detrended
    interannual yield CV (GDHY), pooled over the four staple crops -- a U-shaped,
    non-monotone curve whose EXTREMES separate robustly (top/bottom ratio). Error
    bars are 10 deg spatial-block bootstrap (not naive SEM), so they reflect spatial
    autocorrelation. Right: per-crop Spearman of CHS vs CV and vs mean yield, plus
    the spatial decomposition (WHI tier-(ii) field alone, warming pattern alone).
    """
    chs_v, cv_v, area_v = r["_chs_v"], r["_cv_v"], r["_area_v"]
    lat_v, lon_v = r["_lat_v"], r["_lon_v"]
    pl = r["pooled"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # decile relationship (area-weighted mean CV per CHS decile), with a 10-deg
    # spatial-block bootstrap of each decile mean for honest (autocorrelation-aware) bars
    edges = np.percentile(chs_v, np.arange(0, 101, 10))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(chs_v, edges) - 1, 0, 9)
    rng = np.random.default_rng(0)
    centers, means, lo_e, hi_e = [], [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() < 5:
            continue
        cvb, ab = cv_v[m], area_v[m]
        latb, lonb = lat_v[m], lon_v[m]
        bid = (np.floor(latb / 10.0).astype(int) * 100000
               + np.floor(lonb / 10.0).astype(int))
        groups = {}
        for k, g in enumerate(bid):
            groups.setdefault(g, []).append(k)
        gidx = [np.array(v) for v in groups.values()]
        mu = np.average(cvb, weights=ab) * 100
        boot = []
        for _ in range(200):
            sel = np.concatenate([gidx[p] for p in rng.integers(0, len(gidx), len(gidx))])
            boot.append(np.average(cvb[sel], weights=ab[sel]) * 100)
        centers.append(b + 1); means.append(mu)
        lo_e.append(mu - np.percentile(boot, 5))
        hi_e.append(np.percentile(boot, 95) - mu)
    axL.errorbar(centers, means, yerr=[lo_e, hi_e], marker="o", ms=3, lw=1.0,
                 color="#cb181d", capsize=2)
    axL.set_xlabel("Composite Hazard Score decile (present-day)")
    axL.set_ylabel("Detrended yield CV (%, area-weighted)")
    lo, hi = pl["spearman_chs_vs_cv_block_ci90"]
    axL.set_title(f"CHS vs crop-yield instability\nSpearman $r$="
                  f"{pl['spearman_chs_vs_cv']:.2f} [{lo:.2f}, {hi:.2f}]; "
                  f"top/bottom $\\times${pl['decile_cv_ratio_top_over_bottom']:.2f}",
                  fontsize=6)
    axL.set_xticks(range(1, 11))

    # per-crop Spearman + decomposition
    crops = r["crops"]
    rcv = [r["per_crop"][c]["spearman_chs_vs_cv"] for c in crops]
    rmy = [r["per_crop"][c]["spearman_chs_vs_meanyield"] for c in crops]
    labels = list(crops) + ["WHI\nalone", "warming\nalone"]
    x = np.arange(len(labels)); bw = 0.38
    cv_bars = rcv + [pl["spearman_whi_alone_vs_cv"],
                     pl["spearman_warmpattern_alone_vs_cv"]]
    my_bars = rmy + [np.nan, np.nan]
    axR.bar(x - bw / 2, cv_bars, bw, color="#cb181d", label="vs yield CV")
    axR.bar(x + bw / 2, my_bars, bw, color="#2171b5", label="vs mean yield")
    axR.axhline(0, color="0.5", lw=0.5)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=4.8)
    axR.set_ylabel("Spearman rank correlation with CHS")
    axR.set_title("by crop, and CHS spatial components", fontsize=6)
    axR.legend(frameon=False, fontsize=5, loc="lower left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)


def plot_posterior(post, path):
    """Prior vs posterior for each calibrated parameter."""
    from . import emulator
    names = post["names"]
    fig, axes = plt.subplots(1, len(names), figsize=(ONE_COL, 45 * MM))
    if len(names) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    prior = emulator.sample_prior(rng, 5000)
    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(prior[:, j], bins=40, density=True, color="0.8",
                label="prior", histtype="stepfilled")
        ax.hist(post["theta"][:, j], bins=30, density=True, weights=post["weights"],
                color="#6a51a3", alpha=0.8, label="posterior", histtype="stepfilled")
        ax.set_xlabel(name); ax.set_yticks([])
        if j == 0:
            ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
