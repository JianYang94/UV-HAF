"""Create supplementary Figures S3-S4 for the UV-extended CHS analysis.

S3 combines baseline CHS6/CHS7 consistency with the pure UV term.
S4 uses one-factor-at-a-time counterfactual UV calculations for ozone,
cloud attenuation and aerosol optical depth.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
DATA = ROOT / "data" / "derived"
sys.path.insert(0, str(HERE))

import analysis_helpers as base
import plot_main_figures_4_5 as corrected
try:
    import uv_reconstruction as fut
except (ModuleNotFoundError, PermissionError) as exc:
    # Plotting from the archived counterfactual cache does not require NetCDF.
    # Provide only the interface needed to load the archived arrays
    # in the lightweight plotting environment.
    if isinstance(exc, ModuleNotFoundError) and exc.name != "netCDF4":
        raise
    class _CachedOnly:
        MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)
    fut = _CachedOnly()
try:
    import spectral_uv as uvmodel
except ModuleNotFoundError as exc:
    if exc.name != "netCDF4":
        raise
    uvmodel = None

SCENARIOS = base.SCENARIOS
LABELS = base.LABELS
COLORS = base.COLORS
PERIODS = {"mid": (2041, 2060), "late": (2081, 2100)}
FACTORS = ("ozone", "cloud", "aerosol")
FACTOR_LABELS = {"ozone": "Ozone", "cloud": "Cloud", "aerosol": "Aerosol"}
FACTOR_COLORS = {"ozone": "#7B3294", "cloud": "#008837", "aerosol": "#E66101"}
MONTH_DAYS = fut.MONTH_DAYS


def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "Arial", "font.size": 6.5,
                         "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 5.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42})


def s3_data():
    # Use the same annual-cache baseline, UV scaling and spatial setup as the
    # main figures.  A period-climatology shortcut changes the reference field
    # slightly and therefore does not reproduce the released Figure 4 arrays.
    D = base.common_setup()
    S = corrected.six_variable_setup(D)
    # The UV anomaly averages to zero over the same 2015-2025 reference sample.
    chs6 = np.asarray(S["baseline_disp"], float)
    chs7 = np.asarray(D["baseline_disp"], float)
    delta = chs7 - chs6
    # Recompute the direct UV fields with the current standardisation rather
    # than reading an older Figure 3 cache whose schema and scaling may differ.
    _, pure, _, _, _, _ = corrected.compute(D, S)
    src = FIG / "Supplementary_Figure_S3_source_fields_0p5deg.npz"
    if not src.exists():
        src = DATA / "Supplementary_Figure_S3_source_fields_0p5deg.npz"
    np.savez_compressed(src, lat=D["G"]["lat"], lon=D["G"]["lon"],
                        chs6_baseline=chs6, chs7_baseline=chs7,
                        delta_chs_baseline=delta,
                        **{f"pure_uv_{s}_{p}": pure[(s, p)] for s in SCENARIOS for p in PERIODS})
    return D, chs6, chs7, delta, pure


def plot_s3(D, chs6, chs7, delta, pure):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    style()
    fig, axes = plt.subplots(3, 3, figsize=(7.205, 5.45))
    mask = D["display_land"]
    seq_vals = np.r_[chs6[mask & np.isfinite(chs6)], chs7[mask & np.isfinite(chs7)]]
    lo, hi = np.nanpercentile(seq_vals, [1, 99])
    seq = LinearSegmentedColormap.from_list("chs", ["#f7fbff", "#6baed6", "#08306b"])
    im_chs = None
    for c, (field, title) in enumerate(((chs6, "CHS$_6$"),
                                        (chs7, "CHS$_7$"))):
        im_chs = base.draw_robinson(axes[0, c], D["G"]["lon"], D["G"]["lat"],
                                    np.where(mask, field, np.nan), mask,
                                    cmap=seq, norm=Normalize(lo, hi))
        axes[0, c].set_title(title, weight="bold", pad=2.5)
    div = LinearSegmentedColormap.from_list("div", ["#2166ac", "#f7f7f7", "#b2182b"])
    # The baseline difference is negative everywhere and spans only about
    # 0.01 CHS units.  Retain the spatial map, but use its own narrow sequential
    # scale so the small zonal variation is not hidden by a zero-centred scale.
    dlo, dhi = np.nanpercentile(delta[mask & np.isfinite(delta)], [1, 99])
    delta_cmap = LinearSegmentedColormap.from_list(
        "baseline_delta", ["#2166ac", "#deebf7"])
    im_d = base.draw_robinson(
        axes[0, 2], D["G"]["lon"], D["G"]["lat"],
        np.where(mask, delta, np.nan), mask,
        cmap=delta_cmap, norm=Normalize(dlo, dhi))
    axes[0, 2].set_title(r"$\Delta$CHS = CHS$_7$ − CHS$_6$",
                         weight="bold", pad=2.5)

    uvvals = np.concatenate([np.abs(v[mask & np.isfinite(v)]) for v in pure.values()])
    ulim = float(np.nanpercentile(uvvals, 99))
    unorm = TwoSlopeNorm(vmin=-ulim, vcenter=0, vmax=ulim)
    im_uv = None
    for r, p in enumerate(("mid", "late"), start=1):
        for c, s in enumerate(SCENARIOS):
            f = np.where(mask, pure[(s, p)], np.nan)
            im_uv = base.draw_robinson(axes[r, c], D["G"]["lon"], D["G"]["lat"],
                                       f, mask, cmap=div, norm=unorm)
            mean = np.average(f[D["land"]], weights=D["area"][D["land"]])
            axes[r, c].set_title(f"{LABELS[s]}\nmean $C_{{UV}}$ = {mean:+.3f}",
                                 weight="bold", pad=2.2, linespacing=1.1)
    for i, ax in enumerate(axes.flat):
        ax.text(.01, .97, chr(97+i), transform=ax.transAxes, fontsize=8,
                weight="bold", ha="left", va="top")
    fig.subplots_adjust(left=.047, right=.965, bottom=.105, top=.975,
                        wspace=.035, hspace=.27)
    row_labels=("Baseline (2015–2025)","Mid-century (2041–2060)","Late-century (2081–2100)")
    for r,label in enumerate(row_labels):
        pos=axes[r,0].get_position(); y=pos.y0+pos.height/2
        fig.text(.018,y,label,rotation=90,weight="bold",ha="center",va="center")
    c1 = fig.add_axes([.15, .715, .36, .012]); cb1 = fig.colorbar(im_chs, cax=c1, orientation="horizontal")
    cb1.set_label("CHS", labelpad=1); c1.xaxis.set_ticks_position("bottom"); c1.xaxis.set_label_position("top")
    c2 = fig.add_axes([.69, .715, .20, .012]); cb2 = fig.colorbar(im_d, cax=c2, orientation="horizontal", extend="both")
    cb2.set_label(r"Baseline $\Delta$CHS (narrow range)", labelpad=1)
    delta_ticks = np.linspace(dlo, dhi, 3)
    cb2.set_ticks(delta_ticks)
    cb2.ax.set_xticklabels([f"{value:.3f}".replace("-", "−")
                            for value in delta_ticks])
    c2.xaxis.set_ticks_position("bottom"); c2.xaxis.set_label_position("top")
    c3 = fig.add_axes([.25, .067, .50, .013]); cb3 = fig.colorbar(im_uv, cax=c3, orientation="horizontal", extend="both")
    cb3.set_label(r"$C_{UV}=w_{UV}Z_{UV}$", labelpad=1)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"Supplementary_Figure_S3_baseline_and_pure_UV.{ext}",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    dmin = float(np.nanmin(delta[mask & np.isfinite(delta)]))
    dmax = float(np.nanmax(delta[mask & np.isfinite(delta)]))


def drivers(scenario, period):
    rsds, lat, lon = fut.climatology(scenario, "rsds", period)
    rsdscs, _, _ = fut.climatology(scenario, "rsdscs", period)
    rsuscs, _, _ = fut.climatology(scenario, "rsuscs", period)
    huss, _, _ = fut.climatology(scenario, "huss", period)
    ps, _, _ = fut.climatology(scenario, "ps", period)
    aod, _, _ = fut.climatology(scenario, "od550aer", period)
    ozone = fut.ozone_du(scenario, period, ps)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmf = np.clip(rsds / rsdscs, 0, 1.2)
        alb = np.clip(rsuscs / rsdscs, 0, 1)
    return dict(lat=lat, lon=lon, ozone=ozone, cmf=cmf,
                aod=np.clip(aod, .001, 2.), alb=alb,
                ps=ps, water=np.clip(huss*ps/fut.G/10., .02, 10.))


def dose(d):
    lat, lon = d["lat"], d["lon"]
    ny, nx = len(lat), len(lon)
    phi = np.deg2rad(np.repeat(lat, nx)); monthly=[]
    for mi, day in enumerate(uvmodel.DAYS):
        dec=np.deg2rad(23.44)*np.sin(2*np.pi*(day-81)/365.); total=np.zeros(ny*nx)
        for hour in np.arange(.5,24,1):
            ha=np.deg2rad(15*(hour-12)); cz=np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.cos(ha); ok=cz>0
            if not ok.any(): continue
            f={"sza":np.rad2deg(np.arccos(np.clip(cz[ok],-1,1))),
               "ozone":np.clip(d["ozone"][mi].ravel()[ok],100,600),
               "aod":d["aod"][mi].ravel()[ok], "ssa":np.full(ok.sum(),.95),
               "alpha":np.full(ok.sum(),1.3), "albedo":d["alb"][mi].ravel()[ok],
               "pressure":d["ps"][mi].ravel()[ok]/100., "cot":np.zeros(ok.sum()),
               "reff":np.full(ok.sum(),10.), "cloud_fraction":np.zeros(ok.sum()),
               "water_cm":d["water"][mi].ravel()[ok], "day":np.full(ok.sum(),day)}
            total[ok] += uvmodel.predict_erythema(uvmodel.features(f))*d["cmf"][mi].ravel()[ok]*3600
        monthly.append(total.reshape(ny,nx))
    return np.average(np.asarray(monthly), axis=0, weights=MONTH_DAYS)


def replace_one(base_d, future_d, factor):
    d={k:(np.array(v,copy=True) if isinstance(v,np.ndarray) else v) for k,v in base_d.items()}
    if factor == "ozone": d["ozone"] = future_d["ozone"]
    elif factor == "cloud": d["cmf"] = future_d["cmf"]
    elif factor == "aerosol": d["aod"] = future_d["aod"]
    return d


def counterfactual(D, scenario, period, baseline_d, baseline_uv):
    fd = drivers(scenario, period)
    out={}
    for factor in FACTORS:
        native = dose(replace_one(baseline_d, fd, factor)) - baseline_uv
        rg = base.regrid(native, fd["lat"], fd["lon"], D["G"]["lat"], D["G"]["lon"])
        out[factor] = D["wuv"] * rg / D["uv_scale"]
    return out


def s4_data(D):
    cache = DATA / "Supplementary_Figure_S4_source_fields_0p5deg.npz"
    if not cache.exists():
        cache = OUT / "supplementary_s4_counterfactual.npz"
    if cache.exists():
        z=np.load(cache); maps={(p,f):z[f"map_{p}_{f}"] for p in PERIODS for f in FACTORS}
        years=z["time_years"]; series={(s,f):z[f"series_{s}_{f}"] for s in SCENARIOS for f in FACTORS}
        return maps, years, series
    b = drivers("ssp245", (2015, 2025)); buv=dose(b)
    maps={}
    for p, pr in PERIODS.items(): maps.update({(p,f):v for f,v in counterfactual(D,"ssp245",pr,b,buv).items()})
    # Non-overlapping decadal climatologies: a true counterfactual time series,
    # not a correlation-based attribution.
    windows=[(2015,2024),(2025,2034),(2035,2044),(2045,2054),(2055,2064),
             (2065,2074),(2075,2084),(2085,2094),(2095,2100)]
    years=np.array([(a+b_)/2 for a,b_ in windows]); series={(s,f):[] for s in SCENARIOS for f in FACTORS}
    land,area=D["land"],D["area"]
    for s in SCENARIOS:
        for win in windows:
            cf=counterfactual(D,s,win,b,buv)
            for f in FACTORS: series[(s,f)].append(np.average(cf[f][land],weights=area[land]))
    series={k:np.asarray(v) for k,v in series.items()}
    np.savez_compressed(cache,time_years=years,
                        **{f"map_{p}_{f}":maps[(p,f)] for p in PERIODS for f in FACTORS},
                        **{f"series_{s}_{f}":series[(s,f)] for s in SCENARIOS for f in FACTORS})
    return maps,years,series


def plot_s4(D,maps,years,series):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    style(); div=LinearSegmentedColormap.from_list("div",["#2166ac","#f7f7f7","#b2182b"])
    vals=np.concatenate([np.abs(v[D["display_land"] & np.isfinite(v)]) for v in maps.values()]); lim=float(np.nanpercentile(vals,99))
    norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    # Match Figure S3: two period rows by three driver columns.  A dedicated
    # colour-bar band separates the maps from the time-series diagnostics.
    fig=plt.figure(figsize=(7.205,5.65)); gs=fig.add_gridspec(
        4,3,height_ratios=[1,1,.14,.82],left=.050,right=.985,
        bottom=.075,top=.955,wspace=.035,hspace=.22)
    last=None; map_axes=np.empty((2,3),dtype=object)
    for r,p in enumerate(("mid","late")):
        for c,f in enumerate(FACTORS):
            ax=fig.add_subplot(gs[r,c]); map_axes[r,c]=ax; field=np.where(D["display_land"],maps[(p,f)],np.nan)
            last=base.draw_robinson(ax,D["G"]["lon"],D["G"]["lat"],field,D["display_land"],cmap=div,norm=norm)
            mean=np.average(field[D["land"]],weights=D["area"][D["land"]])
            mean_label = f"{mean:+.3f}".replace("-", "−")
            ax.set_title(f"{FACTOR_LABELS[f]}\nmean = {mean_label}",weight="bold",pad=2,linespacing=1.05)
            ax.text(.01,.97,chr(97+r*3+c),transform=ax.transAxes,fontsize=8,weight="bold",ha="left",va="top")
    ts_axes=[]
    all_ts=np.concatenate([series[(s,f)] for s in SCENARIOS for f in FACTORS]); span=float(np.nanmax(all_ts)-np.nanmin(all_ts)); pad=max(.03,.06*span)
    common_ylim=(min(0.,float(np.nanmin(all_ts)))-pad,float(np.nanmax(all_ts))+pad)
    for i,s in enumerate(SCENARIOS):
        ax=fig.add_subplot(gs[3,i]); ts_axes.append(ax)
        for f in FACTORS: ax.plot(years,series[(s,f)],color=FACTOR_COLORS[f],lw=1.1,marker="o",ms=2,label=FACTOR_LABELS[f])
        ax.axhline(0,color="#777",lw=.5); ax.set_title(LABELS[s],weight="bold",pad=1.5); ax.set_xlim(2017,2124); ax.set_xticks([2025,2050,2075,2100]); ax.set_ylim(*common_ylim); ax.tick_params(direction="out",length=1.5,pad=2)
        ax.set_xlabel("Year")
        if i==2:
            for f in FACTORS:
                label_dy = -7 if f == "aerosol" else 0
                ax.annotate(FACTOR_LABELS[f],xy=(years[-1],series[(s,f)][-1]),
                            xytext=(4,label_dy),textcoords="offset points",ha="left",va="center",
                            color=FACTOR_COLORS[f],fontsize=5.6,clip_on=False)
        ax.spines[["top","right"]].set_visible(False)
    group_y0=ts_axes[0].get_position().y0; group_y1=ts_axes[0].get_position().y1
    fig.text(.018,(group_y0+group_y1)/2,"Mean contribution to CHS over land",
             rotation=90,ha="center",va="center",fontsize=6.2)
    # Panel letter labels the full three-panel time-series group; keep it
    # outside the plotting area so it cannot collide with y ticks or data.
    first_pos=ts_axes[0].get_position()
    fig.text(first_pos.x0,group_y1+.012,"g",fontsize=8,weight="bold",
             ha="left",va="bottom")
    for r,label in enumerate(("SSP2-4.5 mid-century (2041–2060)",
                              "SSP2-4.5 late-century (2081–2100)")):
        pos=map_axes[r,0].get_position(); y=pos.y0+pos.height/2
        fig.text(.017,y,label,rotation=90,weight="bold",ha="center",va="center")
    cpos=gs[2,:].get_position(fig)
    cax=fig.add_axes([cpos.x0+.17*cpos.width,cpos.y0+.64*cpos.height,
                      .66*cpos.width,.22*cpos.height])
    cb=fig.colorbar(last,cax=cax,orientation="horizontal",extend="both")
    cb.set_label("Counterfactual contribution to CHS",labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    for ext in ("png","pdf"): fig.savefig(FIG/f"Supplementary_Figure_S4_UV_driver_attribution.{ext}",dpi=600 if ext=="png" else None)
    plt.close(fig)


def main():
    if not (OUT / "uv_ssp126_2015_2015.npz").exists():
        from source_data_reproduction import supplementary_s3_s4
        supplementary_s3_s4()
        print("created S3 and S4 from archived source data", flush=True)
        return
    D,chs6,chs7,delta,pure=s3_data(); plot_s3(D,chs6,chs7,delta,pure)
    maps,years,series=s4_data(D); plot_s4(D,maps,years,series)
    print("created S3 and S4",flush=True)


if __name__ == "__main__": main()
