"""Recreate released figures directly from archived figure source data."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "derived"
FIG = ROOT / "figures"
sys.path.insert(0, str(HERE))
SCENARIOS = ("ssp126", "ssp245", "ssp585")
LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp126": "#0072B2", "ssp245": "#E69F00", "ssp585": "#CC79A7"}
PERIODS = ("mid", "late")


def smooth11(values):
    values = np.asarray(values, float)
    return np.convolve(np.pad(values, (5, 5), mode="edge"), np.ones(11) / 11, mode="valid")


def figure2() -> None:
    source = DATA / "Figure2_source_data.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    required = {
        "year", "scenario", "ozone_change_pct", "uv_change_pct",
        "chs_anomaly", "haf_with_uv", "haf_without_uv",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid Figure 2 source table: {source}")
    series = {}
    for scenario in SCENARIOS:
        selected = sorted((r for r in rows if r["scenario"] == scenario), key=lambda r: int(r["year"]))
        if [int(r["year"]) for r in selected] != list(range(2015, 2101)):
            raise RuntimeError(f"Incomplete Figure 2 series: {scenario}")
        series[scenario] = {
            key: np.asarray([float(r[key]) for r in selected])
            for key in required - {"year", "scenario"}
        }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 3,
        "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": False,
    })
    years = np.arange(2015, 2101)
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
    for scenario in SCENARIOS:
        values = series[scenario]
        axa.plot(years, smooth11(values["ozone_change_pct"]), color=COLORS[scenario], lw=1.35)
        axb.plot(years, smooth11(values["uv_change_pct"]), color=COLORS[scenario], lw=1.35)
        axc.plot(years, smooth11(values["chs_anomaly"]), color=COLORS[scenario], lw=1.35)
        axd.plot(years, smooth11(values["haf_with_uv"]), color=COLORS[scenario], lw=1.35)
        axd.plot(years, smooth11(values["haf_without_uv"]), color=COLORS[scenario], lw=0.9,
                 ls=(0, (3, 2)), alpha=0.9)
    for ax in (axa, axb, axc):
        ax.axhline(0, color="#777777", lw=0.55)
    axa.set_ylabel("Estimated total-column ozone change (%)")
    axb.set_ylabel("Erythemal UV dose change (%)")
    axc.set_ylabel("Composite Hazard Score anomaly")
    axd.set_ylabel("Habitable Area Fraction")
    for ax in axs.flat:
        ax.set_xlabel("Year", fontweight="normal")
    handles = [Line2D([0], [0], color=COLORS[s], lw=1.35, label=LABELS[s]) for s in SCENARIOS]
    handles.append(Patch(facecolor="#E6E6E6", edgecolor="none", label="2015-2025 baseline"))
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.53, 0.985), handlelength=2.2, columnspacing=1.5)
    axd.legend(handles=[
        Line2D([0], [0], color="#333333", lw=1.35, label="With UV term"),
        Line2D([0], [0], color="#333333", lw=0.9, ls=(0, (3, 2)), label="Without UV term"),
    ], frameon=False, loc="lower left", bbox_to_anchor=(0.14, 0.02), borderaxespad=0)
    for label, ax in zip("abcd", axs.flat):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold",
                fontsize=8, va="bottom")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "Figure2_ozone_UV_CHS_HAF_2015_2100.png", dpi=600, facecolor="white")
    fig.savefig(FIG / "Figure2_ozone_UV_CHS_HAF_2015_2100.pdf", facecolor="white")
    plt.close(fig)


def map_context(lat, lon, sample):
    with np.load(HERE / "haf_core" / "released" / "land_mask_0p5deg.npz") as z:
        full_land = z["mask"].astype(bool)
    lat2d = np.asarray(lat)[:, None] * np.ones((1, len(lon)))
    area = np.cos(np.deg2rad(lat2d))
    land = full_land & (lat2d > -60)
    display_land = full_land & np.isfinite(sample)
    return {
        "G": {"lat": np.asarray(lat), "lon": np.asarray(lon)},
        "land": land, "display_land": display_land, "area": area,
    }


def figure3() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import cartopy
    import cartopy.crs as ccrs
    source = DATA / "Figure3_source_fields_0p5deg.npz"
    metadata = json.loads((DATA / "Figure3_metadata.json").read_text(encoding="utf-8"))
    with np.load(source) as z:
        lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
        maps = {(s,p): np.asarray(z[f"{s}_{p}"]) for p in PERIODS for s in SCENARIOS}
    lookup={(item["scenario"],item["period"]):item for item in metadata["diagnostics"]}
    cartopy.config["data_dir"] = str(HERE / "cartopy_data")
    matplotlib.rcParams.update({"font.family":"Arial","font.size":6.5,"axes.titlesize":7,
        "axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,"pdf.fonttype":42,
        "ps.fonttype":42,"savefig.transparent":False})
    hazard=LinearSegmentedColormap.from_list("hazard",["#f7f4ec","#f6e2b3","#f3c169","#ea9a3c","#d96b2c","#b8342a","#8c1d35","#4d0f27"])
    pooled=np.concatenate([v[np.isfinite(v)] for v in maps.values()]);vmin=float(np.floor(np.percentile(pooled,.5)));vmax=float(np.ceil(np.percentile(pooled,99.5)));norm=Normalize(vmin=vmin,vmax=vmax)
    fig,axs=plt.subplots(2,3,figsize=(7.205,3.82),subplot_kw={"projection":ccrs.Robinson()},constrained_layout=False)
    fig.subplots_adjust(left=.048,right=.915,top=.965,bottom=.035,wspace=.035,hspace=.13);tr=ccrs.PlateCarree();im=None
    for r,p in enumerate(PERIODS):
        for c,s in enumerate(SCENARIOS):
            ax=axs[r,c];field=maps[(s,p)];ax.set_global();im=ax.pcolormesh(lon,lat,field,transform=tr,cmap=hazard,norm=norm,shading="auto",rasterized=True)
            ax.coastlines(resolution="110m",linewidth=.28,color="#222222");ax.spines["geo"].set_linewidth(.45)
            haf=lookup[(s,p)]["haf"];ax.set_title(f"{LABELS[s]}\nHAF = {haf:.3f}",pad=2.5,fontweight="bold",linespacing=1.15)
            ax.text(.01,.96,"abcdef"[r*3+c],transform=ax.transAxes,ha="left",va="top",fontweight="bold",fontsize=8)
    fig.text(.019,.735,"Mid-century (2041–2060)",rotation=90,ha="center",va="center",fontsize=6.4,fontweight="bold")
    fig.text(.019,.275,"Late-century (2081–2100)",rotation=90,ha="center",va="center",fontsize=6.4,fontweight="bold")
    cax=fig.add_axes([.935,.18,.014,.65]);cb=fig.colorbar(im,cax=cax);ticks=np.arange(np.ceil(vmin/2)*2,vmax+.1,2);cb.set_ticks(ticks);cb.set_label("Composite Hazard Score (CHS)",fontsize=7,labelpad=5);cb.ax.tick_params(labelsize=6,width=.5,length=2.5);cb.outline.set_linewidth(.45)
    for ext in ("png","pdf"):fig.savefig(FIG/f"Figure3_global_CHS_maps_mid_late.{ext}",dpi=600 if ext=="png" else None,facecolor="white")
    plt.close(fig)


def figure4() -> None:
    """Recreate main Figure 4 from its released map fields."""
    import matplotlib
    matplotlib.use("Agg")
    import plot_main_figures_4_5 as plotter
    with np.load(DATA / "Figure4_source_fields_0p5deg.npz") as z:
        fields = {(s, p): np.asarray(z[f"delta_total_{s}_{p}"]) for s in SCENARIOS for p in PERIODS}
        pure = {(s, p): np.asarray(z[f"pure_uv_{s}_{p}"]) for s in SCENARIOS for p in PERIODS}
        context = map_context(z["lat"], z["lon"], fields[("ssp126", "mid")])
    plotter.figure3(context, fields, pure)


def figure5() -> None:
    """Recreate main Figure 5 from released emergence and shift fields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch
    import analysis_helpers as base
    with np.load(DATA / "Figure5_source_fields_0p5deg.npz") as z:
        em7 = {s: np.asarray(z[f"emergence7_{s}"]) for s in SCENARIOS}
        shifts = {s: np.asarray(z[f"shift_{s}"]) for s in SCENARIOS}
        haf7 = {s: float(z[f"haf7_{s}"]) for s in SCENARIOS}
        dhaf = {s: float(z[f"delta_haf_{s}"]) for s in SCENARIOS}
        D = map_context(z["lat"], z["lon"], em7["ssp126"])
    matplotlib.rcParams.update({"font.family": "Arial", "font.size": 6.5,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42})
    em_cmap = plt.get_cmap("cividis").copy(); em_cmap.set_under("#3d3d3d"); em_cmap.set_over("#d9d9d9")
    adv_cmap = LinearSegmentedColormap.from_list("advance", ["#2166ac", "#d8e8f2", "#f7f7f7", "#f6c4ad", "#b2182b"])
    fig, axes = plt.subplots(2, 3, figsize=(7.205, 3.82))
    for c, s in enumerate(SCENARIOS):
        im1 = base.draw_robinson(axes[0, c], D["G"]["lon"], D["G"]["lat"],
            np.where(D["display_land"], em7[s], np.nan), D["display_land"], cmap=em_cmap, norm=Normalize(2020, 2100))
        axes[0, c].set_title(f"{LABELS[s]}\nHAF$_7$(2100) = {haf7[s]:.3f}", pad=2.5, weight="bold", linespacing=1.1)
        im2 = base.draw_robinson(axes[1, c], D["G"]["lon"], D["G"]["lat"], shifts[s],
            D["display_land"], cmap=adv_cmap, norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30))
        dh_label = f"{dhaf[s]:+.3f}".replace("-", "−")
        axes[1, c].set_title(f"{LABELS[s]}\n$\\Delta$HAF(2100) = {dh_label}", pad=2.5, weight="bold", linespacing=1.1)
        for r in range(2):
            axes[r, c].text(-.018, 1.035, chr(97+r*3+c), transform=axes[r,c].transAxes, weight="bold", fontsize=8, ha="left", va="bottom", clip_on=False)
    fig.text(.019,.735,"First persistent threshold crossing",rotation=90,ha="center",va="center",fontsize=6.3,weight="bold")
    fig.text(.019,.275,"UV extension effect on crossing year",rotation=90,ha="center",va="center",fontsize=6.3,weight="bold")
    fig.subplots_adjust(left=.058,right=.915,top=.945,bottom=.035,wspace=.040,hspace=.18)
    cax1=fig.add_axes([.935,.555,.014,.31]); cb1=fig.colorbar(im1,cax=cax1,extend="both"); cb1.set_ticks([2020,2040,2060,2080,2100]); cb1.set_label("Crossing year",fontsize=6.7,labelpad=3); cb1.ax.yaxis.set_label_position("left")
    fig.legend(handles=[Patch(facecolor="#3d3d3d",label="Above threshold in baseline"),Patch(facecolor="#d9d9d9",label="No crossing by 2100")],loc="upper right",bbox_to_anchor=(.995,.965),frameon=False,fontsize=5.6,handlelength=1.1,borderaxespad=0)
    cax2=fig.add_axes([.935,.135,.014,.31]); cb2=fig.colorbar(im2,cax=cax2,extend="both"); ticks=[-30,-15,0,15,30];cb2.set_ticks(ticks);cb2.ax.set_yticklabels([str(v).replace("-","−") for v in ticks]);cb2.set_label("Shift in crossing year",fontsize=6.7,labelpad=3);cb2.ax.yaxis.set_label_position("left")
    for ext in ("png","pdf"): fig.savefig(FIG/f"Figure5_HAF_and_first_crossing.{ext}",dpi=600 if ext=="png" else None)
    plt.close(fig)


def supplementary_s3_s4() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import plot_supplementary_figures_s3_s4 as plotter
    with np.load(DATA / "Supplementary_Figure_S3_source_fields_0p5deg.npz") as z:
        D = map_context(z["lat"], z["lon"], z["chs6_baseline"])
        chs6=np.asarray(z["chs6_baseline"]); chs7=np.asarray(z["chs7_baseline"]); delta=np.asarray(z["delta_chs_baseline"])
        pure={(s,p):np.asarray(z[f"pure_uv_{s}_{p}"]) for s in SCENARIOS for p in PERIODS}
    plotter.plot_s3(D, chs6, chs7, delta, pure)
    with np.load(DATA / "Supplementary_Figure_S4_source_fields_0p5deg.npz") as z:
        maps={(p,f):np.asarray(z[f"map_{p}_{f}"]) for p in PERIODS for f in plotter.FACTORS}
        years=np.asarray(z["time_years"])
        series={(s,f):np.asarray(z[f"series_{s}_{f}"]) for s in SCENARIOS for f in plotter.FACTORS}
    plotter.plot_s4(D, maps, years, series)


def supplementary_s1_s2() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_supplementary_figures_s1_s2 as original
    original.style()
    vars_ = original.VARS; labels_=original.VLABELS; methods=original.METHODS; mlabels=original.MLABELS
    doc=json.loads((DATA/"Supplementary_Figure_S1_source_data.json").read_text(encoding="utf-8"))
    w6=doc["six_variable_critic"];w7=doc["seven_variable_weights"];hmethods=doc["haf2100_ssp245_by_method"]
    years=np.asarray(doc["years"]);haf6={s:np.asarray(doc["haf6"][s]) for s in SCENARIOS};haf7={s:np.asarray(doc["haf7"][s]) for s in SCENARIOS}
    fig,axs=plt.subplots(2,2,figsize=(7.205,5.0));fig.subplots_adjust(left=.085,right=.985,top=.965,bottom=.09,wspace=.27,hspace=.34);axa,axb,axc,axd=axs.flat
    for ax in axs.flat:ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.tick_params(direction="out",top=False,right=False)
    x=np.arange(7);bw=.36;axa.bar(x-bw/2,[w6.get(k,0) for k in vars_],bw,color="#777777",label="CHS$_6$");axa.bar(x+bw/2,[w7["critic"][k] for k in vars_],bw,color="#0072B2",label="CHS$_7$");axa.set_xticks(x,labels_);axa.set_ylabel("Weight");axa.set_ylim(0,.43);axa.legend(frameon=False,loc="upper left");axa.set_title("CRITIC weights from 2015–2100",loc="left",weight="bold")
    pal=["#777777","#0072B2","#E69F00","#009E73"];bw2=.18
    for j,m in enumerate(methods):axb.bar(x+(j-1.5)*bw2,[w7[m][k] for k in vars_],bw2,color=pal[j],label=mlabels[m])
    axb.axhline(1/7,color=".5",lw=.6,ls=(0,(3,2)));axb.set_xticks(x,labels_);axb.set_ylabel("Weight");axb.set_ylim(0,.43);axb.legend(frameon=False,ncol=2,loc="upper left");axb.set_title("Objective weighting schemes for CHS$_7$",loc="left",weight="bold")
    vals=[hmethods[m] for m in methods];axc.bar(np.arange(4),vals,color=pal,width=.62)
    for i,v in enumerate(vals):axc.text(i,v+.012,f"{v:.3f}",ha="center",fontsize=6)
    axc.set_xticks(np.arange(4),[mlabels[m] for m in methods]);axc.set_ylabel("HAF in 2100 (SSP2-4.5)");axc.set_ylim(0,1);axc.set_title("HAF robustness to weighting method",loc="left",weight="bold")
    for s in SCENARIOS:axd.plot(years,smooth11(haf7[s]),color=COLORS[s],lw=1.35);axd.plot(years,smooth11(haf6[s]),color=COLORS[s],lw=.9,ls=(0,(3,2)))
    offsets={"ssp126":8,"ssp245":-8,"ssp585":0}
    for s in SCENARIOS:axd.annotate(LABELS[s],xy=(2100,smooth11(haf7[s])[-1]),xytext=(5,offsets[s]),textcoords="offset points",color=COLORS[s],fontsize=5.8,ha="left",va="center",clip_on=False)
    axd.set_xlim(2015,2114);axd.set_xticks([2020,2040,2060,2080,2100]);axd.set_xlabel("Year");axd.set_ylabel("Habitable Area Fraction");axd.set_title("HAF trajectories for CHS$_6$ and CHS$_7$",loc="left",weight="bold");axd.add_artist(axd.legend(handles=[Line2D([0],[0],color=".25",lw=1.35,label="CHS$_7$"),Line2D([0],[0],color=".25",lw=.9,ls=(0,(3,2)),label="CHS$_6$")],frameon=False,loc="lower left"))
    for letter,ax in zip("abcd",axs.flat):ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=8,weight="bold",va="bottom")
    for ext in ("png","pdf"):fig.savefig(FIG/f"Supplementary_Figure_S1_weighting_and_CHS_robustness.{ext}",dpi=600 if ext=="png" else None,facecolor="white")
    plt.close(fig)

    rows=list(csv.DictReader((DATA/"Figure2_source_data.csv").open(encoding="utf-8-sig")));traj={s:[] for s in SCENARIOS}
    for r in rows:traj[r["scenario"]].append((int(r["year"]),float(r["ozone_change_pct"]),float(r["uv_change_pct"])))
    sens=list(csv.DictReader((DATA/"Supplementary_Figure_S2_source_data.csv").open(encoding="utf-8-sig")))
    fig,axs=plt.subplots(2,2,figsize=(7.205,5.15));fig.subplots_adjust(left=.09,right=.985,top=.955,bottom=.09,wspace=.48,hspace=.52);axa,axb,axc,axd=axs.flat
    for ax in axs.flat:ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.tick_params(direction="out",top=False,right=False)
    for s in SCENARIOS:
        a=np.asarray(traj[s]);xx=smooth11(a[:,1]);yy=smooth11(a[:,2]);axa.plot(xx,yy,color=COLORS[s],lw=1.25)
        for year in (2050,2100):j=np.where(a[:,0]==year)[0][0];axa.scatter(xx[j],yy[j],s=15,color=COLORS[s],edgecolor="white",linewidth=.35,zorder=3)
        axa.annotate(LABELS[s],xy=(xx[-1],yy[-1]),xytext=(5,0),textcoords="offset points",color=COLORS[s],fontsize=5.8,ha="left",va="center",clip_on=False)
    axa.axhline(0,color=".65",lw=.5);axa.axvline(0,color=".65",lw=.5);xmin,xmax=axa.get_xlim();axa.set_xlim(xmin,xmax+(xmax-xmin)*.12);axa.set_xlabel("Estimated total-column ozone change (%)");axa.set_ylabel("Erythemal UV dose change (%)");axa.set_title("Ozone and UV response trajectories",loc="left",weight="bold")
    abl=[r for r in sens if r["panel"]=="fixed_chs7_uv_ablation"]
    for i,r in enumerate(abl):c=COLORS[r["scenario"]];h0=float(r["haf_without_direct_uv"]);h1=float(r["haf_full_chs7"]);axb.plot([h0,h1],[i,i],color=".65",lw=.8);axb.scatter(h0,i,s=20,facecolor="white",edgecolor=c,linewidth=.8,zorder=2);axb.scatter(h1,i,s=20,facecolor=c,edgecolor=c,linewidth=.6,zorder=3)
    axb.set_yticks(np.arange(6),[f"{LABELS[r['scenario']]}  {'2041–2060' if r['period']=='mid' else '2081–2100'}" for r in abl]);axb.invert_yaxis();axb.set_xlim(0,1);axb.set_xlabel("Habitable Area Fraction");axb.set_title("Controlled ablation of the direct UV term",loc="left",weight="bold");axb.legend(handles=[Line2D([0],[0],marker="o",lw=0,mfc="white",mec=".25",label="UV term removed"),Line2D([0],[0],marker="o",lw=0,mfc=".25",mec=".25",label="Full CHS$_7$")],frameon=False,loc="lower left",bbox_to_anchor=(.015,.19),borderaxespad=0,handletextpad=.55,labelspacing=.35)
    for panel,ax in (("weight",axc),("threshold",axd)):
        for s in SCENARIOS:
            rr=[r for r in sens if r["panel"]==panel and r["scenario"]==s];ax.plot([float(r["x"]) for r in rr],[float(r["delta_haf"]) for r in rr],color=COLORS[s],lw=1.2,marker="o",ms=2.8,label=LABELS[s])
    wuv=max(float(r["x"]) for r in sens if r["panel"]=="weight");axc.axhline(0,color=".5",lw=.55);axc.axvline(wuv,color=".35",lw=.65,ls=(0,(3,2)));axc.text(wuv,-.003,"CRITIC",rotation=90,ha="right",va="top",fontsize=5.5,color=".35");axc.set_xlabel("UV weight");axc.set_ylabel("Change in HAF");axc.set_title("Sensitivity to UV weight, 2081–2100",loc="left",weight="bold");axc.legend(frameon=False,loc="lower left")
    axd.axhline(0,color=".5",lw=.55);axd.axvline(90,color=".35",lw=.65,ls=(0,(3,2)));axd.set_xlabel("Baseline threshold percentile");axd.set_ylabel("Change in HAF");axd.set_title("Threshold sensitivity, 2081–2100",loc="left",weight="bold");axd.legend(frameon=False,loc="lower left")
    for letter,ax in zip("abcd",axs.flat):ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=8,weight="bold",va="bottom")
    for ext in ("png","pdf"):fig.savefig(FIG/f"Supplementary_Figure_S2_UV_effect_and_sensitivity.{ext}",dpi=600 if ext=="png" else None,facecolor="white")
    plt.close(fig)

