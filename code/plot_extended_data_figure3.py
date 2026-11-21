"""Regenerate Extended Data Figure 3 from archived source fields."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FixedLocator
import cartopy
import cartopy.crs as ccrs

FACTORS = ("ozone", "cloud", "aerosol")
PERIODS = ("mid", "late")
SCENARIOS = ("ssp126", "ssp245", "ssp585")
FACTOR_LABELS = {"ozone": "Ozone", "cloud": "Cloud", "aerosol": "Aerosol"}
SCENARIO_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
FACTOR_COLORS = {"ozone": "#7B3294", "cloud": "#008837", "aerosol": "#E66101"}

def lon_label(x):
    lon = ((x - 90 + 180) % 360) - 180
    if np.isclose(abs(lon), 180): return "180°"
    if np.isclose(lon, 0): return "0°"
    return f"{int(abs(lon))}°{'E' if lon > 0 else 'W'}"

def lat_label(y):
    if np.isclose(y, 0): return "0°"
    return f"{int(abs(y))}°{'N' if y > 0 else 'S'}"

def style_map(ax):
    ax.set_global()
    xs, ys = np.arange(-180,181,30), np.arange(-90,91,30)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(-180,181,10)))
    ax.set_xticklabels([lon_label(x) for x in xs]); ax.set_yticklabels([lat_label(y) for y in ys])
    ax.tick_params(which="major", direction="out", top=True, right=True,
                   labeltop=False, labelright=False, length=4, width=.85,
                   pad=1.8, labelsize=5.2)
    ax.tick_params(axis="x", which="minor", direction="out", top=True,
                   bottom=True, length=2.2, width=.65)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(.85)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--output-dir",type=Path,default=None)
    a=p.parse_args(); repo=a.repo_root.resolve(); out=(a.output_dir or repo/"figures").resolve(); out.mkdir(parents=True,exist_ok=True)
    source=repo/"data"/"derived"/"Extended_Data_Figure_3_source_fields_0p5deg.npz"
    mask_path=repo/"code"/"haf_core"/"released"/"land_mask_0p5deg.npz"
    cartopy.config["data_dir"]=str(repo/"code"/"cartopy_data")
    z=np.load(source); land=np.load(mask_path)["mask"].astype(bool)
    lat=np.linspace(-89.75,89.75,360); lon=np.linspace(-179.75,179.75,720)
    display=land.copy(); area=np.cos(np.deg2rad(lat))[:,None]*np.ones((1,lon.size))
    maps={(p0,f):np.asarray(z[f"map_{p0}_{f}"],float) for p0 in PERIODS for f in FACTORS}
    years=np.asarray(z["time_years"],float)
    series={(s,f):np.asarray(z[f"series_{s}_{f}"],float) for s in SCENARIOS for f in FACTORS}
    means={(p0,f):float(np.average(maps[(p0,f)][land],weights=area[land])) for p0 in PERIODS for f in FACTORS}

    matplotlib.rcParams.update({"font.family":"Arial","font.size":7,"axes.labelsize":8.2,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.7,
        "axes.linewidth":.65,"pdf.fonttype":42,"ps.fonttype":42})
    cmap=LinearSegmentedColormap.from_list("driver",["#2166ac","#67a9cf","#d1e5f0","#f7f7f7","#fddbc7","#ef8a62","#b2182b"],N=256)
    norm=TwoSlopeNorm(vmin=-2.4,vcenter=0,vmax=2.4)
    proj=ccrs.PlateCarree(central_longitude=-90); trans=ccrs.PlateCarree()
    fig=plt.figure(figsize=(10.8,5.80))
    gs=fig.add_gridspec(3,3,left=.060,right=.985,bottom=.125,top=.960,
                        wspace=.105,hspace=.30)
    axes=np.empty((3,2),object); m=None
    for r,f in enumerate(FACTORS):
        for c,p0 in enumerate(PERIODS):
            ax=fig.add_subplot(gs[r,c],projection=proj); axes[r,c]=ax
            fld=np.where(display,maps[(p0,f)],np.nan)
            m=ax.pcolormesh(lon,lat,fld,cmap=cmap,norm=norm,shading="auto",
                            transform=trans,rasterized=True)
            ax.coastlines(resolution="110m",linewidth=.48,color="black"); style_map(ax)
            letter=chr(ord('a')+r+3*c)
            ax.text(0,1.032,letter,transform=ax.transAxes,ha="left",va="bottom",fontsize=10.5,fontweight="bold",clip_on=False)
            ax.text(.5,1.032,f"{means[(p0,f)]:+.3f}".replace("-","−"),transform=ax.transAxes,ha="center",va="bottom",fontsize=8,clip_on=False)
    # Match the compact two-column map figures: retain the left column and shift the right one left.
    for r in range(3): axes[r,1].set_position(axes[r,1].get_position().translated(-.020,0))
    fig.canvas.draw()
    for c in range(2):
        pos=axes[2,c].get_position(); cax=fig.add_axes([pos.x0+.055*pos.width,pos.y0-.047,.89*pos.width,.012])
        cb=fig.colorbar(m,cax=cax,orientation="horizontal",extend="both",ticks=[-2.4,-1.2,0,1.2,2.4],drawedges=False)
        cb.ax.tick_params(direction="out",length=3,width=.75,pad=2); cb.outline.set_linewidth(.75); cb.dividers.set_linewidth(.3)

    allv=np.concatenate([series[k] for k in series]); pad=max(.03,.06*np.ptp(allv)); ylim=(min(0,float(allv.min()))-pad,float(allv.max())+pad)
    curve_axes=[]
    for i,s in enumerate(SCENARIOS):
        ax=fig.add_subplot(gs[i,2]); curve_axes.append(ax)
        for x0,x1,color in ((2015,2025,"#E4E4E4"),(2041,2060,"#EAF1F4"),(2081,2100,"#F6E9EC")):
            ax.axvspan(x0,x1,color=color,lw=0,zorder=0)
            ax.axvline(x0,color="#B0B0B0",lw=.45,ls=(0,(2,2)),zorder=1)
            ax.axvline(x1,color="#B0B0B0",lw=.45,ls=(0,(2,2)),zorder=1)
        for f in FACTORS: ax.plot(years,series[(s,f)],color=FACTOR_COLORS[f],lw=1.45,marker="o",ms=2.8,label=FACTOR_LABELS[f])
        ax.axhline(0,color="#777",lw=.55); ax.set_xlim(2015,2105); ax.set_ylim(*ylim); ax.set_xticks([2020,2040,2060,2080,2100])
        # The 2015--2025 reference band is too narrow for the scenario name;
        # centre it in the clear in-frame gap before the mid-century band.
        ax.text(2028,.66,SCENARIO_LABELS[s],transform=ax.get_xaxis_transform(),
                ha="center",va="center",fontsize=8,fontweight="bold",zorder=7)
        if i==2: ax.set_xlabel("Year")
        ax.tick_params(direction="out",top=True,right=True,length=3,width=.65)
        for sp in ax.spines.values(): sp.set_visible(True)
        ax.legend(frameon=True,fancybox=False,loc="upper left",ncol=3,
                  borderpad=.30,handlelength=1.7,columnspacing=.75,
                  handletextpad=.4,edgecolor="#333",framealpha=.96,
                  fontsize=6.7)
        ax.text(-.12,1.035,chr(ord('g')+i),transform=ax.transAxes,
                fontsize=10.5,fontweight="bold",va="bottom")
    curve_axes[1].set_ylabel("Mean contribution to CHS over land", fontsize=7.2, labelpad=5)
    note_kw=dict(ha="center",va="center",fontsize=6.2,linespacing=1.05,zorder=7)
    for ax in curve_axes:
        ax.text(2020,1.015,"Reference period",transform=ax.get_xaxis_transform(),
                ha="center",va="bottom",fontsize=6.3,fontweight="normal",
                zorder=7,clip_on=False)
        ax.text(2050.5,1.015,"Mid-century",transform=ax.get_xaxis_transform(),
                ha="center",va="bottom",fontsize=6.3,fontweight="normal",
                zorder=7,clip_on=False)
        ax.text(2090.5,1.015,"Late-century",transform=ax.get_xaxis_transform(),
                ha="center",va="bottom",fontsize=6.3,fontweight="normal",
                zorder=7,clip_on=False)
    curve_axes[0].text(2050,.61,"Aerosol\ncontribution\nrises",**note_kw)
    curve_axes[0].text(2090,.72,"Aerosol\nremains\nlargest",**note_kw)
    curve_axes[1].text(2050,.61,"Cloud\nbecomes\nleading",**note_kw)
    curve_axes[1].text(2090,.61,"Cloud\nremains\ndominant",**note_kw)
    # Fixed empty-zone placement is deliberate: data-relative placement made
    # the rendered green and purple lines pass through these two labels.
    curve_axes[2].text(2050.5,.61,"Cloud dominates",**note_kw)
    curve_axes[2].text(2090,.57,"Ozone turns\npositive near\n2100",**note_kw)

    stem="Extended_Data_Figure_3"
    fig.savefig(out/f"{stem}.png",dpi=600,facecolor="white",bbox_inches="tight",pad_inches=.035)
    fig.savefig(out/f"{stem}.pdf",facecolor="white",bbox_inches="tight",pad_inches=.035)
    plt.close(fig); print(out/f"{stem}.png"); print(out/f"{stem}.pdf")

if __name__=="__main__": main()
