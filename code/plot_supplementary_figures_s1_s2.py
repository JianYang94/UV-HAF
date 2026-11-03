"""Create Supplementary Figures S1 and S2 for the UV-extended HAF study."""
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

import analysis_helpers as base
import plot_main_figures_4_5 as corrected
from haf_core import chs, emulator

YEARS = base.YEARS
SCENARIOS = base.SCENARIOS
LABELS = base.LABELS
COLORS = base.COLORS
VARS = ["gmst", "co2", "sst", "ohc", "ph", "omega", "uv_erythema"]
VLABELS = ["GMST", "CO$_2$", "SST", "OHC", "pH", "$\\Omega_{arag}$", "UV"]
METHODS = ("equal", "critic", "entropy", "pca")
MLABELS = {"equal": "Equal", "critic": "CRITIC", "entropy": "Entropy", "pca": "PCA"}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 5.8, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.transparent": False,
    })


def objective_from_matrix(X, method):
    X = np.asarray(X, float)
    lo = X.min(0); rng = np.ptp(X, axis=0)
    Z = (X - lo) / np.where(rng > 0, rng, 1.0)
    n = Z.shape[1]
    if method == "equal":
        w = np.full(n, 1 / n)
    elif method == "critic":
        sd = Z.std(0, ddof=1)
        R = np.nan_to_num(np.corrcoef(Z, rowvar=False))
        c = sd * (1 - R).sum(0)
        w = c / c.sum()
    elif method == "entropy":
        P = Z / np.where(Z.sum(0) > 0, Z.sum(0), 1.0)
        e = -(P * np.where(P > 0, np.log(P), 0)).sum(0) / np.log(Z.shape[0])
        d = 1 - e; w = d / d.sum()
    elif method == "pca":
        C = np.cov(Z - Z.mean(0), rowvar=False)
        vals, vecs = np.linalg.eigh(C)
        load = np.abs(vecs[:, np.argmax(vals)])
        w = load / load.sum()
    else:
        raise ValueError(method)
    return dict(zip(VARS, map(float, w)))


def training_and_weights():
    ally = np.arange(1750, 2301)
    params = {"ecs": 3.65, "gamma": 0.48}
    ref = emulator.run_emulator(params, ally, ssp="ssp245", chem=False)
    scales = chs.reference_scales_window(ref)
    idx = np.array([int(np.argmin(np.abs(ally - y))) for y in YEARS])
    X6 = chs._standardised_matrix(ref, scales)[idx]
    # Apply the same centred 11-year smoothing to every climate-ocean input
    # before estimating any objective weights. This reproduces the official
    # CRITIC training matrix used by the main figures.
    X6 = np.column_stack([base.smooth11(X6[:, j]) for j in range(X6.shape[1])])
    weights_path = OUT / "critic_weights_2015_2100.json"
    if not weights_path.exists():
        weights_path = DATA / "critic_weights_2015_2100.json"
    wd = json.loads(weights_path.read_text(encoding="utf-8"))
    uv = np.array([wd["training"]["annual_uv_11yr_smooth"][str(y)] for y in YEARS])
    X7 = np.column_stack([X6, uv])
    w7 = {m: objective_from_matrix(X7, m) for m in METHODS}
    # Same-window six-variable comparator, defined independently for this study.
    w6 = wd["critic6_11yr_smoothed_all_variables"]["weights"]
    return ally, params, scales, idx, w6, w7


def climate_for_weights(ally, params, scales, idx, weights7):
    six_total = sum(weights7[k] for k in chs.CHS_VARS)
    rel = {k: weights7[k] / six_total for k in chs.CHS_VARS}
    out = {}
    for s in SCENARIOS:
        run = emulator.run_emulator(params, ally, ssp=s, chem=False)
        st, uu = chs.tier_series(run, rel, scales)
        out[s] = (six_total * st[idx], six_total * uu[idx])
    return out


def threshold(D, climate):
    acc = np.zeros_like(D["G"]["B2d"], float); n = 0
    for s in SCENARIOS:
        st, uu = climate[s]
        for year in range(2015, 2026):
            j = year - 2015
            acc += D["G"]["B2d"] + D["G"]["P2d"] * st[j] + uu[j]
            n += 1
    acc /= n
    return base.weighted_percentile(acc[D["land"]], D["area"][D["land"]], 90)


def haf_trajectory(D, climate, wuv, tau):
    result = {s: np.empty(len(YEARS)) for s in SCENARIOS}
    for s in SCENARIOS:
        st, uu = climate[s]
        for j, year in enumerate(YEARS):
            z = (base.load_uv(s, int(year), D["G"]) - D["uv_ref"]) / D["uv_scale"]
            field = D["G"]["B2d"] + D["G"]["P2d"] * st[j] + uu[j] + wuv * z
            valid = D["land"] & np.isfinite(field)
            result[s][j] = D["area"][valid & (field < tau)].sum() / D["area"][valid].sum()
    return result


def haf2100_by_method(D, ally, params, scales, idx, w7):
    vals = {}
    for method in METHODS:
        climate = climate_for_weights(ally, params, scales, idx, w7[method])
        tau = threshold(D, climate)
        st, uu = climate["ssp245"]
        z = (base.load_uv("ssp245", 2100, D["G"]) - D["uv_ref"]) / D["uv_scale"]
        field = (D["G"]["B2d"] + D["G"]["P2d"] * st[-1] + uu[-1]
                 + w7[method]["uv_erythema"] * z)
        valid = D["land"] & np.isfinite(field)
        vals[method] = float(D["area"][valid & (field < tau)].sum() / D["area"][valid].sum())
    return vals


def fixed_chs7_uv_ablation(D, scenario, period_name, period):
    """Remove only C_UV while holding the complete CHS7 definition fixed."""
    q = period[(scenario, period_name)]
    no_direct_uv = (D["G"]["B2d"]
                    + D["G"]["P2d"] * (D["six_total"] * q["st"])
                    + D["six_total"] * q["u"])
    full_chs7 = no_direct_uv + D["wuv"] * q["z"]
    valid = D["land"] & np.isfinite(no_direct_uv) & np.isfinite(full_chs7)
    den = D["area"][valid].sum()
    # Crucially, both fractions use the same fixed CHS7 threshold.
    h_no_uv = float(D["area"][valid & (no_direct_uv < D["tau"])].sum() / den)
    h_full = float(D["area"][valid & (full_chs7 < D["tau"])].sum() / den)
    return h_no_uv, h_full


def figure_s1(D):
    import matplotlib.pyplot as plt
    style()
    ally, params, scales, idx, w6, w7 = training_and_weights()
    S6 = corrected.six_variable_setup(D)
    haf6 = haf_trajectory(D, S6["climate"], 0.0, S6["tau"])
    clim7 = climate_for_weights(ally, params, scales, idx, w7["critic"])
    tau7 = threshold(D, clim7)
    haf7 = haf_trajectory(D, clim7, w7["critic"]["uv_erythema"], tau7)
    hmethods = haf2100_by_method(D, ally, params, scales, idx, w7)

    fig, axs = plt.subplots(2, 2, figsize=(7.205, 5.0))
    fig.subplots_adjust(left=.085, right=.985, top=.965, bottom=.09,
                        wspace=.27, hspace=.34)
    axa, axb, axc, axd = axs.flat
    for ax in axs.flat:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)

    x = np.arange(7); bw = .36
    six = [w6.get(k, 0.0) for k in VARS]
    seven = [w7["critic"][k] for k in VARS]
    axa.bar(x-bw/2, six, bw, color="#777777", label="CHS$_6$")
    axa.bar(x+bw/2, seven, bw, color="#0072B2", label="CHS$_7$")
    axa.set_xticks(x, VLABELS); axa.set_ylabel("Weight"); axa.set_ylim(0, .43)
    axa.legend(frameon=False, loc="upper left")
    axa.set_title("CRITIC weights from 2015–2100", loc="left", weight="bold")

    pal = ["#777777", "#0072B2", "#E69F00", "#009E73"]
    bw2 = .18
    for j, m in enumerate(METHODS):
        axb.bar(x + (j-1.5)*bw2, [w7[m][k] for k in VARS], bw2,
                color=pal[j], label=MLABELS[m])
    axb.axhline(1/7, color=".5", lw=.6, ls=(0,(3,2)))
    axb.set_xticks(x, VLABELS); axb.set_ylabel("Weight"); axb.set_ylim(0, .43)
    axb.legend(frameon=False, ncol=2, loc="upper left")
    axb.set_title("Objective weighting schemes for CHS$_7$", loc="left", weight="bold")

    vals = [hmethods[m] for m in METHODS]
    axc.bar(np.arange(4), vals, color=pal, width=.62)
    for i, v in enumerate(vals): axc.text(i, v+.012, f"{v:.3f}", ha="center", fontsize=6)
    axc.set_xticks(np.arange(4), [MLABELS[m] for m in METHODS])
    axc.set_ylabel("HAF in 2100 (SSP2-4.5)"); axc.set_ylim(0, 1)
    axc.set_title("HAF robustness to weighting method", loc="left", weight="bold")

    for s in SCENARIOS:
        axd.plot(YEARS, base.smooth11(haf7[s]), color=COLORS[s], lw=1.35, label=LABELS[s])
        axd.plot(YEARS, base.smooth11(haf6[s]), color=COLORS[s], lw=.9, ls=(0,(3,2)))
    label_offset={"ssp126":8,"ssp245":-8,"ssp585":0}
    for s in SCENARIOS:
        yend=base.smooth11(haf7[s])[-1]
        axd.annotate(LABELS[s],xy=(2100,yend),xytext=(5,label_offset[s]),
                     textcoords="offset points",color=COLORS[s],fontsize=5.8,
                     ha="left",va="center",clip_on=False)
    axd.set_xlim(2015,2114); axd.set_xticks([2020,2040,2060,2080,2100])
    axd.set_xlabel("Year"); axd.set_ylabel("Habitable Area Fraction")
    axd.set_title("HAF trajectories for CHS$_6$ and CHS$_7$", loc="left", weight="bold")
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=".25",lw=1.35,label="CHS$_7$"),
             Line2D([0],[0],color=".25",lw=.9,ls=(0,(3,2)),label="CHS$_6$")]
    style_legend=axd.legend(handles=handles, frameon=False, loc="lower left")
    axd.add_artist(style_legend)
    for letter, ax in zip("abcd", axs.flat):
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=8,weight="bold",va="bottom")

    for ext in ("png","pdf"):
        fig.savefig(FIG/f"Supplementary_Figure_S1_weighting_and_CHS_robustness.{ext}",
                    dpi=600 if ext=="png" else None, facecolor="white")
    plt.close(fig)
    data={"six_variable_critic":w6,"seven_variable_weights":w7,
          "haf2100_ssp245_by_method":hmethods,"tau6":S6["tau"],"tau7":tau7,
          "years":list(map(int,YEARS)),
          "haf6":{s:list(map(float,haf6[s])) for s in SCENARIOS},
          "haf7":{s:list(map(float,haf7[s])) for s in SCENARIOS}}
    (FIG/"Supplementary_Figure_S1_source_data.json").write_text(json.dumps(data,indent=2),encoding="utf-8")


def figure_s2(D):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style()
    _, _, period, _ = base.analyse(D)
    source = FIG / "Figure2_source_data.csv"
    if not source.exists():
        source = DATA / "Figure2_source_data.csv"
    rows=list(csv.DictReader(source.open(encoding="utf-8-sig")))
    traj={s:[] for s in SCENARIOS}
    for r in rows:
        traj[r["scenario"]].append((int(r["year"]),float(r["ozone_change_pct"]),float(r["uv_change_pct"])))
    fig,axs=plt.subplots(2,2,figsize=(7.205,5.15))
    # Extra separation prevents panel-b tick labels from being crossed by
    # the zero line in the lower-left sensitivity panel.
    fig.subplots_adjust(left=.09,right=.985,top=.955,bottom=.09,wspace=.48,hspace=.52)
    axa,axb,axc,axd=axs.flat
    for ax in axs.flat:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out",top=False,right=False)
    for s in SCENARIOS:
        a=np.asarray(traj[s]); xx=base.smooth11(a[:,1]); yy=base.smooth11(a[:,2])
        axa.plot(xx,yy,color=COLORS[s],lw=1.25)
        for year in (2050,2100):
            j=np.where(a[:,0]==year)[0][0]
            axa.scatter(xx[j],yy[j],s=15,color=COLORS[s],edgecolor="white",linewidth=.35,zorder=3)
        axa.annotate(LABELS[s],xy=(xx[-1],yy[-1]),xytext=(5,0),
                     textcoords="offset points",color=COLORS[s],fontsize=5.8,
                     ha="left",va="center",clip_on=False)
    axa.axhline(0,color=".65",lw=.5); axa.axvline(0,color=".65",lw=.5)
    xmin,xmax=axa.get_xlim(); axa.set_xlim(xmin,xmax+(xmax-xmin)*.12)
    axa.set_xlabel("Estimated total-column ozone change (%)"); axa.set_ylabel("Erythemal UV dose change (%)")
    axa.set_title("Ozone and UV response trajectories",loc="left",weight="bold")

    labels=[]; h0s=[]; h1s=[]; cols=[]
    for p in ("mid","late"):
        for s in SCENARIOS:
            h0,h1=fixed_chs7_uv_ablation(D,s,p,period)
            h0s.append(h0);h1s.append(h1);cols.append(COLORS[s])
            labels.append(f"{LABELS[s]}  {'2041–2060' if p=='mid' else '2081–2100'}")
    for i,(h0,h1,c) in enumerate(zip(h0s,h1s,cols)):
        axb.plot([h0,h1],[i,i],color=".65",lw=.8)
        axb.scatter(h0,i,s=20,facecolor="white",edgecolor=c,linewidth=.8,zorder=2)
        axb.scatter(h1,i,s=20,facecolor=c,edgecolor=c,linewidth=.6,zorder=3)
    axb.set_yticks(np.arange(6),labels);axb.invert_yaxis();axb.set_xlim(0,1)
    axb.set_xlabel("Habitable Area Fraction");axb.set_title("Controlled ablation of the direct UV term",loc="left",weight="bold")
    axb.legend(handles=[Line2D([0],[0],marker="o",lw=0,mfc="white",mec=".25",label="UV term removed"),
                        Line2D([0],[0],marker="o",lw=0,mfc=".25",mec=".25",label="Full CHS$_7$")],
               frameon=False,loc="lower left",bbox_to_anchor=(.015,.19),
               borderaxespad=0,handletextpad=.55,labelspacing=.35)
    sens=[]; weights=np.array([0,.05,.10,.15,.20,.25,D["wuv"]])
    for s in SCENARIOS:
        vals=[]
        for w in weights:
            h0,h1=base.haf_for(D,s,"late",float(w),90,period); vals.append(h1-h0)
            sens.append(dict(panel="weight",scenario=s,x=float(w),delta_haf=h1-h0))
        axc.plot(weights,vals,color=COLORS[s],lw=1.2,marker="o",ms=2.8,label=LABELS[s])
    axc.axhline(0,color=".5",lw=.55);axc.axvline(D["wuv"],color=".35",lw=.65,ls=(0,(3,2)))
    axc.text(D["wuv"],-.003,"CRITIC",rotation=90,ha="right",va="top",fontsize=5.5,color=".35")
    axc.set_xlabel("UV weight");axc.set_ylabel("Change in HAF");axc.set_title("Sensitivity to UV weight, 2081–2100",loc="left",weight="bold")
    axc.legend(frameon=False,loc="lower left")
    pcts=np.array([80,82.5,85,87.5,90,92.5,95,97.5])
    for s in SCENARIOS:
        vals=[]
        for pct in pcts:
            h0,h1=base.haf_for(D,s,"late",D["wuv"],float(pct),period);vals.append(h1-h0)
            sens.append(dict(panel="threshold",scenario=s,x=float(pct),delta_haf=h1-h0))
        axd.plot(pcts,vals,color=COLORS[s],lw=1.2,marker="o",ms=2.8,label=LABELS[s])
    axd.axhline(0,color=".5",lw=.55);axd.axvline(90,color=".35",lw=.65,ls=(0,(3,2)))
    axd.set_xlabel("Baseline threshold percentile");axd.set_ylabel("Change in HAF")
    axd.set_title("Threshold sensitivity, 2081–2100",loc="left",weight="bold")
    axd.legend(frameon=False,loc="lower left")
    for letter,ax in zip("abcd",axs.flat):
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=8,weight="bold",va="bottom")
    for ext in ("png","pdf"):
        fig.savefig(FIG/f"Supplementary_Figure_S2_UV_effect_and_sensitivity.{ext}",
                    dpi=600 if ext=="png" else None,facecolor="white")
    plt.close(fig)
    ablation=[]
    for p in ("mid","late"):
        for s in SCENARIOS:
            h0,h1=fixed_chs7_uv_ablation(D,s,p,period)
            ablation.append(dict(panel="fixed_chs7_uv_ablation",scenario=s,period=p,
                                 x="",haf_without_direct_uv=h0,haf_full_chs7=h1,
                                 delta_haf=h1-h0))
    with (FIG/"Supplementary_Figure_S2_source_data.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["panel","scenario","period","x","haf_without_direct_uv","haf_full_chs7","delta_haf"]
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for row in ablation:
            wr.writerow(row)
        for row in sens:
            wr.writerow(row)


def main():
    if not (OUT / "uv_ssp126_2015_2015.npz").exists():
        from source_data_reproduction import supplementary_s1_s2
        supplementary_s1_s2()
        print("created S1 and S2 from archived source data")
        return
    FIG.mkdir(parents=True,exist_ok=True)
    D=base.common_setup()
    figure_s1(D)
    figure_s2(D)
    print("created S1 and S2")


if __name__=="__main__":
    main()
