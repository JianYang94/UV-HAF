"""Estimate CHS6 and CHS7 weights and calculate the composite diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = PROJECT_ROOT / "outputs"
sys.path.insert(0, str(HERE))

NAMES = ("gmst", "co2", "sst", "ohc", "ph", "omega", "uv_erythema")
YEARS = np.arange(2015, 2101)
SCENARIOS = ("ssp126", "ssp245", "ssp585")


def regrid(source, slat, slon, dlat, dlon):
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


def weighted_percentile(values, weights, q):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    return float(np.interp(q / 100.0, np.cumsum(w) / w.sum(), v))


def critic(X):
    lo = np.nanmin(X, axis=0)
    span = np.nanmax(X, axis=0) - lo
    span = np.where(span > 0, span, 1.0)
    Z = (X - lo) / span
    sd = np.std(Z, axis=0, ddof=1)
    R = np.corrcoef(Z, rowvar=False)
    R = np.where(np.isfinite(R), R, 0.0)
    conflict = np.sum(1.0 - R, axis=1)
    info = sd * conflict
    w = info / info.sum()
    return w, sd, R, conflict, info


def annual_land_uv_series(G):
    values = []
    cache_files = []
    for year in YEARS:
        cache = OUT / f"uv_ssp245_{year}_{year}.npz"
        with np.load(cache) as z:
            annual = np.asarray(z["annual"], float)
            lat, lon = np.asarray(z["lat"], float), np.asarray(z["lon"], float)
        uv05 = regrid(annual, lat, lon, G["lat"], G["lon"])
        land = G["land2d"] & np.isfinite(uv05)
        values.append(float(np.average(uv05[land], weights=G["area2d"][land])))
        cache_files.append(str(OUT / f"uv_ssp245_{year}_{year}.npz"))
    return np.asarray(values), cache_files


def load_period_fields():
    periods = {"reference": (2015, 2025), "mid": (2041, 2060), "late": (2081, 2100)}
    fields = {s: {} for s in SCENARIOS}
    lat = lon = None
    for s in SCENARIOS:
        for label, (y0, y1) in periods.items():
            p = OUT / f"uv_{s}_{y0}_{y1}.npz"
            z = np.load(p)
            fields[s][label] = np.asarray(z["annual"], float)
            lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    return fields, lat, lon


def main():
    from haf_core import emulator, chs, grid

    all_years = np.arange(1750, 2301)
    params = {"ecs": 3.65, "gamma": 0.48}
    refout = emulator.run_emulator(params, all_years, ssp="ssp245", chem=False)
    scales = chs.reference_scales_window(refout)

    G = grid.build()
    uv_series, annual_cache = annual_land_uv_series(G)
    idx = np.array([int(np.argmin(np.abs(all_years - y))) for y in YEARS])
    six = np.column_stack([
        chs._standardise(refout[k], all_years, chs.CHS_VARS[k], scales[k])[idx]
        for k in chs.CHS_VARS
    ])
    X = np.column_stack([six, uv_series])
    w, sd, R, conflict, info = critic(X)
    weights7 = {k: float(v) for k, v in zip(NAMES, w)}

    # Apply the same centred 11-year running mean to all seven variables so the
    # weighting reflects forced multi-decadal response rather than UKESM's
    # single-member year-to-year variability. Edge values are padded.
    kernel = np.ones(11) / 11.0
    X_smooth = np.column_stack([
        np.convolve(np.pad(X[:, i], (5, 5), mode="edge"), kernel, mode="valid")
        for i in range(X.shape[1])
    ])
    uv_smooth = X_smooth[:, -1]
    Xs = X_smooth
    ws, sds, Rs, conflicts, infos = critic(Xs)
    weights7_smooth = {k: float(v) for k, v in zip(NAMES, ws)}
    w6s, sd6s, R6s, conflict6s, info6s = critic(X_smooth[:, :6])
    weights6_smooth = {k: float(v) for k, v in zip(NAMES[:6], w6s)}

    weights_only = {
        "status": "final_weights",
        "training": {
            "scenario": "ssp245", "years": [2015, 2100],
            "standardisation_center": [1750, 1800],
            "standardisation_scale": [1750, 2100],
            "sample_unit": "annual values (n=86)",
            "uv_metric": "area-weighted global-land mean erythemal daily dose",
            "annual_uv_values": {str(y): float(v) for y, v in zip(YEARS, uv_series)},
            "annual_uv_11yr_smooth": {str(y): float(v) for y, v in zip(YEARS, uv_smooth)},
            "annual_uv_cache_files": annual_cache,
        },
        "critic7_raw_annual_uv": {
            "weights": weights7,
            "standard_deviation_after_minmax": {k: float(v) for k, v in zip(NAMES, sd)},
            "conflict": {k: float(v) for k, v in zip(NAMES, conflict)},
            "information": {k: float(v) for k, v in zip(NAMES, info)},
            "correlation_matrix_order": list(NAMES),
            "correlation_matrix": R.tolist(),
        },
        "critic7_11yr_smoothed_all_variables": {
            "weights": weights7_smooth,
            "standard_deviation_after_minmax": {k: float(v) for k, v in zip(NAMES, sds)},
            "conflict": {k: float(v) for k, v in zip(NAMES, conflicts)},
            "information": {k: float(v) for k, v in zip(NAMES, infos)},
            "correlation_matrix_order": list(NAMES),
            "correlation_matrix": Rs.tolist(),
        },
        "critic6_11yr_smoothed_all_variables": {
            "weights": weights6_smooth,
            "standard_deviation_after_minmax": {k: float(v) for k, v in zip(NAMES[:6], sd6s)},
            "conflict": {k: float(v) for k, v in zip(NAMES[:6], conflict6s)},
            "information": {k: float(v) for k, v in zip(NAMES[:6], info6s)},
            "correlation_matrix_order": list(NAMES[:6]),
            "correlation_matrix": R6s.tolist(),
        },
        "limitations": [
            "The original six variables come from the calibrated HAF emulator, while UV comes from UKESM1-0-LL.",
            "CRITIC measures contrast and non-redundancy, not biological importance.",
        ],
    }
    weights_path = OUT / "critic_weights_2015_2100.json"
    weights_path.write_text(json.dumps(weights_only, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"raw": weights7, "critic7_11yr_smooth": weights7_smooth,
                      "critic6_11yr_smooth": weights6_smooth},
                     ensure_ascii=False, indent=2))
    print(f"WROTE {weights_path}")
    return

    # Six-variable relative weights passed to the existing tiering routine;
    # their total contribution is restored explicitly with six_total below.
    six_total = float(w[:6].sum())
    six_relative = {k: float(w[i] / six_total) for i, k in enumerate(chs.CHS_VARS)}
    wuv = float(w[6])

    fields, lat, lon = load_period_fields()
    ref_uv = np.mean([fields[s]["reference"] for s in base.SCENARIOS], axis=0)
    sT0, U0 = chs.tier_series(refout, six_relative, scales)
    i2020 = int(np.argmin(np.abs(all_years - 2020)))
    baseline_climate = G["P2d"] * sT0[i2020] + U0[i2020]
    land, area = G["land2d"], G["area2d"]
    den = area[land].sum()

    # Baseline UV anomaly is zero by construction. Refit threshold after the
    # new CRITIC weights, rather than reusing the old 1/7 threshold.
    baseline_chs = G["B2d"] + six_total * baseline_climate
    taus = {str(p): weighted_percentile(baseline_chs[land], area[land], p)
            for p in (85, 90, 95)}
    tau = taus["90"]

    rows = []
    for scen in base.SCENARIOS:
        out = emulator.run_emulator(params, all_years, ssp=scen, chem=False)
        sT, U = chs.tier_series(out, six_relative, scales)
        for label, year in (("mid", 2050), ("late", 2090)):
            i = int(np.argmin(np.abs(all_years - year)))
            climate = G["P2d"] * sT[i] + U[i]
            z = np.log(np.maximum(fields[scen][label], 1e-9) /
                       np.maximum(ref_uv, 1e-9)) / np.log(1.10)
            z05 = base.regrid(z, lat, lon, G["lat"], G["lon"])
            no_uv = G["B2d"] + six_total * climate
            with_uv = no_uv + wuv * z05
            h0 = float(area[land & (no_uv < tau)].sum() / den)
            h1 = float(area[land & (with_uv < tau)].sum() / den)
            rows.append({
                "scenario": scen, "period": label,
                "haf_no_uv": h0, "haf_with_uv": h1,
                "delta_haf": h1 - h0,
                "land_mean_uv_change_pct": float(np.average(
                    (np.exp(z05[land] * np.log(1.10)) - 1.0) * 100.0,
                    weights=area[land])),
            })

    result = {
        "status": "final_analysis",
        "training": {
            "scenario": "ssp245", "years": [2015, 2100],
            "sample_unit": "annual values (n=86)",
            "uv_metric": "area-weighted global-land mean erythemal daily dose",
            "annual_uv_values": {str(y): float(v) for y, v in zip(YEARS, uv_series)},
            "annual_uv_cache_files": annual_cache,
        },
        "critic7_weights": weights7,
        "critic_diagnostics": {
            "standard_deviation_after_minmax": {k: float(v) for k, v in zip(NAMES, sd)},
            "conflict": {k: float(v) for k, v in zip(NAMES, conflict)},
            "information": {k: float(v) for k, v in zip(NAMES, info)},
            "correlation_matrix_order": list(NAMES),
            "correlation_matrix": R.tolist(),
        },
        "threshold": {
            "reference": "2015-2025 modern baseline; UV anomaly zero; climate state at 2020",
            "weighted_percentiles": taus,
            "primary_percentile": 90,
        },
        "rows": rows,
        "limitations": [
            "Only 11 annual samples are available in the modern training window.",
            "The original six variables come from the calibrated HAF emulator, while UV comes from UKESM1-0-LL.",
            "CRITIC measures contrast and non-redundancy, not biological importance.",
            "UV hazard scaling remains 10% dose change per one hazard unit.",
        ],
    }
    p = OUT / "composite_analysis_summary.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"weights": weights7, "tau90": tau, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"WROTE {p}")


if __name__ == "__main__":
    main()
