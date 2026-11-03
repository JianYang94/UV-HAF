"""Composite Hazard Score (CHS) and Habitable Area Fraction (HAF).

Implements the hardened metric from EH_shallow.tex on a REAL 0.5-degree grid
(see `grid.py`) with the proposal's *tiered* disaggregation:

  - each variable is *oriented* so larger = more hazard, then standardised using
    the PREINDUSTRIAL baseline (1750-1800) mean (full-period scale -- see the
    `_standardise` note);
  - tier (i): surface-temperature-driven variables are pattern-scaled onto the
    present-day land warming pattern; tier (iii): variables with no defensible
    land field are spatially-uniform modifiers (tier (ii), the locally-forced
    vars, needs the gridded WHI stack and is not yet wired in);
  - CHS(cell, t) = P_temp(cell)*sT(t) + B(cell) + U(t), with sT the tier-(i)
    driver, U the tier-(iii) uniform sum, and B the present-day baseline field;
  - HAF = area-weighted fraction of LAND with CHS below a reference level tau
    (90th percentile of the preindustrial CHS distribution ~ percentile of B).
    We also report a SMOOTH exceedance and the sensitivity to the percentile,
    per the proposal's fix to the "threshold-free" overclaim.
"""
from __future__ import annotations

import numpy as np

from . import data, grid

# CHS variables and their hazard orientation (+1: larger is worse; -1: smaller is worse)
CHS_VARS = {
    "gmst": +1, "co2": +1, "sst": +1, "ohc": +1, "ph": -1, "omega": -1,
}
# Tier assignment for disaggregation onto the grid (EH_shallow.tex):
#   tier (i): a defensible LAND spatial pattern (surface air temperature);
#   tier (iii): no defensible land field -> spatially-uniform global modifier.
# (sst/ohc/ph/omega are ocean quantities; on the land HAF they act as uniform
#  modifiers. tier (ii) -- groundwater/land-use, the only tier that may claim
#  emergent hotspots -- arrives with the WHI stack.)
TIER_I = ("gmst",)
TIER_III = ("co2", "sst", "ohc", "ph", "omega")


def _weights(weights: dict | None) -> dict:
    names = list(CHS_VARS)
    if weights is None:
        return {k: 1.0 / len(names) for k in names}
    s = sum(weights.values())
    return {k: weights[k] / s for k in names}


def _standardise(series: np.ndarray, years: np.ndarray, sign: int,
                 scale: tuple | None = None) -> np.ndarray:
    """Orient by `sign`, anchor the mean at the preindustrial baseline, and scale.

    The preindustrial window is near-constant for monotonic variables, so its
    *variance* is ~0 and using it
    as the z-score scale makes the score explode. We therefore anchor the MEAN at
    the preindustrial baseline (so anomalies are measured against it) but take the
    SCALE from the full analysis period -- a pragmatic, documented choice. A
    production metric would standardise per-variable against a physically
    meaningful range, not a degenerate preindustrial variance.

    `scale` = (mu, sd) overrides the per-series constants with a COMMON reference
    (see `reference_scales`), so the CHS is comparable across draws and scenarios
    -- without it, a high-emission scenario self-standardises to its own large
    variance and would spuriously look *more* habitable than a low one.
    """
    oriented = sign * series
    if scale is not None:
        mu, sd = scale
        return (oriented - mu) / sd
    m = (years >= data.PREINDUSTRIAL[0]) & (years <= data.PREINDUSTRIAL[1])
    mu = oriented[m].mean()
    sd = oriented.std()  # full-period scale (preindustrial-only sd is ~0)
    if sd == 0 or not np.isfinite(sd):
        sd = 1.0
    return (oriented - mu) / sd


def reference_scales(out_ref: dict) -> dict:
    """Per-variable (mu, sd) standardisation constants from a reference run.

    Fixing the scale to one reference scenario/draw makes the CHS a well-defined
    metric that is comparable across the posterior ensemble and across SSPs.
    """
    years = out_ref["year"]
    sc = {}
    for k, sign in CHS_VARS.items():
        oriented = sign * out_ref[k]
        m = (years >= data.PREINDUSTRIAL[0]) & (years <= data.PREINDUSTRIAL[1])
        sd = oriented.std()
        sc[k] = (float(oriented[m].mean()),
                 float(sd) if (sd and np.isfinite(sd)) else 1.0)
    return sc


def reference_scales_window(out_ref: dict,
                            center_window: tuple[int, int] = (1750, 1800),
                            scale_window: tuple[int, int] = (1750, 2100)) -> dict:
    """Fixed preindustrial-centred standardisation constants.

    The oriented mean is calculated over the preindustrial ``center_window``.
    The scale is the temporal standard deviation over ``scale_window`` from
    the common SSP2-4.5 reference run supplied by the caller.  The default
    scale ends in 2100 so that observations beyond the study period do not
    determine the magnitude of the 2015--2100 score.
    """
    years = np.asarray(out_ref["year"])
    m_center = ((years >= center_window[0]) & (years <= center_window[1]))
    m_scale = ((years >= scale_window[0]) & (years <= scale_window[1]))
    if not m_center.any() or not m_scale.any():
        raise ValueError("Requested reference windows are absent")
    sc = {}
    for k, sign in CHS_VARS.items():
        oriented = sign * np.asarray(out_ref[k], float)
        mu = float(np.mean(oriented[m_center]))
        sd = float(np.std(oriented[m_scale], ddof=1))
        sc[k] = (mu, sd if (sd > 0 and np.isfinite(sd)) else 1.0)
    return sc


def _standardised_matrix(out_ref: dict, scales: dict | None = None) -> np.ndarray:
    """[n_time, 6] matrix of the oriented, standardised carried series.

    Columns follow the order of `CHS_VARS`. Uses the common reference `scales`
    when supplied (so the matrix matches the CHS the pipeline actually scores).
    """
    years = out_ref["year"]
    sc = scales or {}
    cols = [_standardise(out_ref[k], years, CHS_VARS[k], sc.get(k)) for k in CHS_VARS]
    return np.column_stack(cols)


def objective_weights(out_ref: dict, scales: dict | None = None,
                      method: str = "critic") -> dict:
    """Data-driven CHS weights from the reference-run series matrix.

    The six carried variables are strongly collinear (all monotone anthropogenic),
    so equal weights are a defensible but arbitrary reference. These objective
    multi-criteria schemes derive a weight vector purely from the reference run --
    no tuning, fully reproducible -- and down-weight the most redundant variables.

      - ``critic`` (CRITIC, Diakoulaki 1995): w_j proportional to
        sigma_j * sum_k (1 - r_jk), i.e. contrast (spread) times conflict
        (1 - correlation). Redundant, highly-correlated variables get less weight.
      - ``entropy`` (Shannon entropy weight method): w_j proportional to (1 - e_j),
        e_j = -(1/ln n) sum_i p_ij ln p_ij, on min-max-normalised columns.
      - ``pca`` (first-principal-component / communality): w_j proportional to
        |loading_j1| * sqrt(explained_var_1).

    Returns a dict over `CHS_VARS` summing to 1.
    """
    names = list(CHS_VARS)
    X = _standardised_matrix(out_ref, scales)              # [n_time, 6]
    # min-max normalise each column to [0, 1] (all methods below assume this)
    lo = X.min(0); rng = np.ptp(X, axis=0)
    rng = np.where(rng > 0, rng, 1.0)
    Z = (X - lo) / rng

    method = method.lower()
    if method == "critic":
        sd = Z.std(0, ddof=1)
        R = np.corrcoef(Z, rowvar=False)
        R = np.where(np.isfinite(R), R, 0.0)
        conflict = (1.0 - R).sum(0)                        # sum_k (1 - r_jk)
        c = sd * conflict
        w = c / c.sum() if c.sum() > 0 else np.full(len(names), 1.0 / len(names))
    elif method == "entropy":
        col = Z.sum(0)
        col = np.where(col > 0, col, 1.0)
        P = Z / col
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -(P * np.where(P > 0, np.log(P), 0.0)).sum(0) / np.log(Z.shape[0])
        d = 1.0 - e                                        # degree of divergence
        w = d / d.sum() if d.sum() > 0 else np.full(len(names), 1.0 / len(names))
    elif method == "pca":
        Xc = Z - Z.mean(0)
        C = np.cov(Xc, rowvar=False)
        evals, evecs = np.linalg.eigh(C)
        k = int(np.argmax(evals))
        load = np.abs(evecs[:, k])
        expl = evals[k] / evals.sum() if evals.sum() > 0 else 1.0
        c = load * np.sqrt(max(expl, 0.0))
        w = c / c.sum() if c.sum() > 0 else np.full(len(names), 1.0 / len(names))
    else:
        raise ValueError(f"unknown weighting method: {method!r}")
    return {k: float(w[i]) for i, k in enumerate(names)}


def composite_hazard(out: dict, weights: dict | None = None,
                     scales: dict | None = None) -> np.ndarray:
    """Global (area-mean) CHS(t) from an emulator output dict.

    Equals sT(t) + U(t): the area-mean of the per-cell field, since P_temp has
    global mean 1 and B is centred over land.
    """
    sT, U = tier_series(out, weights, scales)
    return sT + U


def tier_series(out: dict, weights: dict | None = None, scales: dict | None = None):
    """Split the standardised CHS into the tier-(i) driver sT(t) and uniform U(t).

    sT(t) = sum_{tier i} w_m * vtilde_m(t)  (weight folded in, so the tier-(i)
    field P_temp(cell)*sT(t) has area-mean sT(t)); U(t) = sum_{tier iii} w_m * vtilde_m.
    Pass `scales` (from `reference_scales`) for a common cross-scenario scale.
    """
    years = out["year"]
    w = _weights(weights)
    sc = scales or {}
    sT = np.zeros(len(years))
    for k in TIER_I:
        sT += w[k] * _standardise(out[k], years, CHS_VARS[k], sc.get(k))
    U = np.zeros(len(years))
    for k in TIER_III:
        U += w[k] * _standardise(out[k], years, CHS_VARS[k], sc.get(k))
    return sT, U


def haf_ensemble(sT_all: np.ndarray, U_all: np.ndarray, years: np.ndarray,
                 percentile: float = 90.0):
    """HAF(t) for a whole ensemble of draws via the precomputed grid lookup.

    sT_all, U_all: [n_draw, n_time]. Returns HAF [n_draw, n_time]. The HAF(sT, c)
    table is built once (covering the ensemble's range) and queried for all draws.
    """
    G = grid.build()
    tau = grid.tau_of(percentile, G)
    sT_all = np.atleast_2d(sT_all)
    U_all = np.atleast_2d(U_all)
    c = tau - U_all
    tab = grid.make_table(float(sT_all.max()) * 1.15 + 1e-3,
                          float(c.min()) - 0.5, float(c.max()) + 0.5, G)
    return tab(sT_all, c)


def haf_trajectory(out: dict, percentile: float = 90.0, weights: dict | None = None,
                   scales: dict | None = None):
    """Single-draw HAF on the real grid: hard indicator + smooth exceedance + tau."""
    years = out["year"]
    sT, U = tier_series(out, weights, scales)
    G = grid.build()
    tau = grid.tau_of(percentile, G)
    haf = haf_ensemble(sT[None, :], U[None, :], years, percentile)[0]
    # smooth exceedance for this draw: per-cell logistic, area-weighted over land
    # (float32 to bound the transient [n_land x n_time] field)
    P = G["P"].astype(np.float32)
    B = G["B"].astype(np.float32)
    area = G["area"]
    field = P[:, None] * sT[None, :].astype(np.float32) + B[:, None] + U[None, :].astype(np.float32)
    scale = max(float(G["B"].std()), 1e-6)
    smooth = (area[:, None] * _sigmoid((tau - field) / scale)).sum(0) / area.sum()
    return {"haf": haf, "haf_smooth": smooth, "tau": tau,
            "land_area_frac": G["land_area_frac"]}


def haf_percentile_sensitivity(out: dict, percentiles=(80, 85, 90, 95, 99),
                               weights: dict | None = None, scales: dict | None = None):
    """HAF(t) for several tau-percentiles (the proposal's sensitivity report)."""
    sT, U = tier_series(out, weights, scales)
    years = out["year"]
    return {p: haf_ensemble(sT[None, :], U[None, :], years, percentile=p)[0]
            for p in percentiles}


def chs_field(out: dict, year: int, weights: dict | None = None,
              scales: dict | None = None):
    """2-D CHS field [nlat, nlon] (ocean masked NaN) at `year`, for mapping."""
    sT, U = tier_series(out, weights, scales)
    years = out["year"]
    i = int(np.argmin(np.abs(years - year)))
    return grid.field_at(float(sT[i]), float(U[i]))


def time_of_emergence(out: dict, percentile: float = 90.0,
                      weights: dict | None = None, scales: dict | None = None,
                      G=None, start_year: int = 2020):
    """First year each currently-habitable land cell crosses the threshold tau.

    Emergence is measured relative to `start_year` (default 2020, the present):
    for land that is still below tau in `start_year`, `toe2d` gives the first year
    it crosses tau (NaN = never within 2300); land already above tau in
    `start_year` is set to the sentinel `start_year - 1` (drawn distinctly on the
    map), and ocean is NaN. This localises *when* -- not just how much --
    presently-habitable land is projected to be lost, cell by cell.

    Returns (toe2d, summary) with the area-weighted fraction of presently-habitable
    land that emerges by 2300 and the area-weighted median emergence year over it.
    """
    years = out["year"]
    sT, U = tier_series(out, weights, scales)
    G = G or grid.build()
    tau = grid.tau_of(percentile, G)
    P = G["P"].astype(np.float64)
    B = G["B"].astype(np.float64)
    area = G["area"]
    field = P[:, None] * sT[None, :] + B[:, None] + U[None, :]      # [n_land, n_time]
    s0 = int(np.argmin(np.abs(years - start_year)))
    habitable0 = field[:, s0] < tau                                 # habitable today
    fut = field[:, s0:] >= tau
    emerges = habitable0 & fut.any(1)
    first_idx = np.where(emerges, fut.argmax(1) + s0, -1)
    toe_land = np.full(field.shape[0], np.nan)
    toe_land[emerges] = years[np.clip(first_idx[emerges], 0, len(years) - 1)]
    toe_land[~habitable0] = start_year - 1                          # already hazardous
    if emerges.any():
        yv = toe_land[emerges]; av = area[emerges]
        order = np.argsort(yv)
        cw = np.cumsum(av[order]) / av.sum()
        med_year = float(np.interp(0.5, cw, yv[order]))
    else:
        med_year = float("nan")
    frac_hab = float(area[habitable0].sum() / area.sum())
    frac_emerges = float(area[emerges].sum() / max(area[habitable0].sum(), 1e-12))
    toe2d = np.full(G["land2d"].shape, np.nan, dtype=float)
    toe2d[G["land2d"]] = toe_land
    summary = {"start_year": start_year,
               "frac_present_habitable_land": frac_hab,
               "frac_of_habitable_that_emerges_by_2300": frac_emerges,
               "median_emergence_year": med_year}
    return toe2d, summary


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def headline_config(datadir=None):
    """Posterior-mean parameters and headline weights from the last pipeline run.

    The independent evaluation modules call this so that they score the same
    calibrated configuration the main results use, instead of a value fixed in
    their own source. Returns ``(None, None)`` when no run output is present, in
    which case the caller keeps its documented fallback.
    """
    import json as _json
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    roots = [datadir] if datadir else []
    roots += [_os.path.join(_os.path.dirname(here), "outputs"),
              _os.path.join(here, "out")]
    for r in roots:
        if not r:
            continue
        p = _os.path.join(r, "metrics.json")
        if _os.path.exists(p):
            try:
                with open(p) as fh:
                    m = _json.load(fh)
                theta = {k: float(v["mean"]) for k, v in m["posterior"].items()}
                ow = m["objective_weights"]
                w = {k: float(v) for k, v in
                     ow["weights"][ow["primary_method"]].items()}
                return theta, w
            except Exception:
                continue
    return None, None
