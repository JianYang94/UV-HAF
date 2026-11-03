"""Construct the 0.5 degree grid and spatial fields used for CHS and HAF.

The module combines a Natural Earth land mask, an area-normalised temperature
pattern and the gridded Water Hazard Index baseline used by the analysis. The
analytic baseline builder remains available only as an explicitly flagged
fallback and is rejected by publication drivers.

The per-cell composite hazard separates as

    CHS(cell, t) = P_temp(cell) * sT(t)  +  B(cell)  +  U(t)

with P_temp the (fixed) tier-(i) land warming pattern, sT(t) the standardised
global temperature driver, B(cell) the fixed present-day baseline field, and U(t)
the tier-(iii) spatially-uniform contribution. Because the preindustrial drivers
are ~0 by construction, tau (the HAF reference level) is a fixed percentile of B,
and HAF(sT, U) is a fixed 2-D function -- precomputed once as a lookup table so
the whole posterior ensemble is evaluated by cheap interpolation rather than
materialising a [land-cell x year] field per draw.

The released Natural Earth land mask is used for the publication calculations.
"""
from __future__ import annotations

import os

import numpy as np

from . import data

RES = 0.5  # degrees
SIGMA_AGG = 0.6  # aggregation/Jensen heterogeneity carried in B (sigma_agg)
# Total spread used only by the explicitly flagged analytic fallback field.
B_STD = 1.8
# Southern limit of the analysis domain; land south of this is the Antarctic ice
# sheet. Set to None to evaluate the metric over the full Natural Earth land mask.
ICE_SHEET_LAT = -60.0
_CACHE = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(_CACHE, exist_ok=True)

_GRID = None  # lazily-built singleton


def _land_mask(lon: np.ndarray, lat: np.ndarray):
    """Boolean land mask [nlat, nlon]; real (Natural Earth) when possible."""
    released = os.path.join(os.path.dirname(__file__), "released", "land_mask_0p5deg.npz")
    if lon.size == 720 and lat.size == 360 and os.path.exists(released):
        return np.load(released)["mask"].astype(bool), "Natural Earth land_110 (released 0.5-degree mask)"
    cache = os.path.join(_CACHE, f"land_mask_{lon.size}x{lat.size}.npy")
    if os.path.exists(cache):
        return np.load(cache), "Natural Earth land_110 (cached)"
    try:
        import regionmask
        m = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
        land = np.isfinite(np.asarray(m.values))
        np.save(cache, land)
        return land, "Natural Earth land_110 (regionmask)"
    except Exception:
        # coarse analytic fallback: latitudinal land fraction, flagged as such
        frac = np.clip(0.25 + 0.45 * (lat / 90.0) ** 2
                       + 0.15 * np.cos(np.radians(lat)) * (lat > 0), 0.05, 0.95)
        rng = np.random.default_rng(0)
        land = rng.random((lat.size, lon.size)) < frac[:, None]
        return land, "ANALYTIC FALLBACK (no regionmask/network)"


def _temperature_pattern(lat2d, land2d, area2d):
    """Tier-(i) surface warming pattern, area-normalised to global mean 1.

    Polar amplification (poles ~2.6x equator) and a land/ocean contrast (land
    warms ~1.4x the global mean). Only land cells enter the HAF, but the pattern
    is normalised over the whole globe so land cells correctly sit above 1.
    """
    raw = 1.0 + 1.6 * (np.abs(lat2d) / 90.0) ** 2          # polar amplification
    raw = np.where(land2d, 1.4 * raw, raw)                  # land/ocean contrast
    gmean = (area2d * raw).sum() / area2d.sum()
    return raw / gmean


def _baseline_field(lat2d, land2d, sigma_agg, seed, b_std=None):
    """Construct the analytic fallback baseline field B(cell).

    Smooth structure -- elevated hazard in the subtropical dry belts (~15-35 deg)
    and the cold high latitudes (poleward of ~55 deg) -- plus sigma_agg sub-grid
    heterogeneity (the aggregation/Jensen term). The whole field is rescaled to a
    total land std of `b_std` (B_STD). Publication calculations replace this
    fallback with the released gridded Water Hazard Index field.
    """
    b_std = B_STD if b_std is None else b_std
    rng = np.random.default_rng(seed)
    al = np.abs(lat2d)
    dry = np.exp(-((al - 27.0) / 11.0) ** 2)               # subtropical dry belts
    cold = np.clip((al - 55.0) / 35.0, 0.0, 1.0)           # cold high latitudes
    base = dry + 0.9 * cold
    noise = rng.normal(0.0, sigma_agg, lat2d.shape)
    B = base + noise
    B = B - B[land2d].mean()                               # centre over land
    s = B[land2d].std() or 1.0
    return (B / s) * b_std                                 # rescale to target std


def build(res: float = RES, seed: int = 0, sigma_agg: float = SIGMA_AGG):
    """Build (and cache) the static grid: mask, areas, P_temp, B over land."""
    global _GRID
    if _GRID is not None and _GRID["res"] == res:
        return _GRID
    lon = np.arange(-180 + res / 2, 180, res)
    lat = np.arange(-90 + res / 2, 90, res)
    lon2d, lat2d = np.meshgrid(lon, lat)
    area2d = np.cos(np.radians(lat))[:, None] * np.ones((1, lon.size))
    land2d, source = _land_mask(lon, lat)
    # Analysis domain: the ice-free land surface. The Antarctic ice sheet carries
    # no permanent population and no surface-water hazard observation, so leaving
    # it in would add a large block of land that is classified habitable by an
    # imputed value rather than by data. Excluding it also removes the only
    # substantial gap in the observed water-hazard field.
    if ICE_SHEET_LAT is not None:
        land2d = land2d & (lat2d > ICE_SHEET_LAT)
        source += f"; ice-free domain (south of {abs(ICE_SHEET_LAT):.0f} deg S excluded)"
    P = _temperature_pattern(lat2d, land2d, area2d)
    B = _baseline_field(lat2d, land2d, sigma_agg, seed)
    m = land2d
    _GRID = {
        "res": res, "lon": lon, "lat": lat, "lon2d": lon2d, "lat2d": lat2d,
        "area2d": area2d, "land2d": land2d, "source": source,
        "P2d": P, "B2d": B,
        # flattened land-only arrays for the HAF computation
        "P": P[m].astype(np.float64), "B": B[m].astype(np.float64),
        "area": area2d[m].astype(np.float64),
        "n_land": int(m.sum()), "n_cells": int(m.size),
        "land_area_frac": float(area2d[m].sum() / area2d.sum()),
        "sigma_agg": sigma_agg,
    }
    return _GRID


def set_baseline(B2d, G=None):
    """Override the baseline land-hazard field B (e.g. with the real gridded WHI).

    Replaces the analytic fallback produced by `_baseline_field`. `B2d` is a
    [nlat, nlon] field (NaN over
    ocean); the flattened land-only `B` is refreshed to match. Mutates the cached
    grid singleton so all downstream HAF calls (which call `build()`) see it.
    """
    G = G or build()
    B2d = np.asarray(B2d, dtype=float)
    land = G["land2d"]
    G["B2d"] = np.where(land, B2d, np.nan)
    G["B"] = B2d[land].astype(np.float64)
    G["baseline_source"] = "real gridded WHI"
    return G


def tau_of(percentile: float, G=None) -> float:
    """HAF reference level: `percentile` of the preindustrial field (~ B) over land.

    Area-weighted percentile of B (preindustrial sT=U=0), so HAF(preindustrial)
    ~= percentile/100 by construction.
    """
    G = G or build()
    B, area = G["B"], G["area"]
    order = np.argsort(B)
    cw = np.cumsum(area[order]) / area.sum()
    return float(np.interp(percentile / 100.0, cw, B[order]))


class HafTable:
    """Precomputed HAF(sT, c) lookup, where c = tau - U; built once per grid."""

    def __init__(self, G, sT_grid, c_grid):
        self.sT_grid = sT_grid
        self.c_grid = c_grid
        P, B, area = G["P"], G["B"], G["area"]
        A = area.sum()
        tab = np.empty((sT_grid.size, c_grid.size))
        for a, s in enumerate(sT_grid):
            v = P * s + B
            order = np.argsort(v)
            cw = np.cumsum(area[order]) / A
            tab[a] = np.interp(c_grid, v[order], cw, left=0.0, right=1.0)
        self.tab = tab

    def __call__(self, sT, c):
        """Bilinear interpolation at arrays (sT, c) -> HAF (clipped to [0,1])."""
        sT = np.asarray(sT, float)
        c = np.asarray(c, float)
        ia = np.clip(np.searchsorted(self.sT_grid, sT) - 1, 0, self.sT_grid.size - 2)
        ib = np.clip(np.searchsorted(self.c_grid, c) - 1, 0, self.c_grid.size - 2)
        s0, s1 = self.sT_grid[ia], self.sT_grid[ia + 1]
        c0, c1 = self.c_grid[ib], self.c_grid[ib + 1]
        ws = np.clip((sT - s0) / (s1 - s0), 0, 1)
        wc = np.clip((c - c0) / (c1 - c0), 0, 1)
        t = self.tab
        f = ((1 - ws) * (1 - wc) * t[ia, ib]
             + ws * (1 - wc) * t[ia + 1, ib]
             + (1 - ws) * wc * t[ia, ib + 1]
             + ws * wc * t[ia + 1, ib + 1])
        return np.clip(f, 0.0, 1.0)


def make_table(sT_max: float, c_lo: float, c_hi: float, G=None,
               n_sT: int = 140, n_c: int = 200) -> HafTable:
    """Build a HafTable covering sT in [0, sT_max] and c in [c_lo, c_hi]."""
    G = G or build()
    sT_grid = np.linspace(0.0, max(sT_max, 1e-3), n_sT)
    c_grid = np.linspace(c_lo, c_hi, n_c)
    return HafTable(G, sT_grid, c_grid)


def field_at(sT_val: float, U_val: float, G=None):
    """Full 2-D CHS field at one time slice (ocean masked NaN), for mapping."""
    G = G or build()
    f = G["P2d"] * sT_val + G["B2d"] + U_val
    return np.where(G["land2d"], f, np.nan)
