"""Public observational and forcing data used by the climate emulator.

Sources are public, downloaded once and cached under ``haf_core/_cache``:
  - HadCRUT5 global annual mean surface temperature anomaly (Met Office), the
    SMC calibration target, with 2.5/97.5% confidence limits for the likelihood.
  - IPCC AR6 effective radiative forcing (ERF) 1750-2019 (Forster et al. 2021,
    via chrisroadmap/ar6), the real forcing that drives the emulator.
  - CO2 concentration pathway: a small embedded historical+SSP2-4.5 table
    (CMIP6/Meinshausen-style values), interpolated. Used for ocean chemistry
    and as a CHS variable.

If the network is unavailable, HadCRUT5/AR6 fall back to embedded coarse tables
so the pipeline still runs (clearly flagged in the returned metadata).
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd

_CACHE = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(_CACHE, exist_ok=True)

HADCRUT5_URL = (
    "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/"
    "analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv"
)
AR6_ERF_URL = (
    "https://raw.githubusercontent.com/chrisroadmap/ar6/main/"
    "data_output/AR6_ERF_1750-2019.csv"
)
OHC_URL = (
    "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/"
    "3M_HEAT_CONTENT/DATA/basin/yearly/h22-w0-2000m.dat"
)
# Two further INDEPENDENT global temperature reconstructions, used (alongside
# HadCRUT5) to estimate the emulator's structural error without conflating it
# with the non-independent CMIP6 model spread (see structural.py).
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
BERKELEY_URL = (
    "https://berkeley-earth-temperature.s3.amazonaws.com/Global/"
    "Land_and_Ocean_summary.txt"
)
# CRU 1961-90 absolute mean monthly surface temperature climatology (5 deg),
# used to define the human climate niche (Xu et al. 2020) in absolute MAT.
MAT_URL = "https://crudata.uea.ac.uk/cru/data/temperature/absolute.nc"
# NASA GISTEMP gridded (2 deg, 1200 km interpolated), for the pattern-stationarity
# test of the tier-(i) pattern-scaling assumption (structural.py / stationarity.py).
GISTEMP_GRID_URL = (
    "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz"
)
# GDHY (Iizumi & Sakai 2020, doi:10.1038/s41597-020-0433-7): 0.5 deg gridded
# annual yields (t/ha) of major crops 1981-2016, open via PANGAEA (no login).
# Used to validate the composite CHS/HAF against an independent agronomic impact
# (Major-3, the impacts half: cropyield.py). The .nc4 grid (lat -89.75..89.75
# ascending, lon 0.25..359.75 on 0-360) matches the 0.5 degree grid after
# a longitude roll to -180..180 -- no interpolation needed.
GDHY_URL = ("https://store.pangaea.de/Publications/IizumiT_2019/"
            "gdhy_v1.2_v1.3_20190128.zip")
# The 4 staple crops with global coverage in GDHY (the *_major/_second/_spring/
# _winter folders are regional splits of these and are not used here).
GDHY_CROPS = ("maize", "rice", "wheat", "soybean")

# Reference period used to put model and observations on a common baseline.
REF_PERIOD = (1850, 1900)
# Preindustrial location window for standardising the six climate and ocean indicators.
PREINDUSTRIAL = (1750, 1800)
# OHC has no preindustrial record; reference it to its own common decade instead.
OHC_REF_PERIOD = (2005, 2014)

F2X = 3.93  # W m-2, ERF for CO2 doubling (AR6 central)
CO2_PI = 278.0  # ppm, preindustrial CO2


def _download(url: str, fname: str, timeout: int = 60) -> str | None:
    """Download `url` to the cache as `fname`; return path, or None on failure."""
    path = os.path.join(_CACHE, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "uv-haf/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data:
            return None
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


# ----------------------------------------------------------------------------- #
# Data-provenance guard: never let SYNTHETIC fallback data silently enter a figure.
# Every loader tags its output with a `source` string; the embedded coarse tables
# and "UNAVAILABLE" returns are fallbacks. Each fallback actually used this process
# is registered here so a figure driver can refuse (or be explicitly forced) to
# emit a publication artifact built on non-real data. See assert_real_data().
# ----------------------------------------------------------------------------- #
_PROVENANCE: list[str] = []   # synthetic-fallback source strings used this process


def _is_fallback(source: str) -> bool:
    s = (source or "").upper()
    return "FALLBACK" in s or "UNAVAILABLE" in s


def _record(source: str) -> str:
    """Register a data source; return it unchanged so it can wrap a `source=`
    assignment. Only fallback / unavailable sources are retained."""
    if _is_fallback(source):
        _PROVENANCE.append(source)
    return source


def fallbacks_used() -> list[str]:
    """Distinct synthetic-fallback sources loaded so far this process."""
    return sorted(set(_PROVENANCE))


def reset_provenance() -> None:
    _PROVENANCE.clear()


def assert_real_data(context: str = "", extra: list[str] | None = None) -> None:
    """Raise RuntimeError if any input loaded this process used a SYNTHETIC
    fallback. `extra` adds non-loader provenance to the check. Call this in any
    driver that writes a publication figure so synthetic fallback data can never
    enter one silently."""
    bad = fallbacks_used() + [e for e in (extra or [])
                              if _is_fallback(e)]
    if bad:
        raise RuntimeError(
            "Refusing to emit a publication artifact"
            + (f" [{context}]" if context else "")
            + " because these inputs used SYNTHETIC FALLBACK data, not real "
              "downloads:\n  - " + "\n  - ".join(bad)
            + "\nRe-run with network access (so the real datasets are fetched and "
              "cached), or explicitly opt in to a clearly NON-PUBLICATION run.")


# ----------------------------------------------------------------------------- #
# Embedded fallbacks (coarse, used only if the network is down)
# ----------------------------------------------------------------------------- #
_HADCRUT5_FALLBACK = {  # decadal anomalies vs 1961-1990, approximate
    1850: -0.42, 1880: -0.26, 1900: -0.17, 1920: -0.25, 1940: 0.03,
    1960: -0.03, 1980: 0.09, 2000: 0.40, 2010: 0.62, 2020: 0.93, 2025: 1.19,
}
# AR6-style total ERF (W m-2), approximate decadal values
_ERF_FALLBACK = {
    1750: 0.0, 1850: 0.18, 1900: 0.35, 1950: 0.62, 1980: 1.25,
    2000: 1.95, 2010: 2.45, 2019: 2.84,
}
# The four CMIP6 Tier-1 SSP scenarios (O'Neill et al. 2016).
SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")

# CO2 ppm: shared historical record (<=2025), then per-SSP futures.
# Approximate CMIP6/Meinshausen (2020) concentrations; the SSP label is ~ its
# nominal 2100 forcing, and the 2100 CO2 values match the standard headline
# numbers (446 / 560 / 867 / 1135 ppm).
_CO2_HIST = {
    1750: 278, 1800: 283, 1850: 285, 1900: 296, 1950: 311, 1980: 339,
    2000: 369, 2010: 389, 2020: 412, 2025: 424,
}
_CO2_FUTURE = {
    "ssp126": {2030: 439, 2050: 456, 2060: 460, 2075: 455, 2100: 446,
               2150: 426, 2200: 412, 2250: 405, 2300: 401},
    "ssp245": {2030: 440, 2040: 463, 2050: 487, 2075: 534, 2100: 560,
               2150: 572, 2200: 563, 2250: 550, 2300: 543},
    "ssp370": {2030: 449, 2050: 544, 2075: 700, 2100: 867,
               2150: 1163, 2200: 1429, 2250: 1635, 2300: 1760},
    "ssp585": {2030: 460, 2050: 603, 2075: 856, 2100: 1135,
               2150: 1672, 2200: 1962, 2250: 2072, 2300: 2155},
}
# Total-ERF anchors for the post-2019 extension (W m-2). The 2100 values are the
# nominal SSP forcing levels (2.6 / 4.5 / 7.0 / 8.5 W m-2).
_ERF_FUTURE = {
    "ssp126": {2019: 2.84, 2030: 2.9, 2050: 3.0, 2075: 2.7, 2100: 2.6,
               2150: 2.4, 2200: 2.3, 2250: 2.2, 2300: 2.1},
    "ssp245": {2019: 2.84, 2030: 3.2, 2050: 3.7, 2075: 4.2, 2100: 4.5,
               2150: 4.5, 2200: 4.3, 2250: 4.15, 2300: 4.0},
    "ssp370": {2019: 2.84, 2030: 3.4, 2050: 4.5, 2075: 5.8, 2100: 7.0,
               2150: 8.5, 2200: 9.5, 2250: 10.0, 2300: 10.5},
    "ssp585": {2019: 2.84, 2030: 3.6, 2050: 5.0, 2075: 6.8, 2100: 8.5,
               2150: 10.5, 2200: 11.5, 2250: 12.0, 2300: 12.3},
}
# NOAA/NCEI world 0-2000 m OHC anomaly (native units 10^22 J), coarse anchors
# used only if the network is down (real file is parsed in full when available).
_OHC_FALLBACK = {
    2005: 10.171, 2006: 12.638, 2007: 12.394, 2008: 13.257,
    2023: 29.220, 2024: 30.316, 2025: 32.533,
}


def load_hadcrut5() -> pd.DataFrame:
    """Return DataFrame[year, gmst, lo, hi] of HadCRUT5 annual GMST anomaly."""
    path = _download(HADCRUT5_URL, "hadcrut5_annual_global.csv")
    if path is not None:
        df = pd.read_csv(path)
        df = df.rename(columns={
            df.columns[0]: "year", df.columns[1]: "gmst",
            df.columns[2]: "lo", df.columns[3]: "hi"})
        df.attrs["source"] = "HadCRUT5 (Met Office), downloaded"
        return df[["year", "gmst", "lo", "hi"]]
    yrs = np.array(sorted(_HADCRUT5_FALLBACK))
    vals = np.array([_HADCRUT5_FALLBACK[y] for y in yrs])
    full = np.arange(1850, 2026)
    g = np.interp(full, yrs, vals)
    df = pd.DataFrame({"year": full, "gmst": g, "lo": g - 0.15, "hi": g + 0.15})
    df.attrs["source"] = _record("EMBEDDED FALLBACK (no network)")
    return df


def load_gistemp() -> pd.DataFrame:
    """Return DataFrame[year, gmst] of NASA GISTEMP v4 annual GMST anomaly.

    Anomalies are vs 1951-1980 (the J-D annual column); `rebaseline` puts them on
    the common reference period. Independent of HadCRUT5 (different SST product,
    interpolation, and land analysis)."""
    path = _download(GISTEMP_URL, "gistemp_v4_global.csv")
    if path is not None:
        try:
            df = pd.read_csv(path, skiprows=1)
            y = pd.to_numeric(df["Year"], errors="coerce")
            g = pd.to_numeric(df["J-D"], errors="coerce")
            out = pd.DataFrame({"year": y, "gmst": g}).dropna()
            out["year"] = out["year"].astype(int)
            out.attrs["source"] = "NASA GISTEMP v4, downloaded"
            return out
        except Exception:
            pass
    yrs = np.array([1880, 1900, 1920, 1940, 1960, 1980, 2000, 2010, 2020, 2024])
    vals = np.array([-0.16, -0.08, -0.15, 0.13, -0.03, 0.26, 0.61, 0.72, 1.02, 1.29])
    full = np.arange(1880, 2025)
    out = pd.DataFrame({"year": full, "gmst": np.interp(full, yrs, vals)})
    out.attrs["source"] = _record("EMBEDDED FALLBACK (no network)")
    return out


def load_berkeley() -> pd.DataFrame:
    """Return DataFrame[year, gmst] of Berkeley Earth Land+Ocean annual anomaly.

    Anomalies are vs 1951-1980; `rebaseline` puts them on the common reference.
    Independent reconstruction (kriging-based, different station network)."""
    path = _download(BERKELEY_URL, "berkeley_land_ocean_summary.txt")
    if path is not None:
        try:
            rows = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            rows.append((int(float(parts[0])), float(parts[1])))
                        except ValueError:
                            continue
            out = pd.DataFrame(rows, columns=["year", "gmst"]).dropna()
            out.attrs["source"] = "Berkeley Earth Land+Ocean, downloaded"
            return out
        except Exception:
            pass
    yrs = np.array([1850, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2010, 2020, 2024])
    vals = np.array([-0.37, -0.20, -0.12, -0.20, 0.07, -0.02, 0.18, 0.52, 0.70, 1.02, 1.29])
    full = np.arange(1850, 2025)
    out = pd.DataFrame({"year": full, "gmst": np.interp(full, yrs, vals)})
    out.attrs["source"] = _record("EMBEDDED FALLBACK (no network)")
    return out


def load_mat_climatology() -> dict:
    """Return {lon, lat(asc), mat[lat,lon] (degC), source}: present-day mean annual
    temperature from the CRU 1961-90 absolute climatology (5 deg), used to define
    the human climate niche. Falls back to a coarse zonal field if unavailable."""
    path = _download(MAT_URL, "cru_absolute.nc")
    if path is not None:
        try:
            import netCDF4 as nc
            d = nc.Dataset(path)
            tem = np.ma.filled(d.variables["tem"][:].astype(float), np.nan)  # (12,lat,lon)
            mat = np.nanmean(tem, axis=0)
            lat = np.array(d.variables["lat"][:], dtype=float)
            lon = np.array(d.variables["lon"][:], dtype=float)
            order = np.argsort(lat)                      # ensure ascending lat
            return {"lon": lon, "lat": lat[order], "mat": mat[order],
                    "source": "CRU 1961-90 absolute MAT (5 deg), downloaded"}
        except Exception:
            pass
    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(-177.5, 180, 5.0)
    zonal = 30.0 - 60.0 * (np.abs(lat) / 90.0) ** 1.25
    return {"lon": lon, "lat": lat, "mat": zonal[:, None] * np.ones((1, lon.size)),
            "source": _record("EMBEDDED FALLBACK (no network)")}


def load_gistemp_gridded() -> dict:
    """Return {lon, lat, years, anom[year,lat,lon] (masked), source}: NASA GISTEMP
    gridded 2 deg annual temperature anomalies (vs 1951-1980). Used to test whether
    the tier-(i) warming pattern is stationary across warming levels."""
    gz = _download(GISTEMP_GRID_URL, "gistemp_gridded.nc.gz")
    if gz is not None:
        try:
            import gzip
            import shutil

            import netCDF4 as ncmod
            ncpath = os.path.join(_CACHE, "gistemp_gridded.nc")
            if not (os.path.exists(ncpath) and os.path.getsize(ncpath) > 0):
                with gzip.open(gz, "rb") as fi, open(ncpath, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            d = ncmod.Dataset(ncpath)
            t = d.variables["time"]
            dates = ncmod.num2date(t[:], t.units)
            yrs = np.array([dd.year for dd in dates])
            ta = d.variables["tempanomaly"][:]                 # (month, lat, lon)
            lat = np.array(d.variables["lat"][:], dtype=float)
            lon = np.array(d.variables["lon"][:], dtype=float)
            uy = np.unique(yrs)
            ann = np.ma.masked_all((uy.size, lat.size, lon.size))
            for i, y in enumerate(uy):
                ann[i] = np.ma.mean(ta[yrs == y], axis=0)
            return {"lon": lon, "lat": lat, "years": uy, "anom": ann,
                    "source": "NASA GISTEMP gridded 2 deg, downloaded"}
        except Exception:
            pass
    # coarse synthetic fallback: a fixed (stationary) polar-amplified pattern
    lat = np.arange(-89, 90, 2.0)
    lon = np.arange(-179, 180, 2.0)
    yrs = np.arange(1950, 2025)
    pat = 1.0 + 1.6 * (np.abs(lat) / 90.0) ** 2
    g = (yrs - 1950) / 75.0 * 1.2
    ann = np.ma.array(g[:, None, None] * pat[None, :, None] * np.ones((1, 1, lon.size)))
    return {"lon": lon, "lat": lat, "years": yrs, "anom": ann,
            "source": _record("EMBEDDED FALLBACK (no network)")}


def load_gdhy_yields(crops=GDHY_CROPS, grid_lon=None, grid_lat=None) -> dict:
    """Return {crop: yield[year,lat,lon] (t/ha, NaN off-cropland), years, source}.

    GDHY (Iizumi & Sakai 2020) 0.5 deg annual yields 1981-2016, downloaded as the
    open PANGAEA zip and cached. The native grid (lat -89.75..89.75 ascending, lon
    0.25..359.75 on a 0-360 frame) is realigned to the analysis grid's lon frame
    (-180..180 ascending) by a column roll -- the 0.5 deg lat grid is identical, so
    NO interpolation is applied. If `grid_lon`/`grid_lat` are given they are used as
    an assertion that the alignment matches the analysis grid.
    """
    import glob
    import zipfile

    zpath = _download(GDHY_URL, "gdhy_v1.2_v1.3.zip", timeout=180)
    ddir = os.path.join(_CACHE, "gdhy")
    if zpath is not None and not os.path.isdir(os.path.join(ddir, "maize")):
        try:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(ddir)
        except Exception:
            pass
    if not os.path.isdir(os.path.join(ddir, "maize")):
        return {"years": np.array([]), "source": _record("GDHY UNAVAILABLE (no network)"),
                **{c: None for c in crops}}

    import netCDF4 as ncmod
    out = {}
    years_ref = None
    for crop in crops:
        files = sorted(glob.glob(os.path.join(ddir, crop, "yield_*.nc4")))
        if not files:
            out[crop] = None
            continue
        yrs, cube, lat, lon = [], [], None, None
        for f in files:
            yr = int(os.path.basename(f).split("_")[1].split(".")[0])
            d = ncmod.Dataset(f)
            v = np.ma.filled(d.variables["var"][:].astype(float), np.nan)  # [lat,lon]
            if lat is None:
                lat = np.array(d.variables["lat"][:], dtype=float)
                lon = np.array(d.variables["lon"][:], dtype=float)
            d.close()
            yrs.append(yr)
            cube.append(v)
        cube = np.array(cube)                              # [year, lat, lon] on 0-360
        # roll 0-360 -> -180..180 ascending: cols [180.25..359.75] become the West
        west = lon >= 180.0
        cube = np.concatenate([cube[:, :, west], cube[:, :, ~west]], axis=2)
        lon = np.concatenate([lon[west] - 360.0, lon[~west]])
        out[crop] = cube
        years_ref = np.array(yrs) if years_ref is None else years_ref
        if grid_lat is not None:
            assert np.allclose(lat, grid_lat), "GDHY lat mismatch vs analysis grid"
        if grid_lon is not None:
            assert np.allclose(lon, grid_lon), "GDHY lon mismatch vs analysis grid"
        out["_lat"], out["_lon"] = lat, lon
    out["years"] = years_ref
    out["source"] = "GDHY v1.2/v1.3 (Iizumi & Sakai 2020), PANGAEA, downloaded"
    return out


def load_ar6_erf() -> pd.DataFrame:
    """Return DataFrame[year, total, co2] of AR6 effective radiative forcing."""
    path = _download(AR6_ERF_URL, "AR6_ERF_1750-2019.csv")
    if path is not None:
        df = pd.read_csv(path)
        out = df[["year", "total", "co2"]].copy()
        out.attrs["source"] = "IPCC AR6 ERF (Forster et al. 2021), downloaded"
        return out
    yrs = np.array(sorted(_ERF_FALLBACK))
    vals = np.array([_ERF_FALLBACK[y] for y in yrs])
    full = np.arange(1750, 2020)
    df = pd.DataFrame({"year": full, "total": np.interp(full, yrs, vals)})
    df["co2"] = 0.7 * df["total"]
    df.attrs["source"] = _record("EMBEDDED FALLBACK (no network)")
    return df


def load_ohc() -> pd.DataFrame:
    """Return DataFrame[year, ohc, sigma] of global 0-2000 m ocean heat content.

    Source: NOAA/NCEI world 0-2000 m OHC (file ``h22-w0-2000m.dat``), columns
    YEAR/WO/WOse. Native units are 10^22 J; this returns ZJ (10^21 J) so it is
    directly comparable to the emulator's ``ohc``. ``sigma`` is the reported
    standard error (also ZJ). The mid-year stamp (e.g. 2005.5) is floored to its
    integer year. Coverage is 2005-present (no preindustrial record), which is
    why the OHC likelihood term uses its own 2005-2014 baseline.
    """
    path = _download(OHC_URL, "ncei_ohc_0-2000m_yearly.dat")
    if path is not None:
        try:
            df = pd.read_csv(path, sep=r"\s+")
            yr = np.floor(df["YEAR"].to_numpy()).astype(int)
            out = pd.DataFrame({
                "year": yr,
                "ohc": df["WO"].to_numpy() * 10.0,      # 10^22 J -> ZJ
                "sigma": df["WOse"].to_numpy() * 10.0,
            })
            out.attrs["source"] = "NOAA/NCEI 0-2000 m OHC, downloaded"
            return out
        except Exception:
            pass
    yrs = np.array(sorted(_OHC_FALLBACK))
    vals = np.array([_OHC_FALLBACK[y] for y in yrs]) * 10.0
    full = np.arange(2005, 2026)
    out = pd.DataFrame({"year": full, "ohc": np.interp(full, yrs, vals),
                        "sigma": np.full(full.shape, 5.0)})
    out.attrs["source"] = _record("EMBEDDED FALLBACK (no network)")
    return out


def co2_pathway(years: np.ndarray, ssp: str = "ssp245") -> np.ndarray:
    """CO2 concentration (ppm) for `years`: shared history + the `ssp` future."""
    if ssp not in _CO2_FUTURE:
        raise NotImplementedError(f"unknown ssp {ssp!r}; have {tuple(_CO2_FUTURE)}")
    table = {**_CO2_HIST, **_CO2_FUTURE[ssp]}
    yrs = np.array(sorted(table), dtype=float)
    vals = np.array([table[int(y)] for y in yrs], dtype=float)
    return np.interp(years, yrs, vals)


def total_erf(years: np.ndarray, ssp: str = "ssp245") -> np.ndarray:
    """Total ERF (W m-2) over `years`: AR6 historical (<=2019), SSP future after."""
    if ssp not in _ERF_FUTURE:
        raise NotImplementedError(f"unknown ssp {ssp!r}; have {tuple(_ERF_FUTURE)}")
    erf = load_ar6_erf()
    hist_yr, hist_val = erf["year"].to_numpy(), erf["total"].to_numpy()
    fut = _ERF_FUTURE[ssp]
    fyrs = np.array(sorted(fut), dtype=float)
    fval = np.array([fut[int(y)] for y in fyrs], dtype=float)
    out = np.empty_like(years, dtype=float)
    hist_mask = years <= hist_yr.max()
    out[hist_mask] = np.interp(years[hist_mask], hist_yr, hist_val)
    out[~hist_mask] = np.interp(years[~hist_mask], fyrs, fval)
    return out


def rebaseline(years: np.ndarray, series: np.ndarray, ref=REF_PERIOD) -> np.ndarray:
    """Subtract the mean over the reference period so series are comparable."""
    m = (years >= ref[0]) & (years <= ref[1])
    return series - np.nanmean(series[m])
