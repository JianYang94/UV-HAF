"""Week-4 external-product comparison: OMUVB 2019 versus the spectral emulator.

OMUVB overpass irradiances are paired with co-located monthly MERRA-2 aerosol,
water-vapour and pressure fields.  OMI supplies ozone, surface albedo, effective
cloud optical thickness and overpass SZA.  This is an independent satellite-
product consistency test, not a comparison with direct ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations_with_replacement
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "spectral_pca_emulator" / "allsky_spectral_pca_emulator.npz"
MERRA = ROOT / "merra2_variable_optics_analysis" / "merra2_variable_optics_monthly_2013_2023.csv.gz"
OUT = ROOT / "omuvb_2019_validation"
WAVELENGTHS = (305, 310, 324, 380)


def read_field(group, name: str) -> np.ndarray:
    variable = group[name]
    data = np.asarray(variable[:])
    fill = float(getattr(variable, "FillValue", -1.2676506e30))
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
        data[(~np.isfinite(data)) | (data <= fill / 10)] = np.nan
    return data


def extract_granule(path: Path) -> pd.DataFrame:
    match = re.search(r"2019m(\d{2})(\d{2})", path.name)
    if not match:
        return pd.DataFrame()
    month, day = map(int, match.groups())
    date = pd.Timestamp(2019, month, day)
    with netCDF4.Dataset(path) as dataset:
        swath = dataset["HDFEOS"]["SWATHS"]["UVB"]
        data = swath["Data Fields"]
        geo = swath["Geolocation Fields"]
        values = {
            "longitude": read_field(geo, "Longitude"),
            "latitude": read_field(geo, "Latitude"),
            "sza_deg": read_field(geo, "SolarZenithAngle"),
            "terrain_m": read_field(geo, "TerrainHeight"),
            "ozone_du": read_field(data, "OMTO3ColumnAmountO3"),
            "surface_albedo": read_field(data, "SurfaceAlbedo"),
            "cloud_optical_thickness_550": read_field(data, "CloudOpticalThickness"),
            "quality": read_field(data, "OMUVBQuality"),
            "xtrack_quality": read_field(geo, "XTrackQualityFlags"),
        }
        for wavelength in WAVELENGTHS:
            values[f"observed_E_{wavelength}"] = read_field(data, f"OPIrradiance{wavelength}") / 1000
        values["observed_erythemal_daily_dose_J_m2"] = read_field(data, "ErythemalDailyDose")
        values["observed_clear_sky_erythemal_daily_dose_J_m2"] = read_field(data, "CSErythemalDailyDose")
        values["observed_overpass_erythemal_rate_W_m2"] = read_field(data, "OPErythemalDoseRate") / 1000

    lon, lat, sza = values["longitude"], values["latitude"], values["sza_deg"]
    quality = values["quality"].astype(np.int64)
    xtrack = values["xtrack_quality"].astype(np.int64)
    valid = (
        np.isfinite(lon) & np.isfinite(lat) & np.isfinite(sza)
        & (lon >= 73) & (lon <= 135) & (lat >= 18) & (lat <= 54)
        & (sza > 0) & (sza < 80)
        & ((quality & ((1 << 0) | (1 << 1) | (1 << 15))) == 0)
        & (xtrack == 0)
        & np.isfinite(values["ozone_du"]) & (values["ozone_du"] >= 150) & (values["ozone_du"] <= 550)
        & np.isfinite(values["surface_albedo"]) & (values["surface_albedo"] >= 0) & (values["surface_albedo"] <= 0.95)
        & np.isfinite(values["cloud_optical_thickness_550"])
        & (values["cloud_optical_thickness_550"] >= 0) & (values["cloud_optical_thickness_550"] <= 100)
    )
    for wavelength in WAVELENGTHS:
        observed = values[f"observed_E_{wavelength}"]
        valid &= np.isfinite(observed) & (observed > 1e-8) & (observed < 10)
    daily = values["observed_erythemal_daily_dose_J_m2"]
    clear_daily = values["observed_clear_sky_erythemal_daily_dose_J_m2"]
    valid &= np.isfinite(daily) & (daily > 0) & (daily < 2e5)
    valid &= np.isfinite(clear_daily) & (clear_daily > 0) & (clear_daily < 2e5)
    if not valid.any():
        return pd.DataFrame()
    frame = pd.DataFrame({key: value[valid] for key, value in values.items()})
    frame["date"] = date.strftime("%Y-%m-%d")
    frame["month"] = month
    frame["day_of_year"] = date.dayofyear
    frame["source_file"] = path.name
    return frame


def nearest_merra_key(values: np.ndarray, start: float, step: float) -> np.ndarray:
    return np.round((values - start) / step) * step + start


def pair_merra(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["year", "month", "latitude", "longitude", "aod550", "ssa550",
               "angstrom", "water_mm", "pressure_hpa"]
    merra = pd.read_csv(MERRA, usecols=columns)
    merra = merra[merra.year.eq(2019)].drop(columns="year")
    frame = frame.copy()
    frame["merra_latitude"] = nearest_merra_key(frame.latitude.to_numpy(float), -90.0, 0.5)
    frame["merra_longitude"] = nearest_merra_key(frame.longitude.to_numpy(float), -180.0, 0.625)
    paired = frame.merge(
        merra,
        left_on=["month", "merra_latitude", "merra_longitude"],
        right_on=["month", "latitude", "longitude"],
        how="inner", suffixes=("", "_merra"),
    )
    return paired.drop(columns=["latitude_merra", "longitude_merra"])


def physical_features(frame: pd.DataFrame) -> np.ndarray:
    mu = np.maximum(np.cos(np.radians(frame.sza_deg.to_numpy(float))), 0.03)
    aod = frame.aod550.to_numpy(float)
    ssa = frame.ssa550.to_numpy(float)
    alpha = frame.angstrom.to_numpy(float)
    ozone = frame.ozone_du.to_numpy(float) / 350
    cot = frame.cloud_optical_thickness_550.to_numpy(float)
    day = frame.day_of_year.to_numpy(float)
    # OMUVB COT is an effective homogeneous-pixel optical thickness; cf=1 is
    # therefore the closest representation in the independent-column emulator.
    cf = np.ones(len(frame))
    return np.column_stack([
        mu, 1 / mu, ozone, aod, ssa, alpha / 1.25,
        frame.surface_albedo.to_numpy(float), frame.pressure_hpa.to_numpy(float) / 1000,
        np.log1p(cot), np.full(len(frame), 0.5), cf, frame.water_mm.to_numpy(float) / 30,
        1 + 0.033 * np.cos(2 * np.pi * day / 365),
        ozone / mu, aod / mu, aod * 0.55**alpha, aod * (1 - ssa),
        cf * np.log1p(cot) / mu,
    ])


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    model = np.load(MODEL, allow_pickle=False)
    x = physical_features(frame)
    z = (x - model["input_mean"]) / model["input_scale"]
    combos = [tuple(map(int, item.split(","))) for item in model["combos"].astype(str)]
    poly = np.column_stack([np.ones(len(z)), *[np.prod(z[:, combo], axis=1) for combo in combos]])
    spectrum = np.exp(model["mean_log_spectrum"] + (poly @ model["coefficients"]) @ model["pca_basis"])
    result = frame.copy()
    grid = model["wavelength"].astype(int)
    for wavelength in WAVELENGTHS:
        index = int(np.where(grid == wavelength)[0][0])
        result[f"predicted_E_{wavelength}"] = spectrum[:, index]
    return result


def assign_region(frame: pd.DataFrame) -> pd.Series:
    lon, lat = frame.longitude, frame.latitude
    region = np.full(len(frame), "Other China", dtype=object)
    region[(lon >= 78) & (lon <= 103) & (lat >= 27) & (lat <= 37)] = "Tibetan Plateau"
    region[(lon >= 102) & (lon <= 108) & (lat >= 27) & (lat <= 33)] = "Sichuan Basin"
    region[(lon >= 112) & (lon <= 120) & (lat >= 34) & (lat <= 41)] = "North China Plain"
    region[(lon >= 118) & (lon <= 123) & (lat >= 28) & (lat <= 34)] = "Yangtze River Delta"
    region[(lon >= 110) & (lon <= 116) & (lat >= 21) & (lat <= 26)] = "Pearl River Delta"
    region[(lon >= 118) & (lon <= 135) & (lat >= 40) & (lat <= 54)] = "Northeast China"
    region[(lon >= 73) & (lon < 110) & (lat > 37) & (lat <= 50)] = "Northwest China"
    return pd.Series(region, index=frame.index)


def metrics(frame: pd.DataFrame, group_name: str, group_value: str) -> list[dict]:
    rows = []
    for wavelength in WAVELENGTHS:
        observed = frame[f"observed_E_{wavelength}"].to_numpy(float)
        predicted = frame[f"predicted_E_{wavelength}"].to_numpy(float)
        relative = (predicted / observed - 1) * 100
        abs_relative = np.abs(relative)
        rank_o = pd.Series(observed).rank().to_numpy()
        rank_p = pd.Series(predicted).rank().to_numpy()
        rows.append({
            "group": group_name, "value": group_value, "wavelength_nm": wavelength,
            "n": len(frame), "normalized_mean_bias_percent": float((predicted.sum() / observed.sum() - 1) * 100),
            "median_bias_percent": float(np.median(relative)),
            "median_absolute_error_percent": float(np.median(abs_relative)),
            "p95_absolute_error_percent": float(np.quantile(abs_relative, 0.95)),
            "spearman": float(np.corrcoef(rank_o, rank_p)[0, 1]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--granules", required=True, type=Path)
    parser.add_argument("--max-per-granule", type=int, default=4000,
                        help="Deterministic cap after QA to control output size; 0 means no cap")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    paths = sorted(args.granules.glob("*.he5"))
    for number, path in enumerate(paths, 1):
        piece = extract_granule(path)
        if args.max_per_granule and len(piece) > args.max_per_granule:
            piece = piece.iloc[np.linspace(0, len(piece) - 1, args.max_per_granule, dtype=int)]
        if len(piece):
            pieces.append(piece)
        if number % 20 == 0 or number == len(paths):
            print(f"extracted {number}/{len(paths)} granules", flush=True)
    if not pieces:
        raise RuntimeError("No valid OMUVB pixels found")
    extracted = pd.concat(pieces, ignore_index=True)
    paired = pair_merra(extracted)
    if not len(paired):
        raise RuntimeError("No OMUVB pixels paired with MERRA-2")
    result = predict(paired)
    result["region"] = assign_region(result)
    result["sza_bin"] = pd.cut(result.sza_deg, [-np.inf, 60, 70, 80], labels=["<60", "60-70", "70-80"])
    result["cot_bin"] = pd.cut(result.cloud_optical_thickness_550, [-np.inf, 1, 10, 30, np.inf],
                               labels=["<1", "1-10", "10-30", ">30"])
    metric_rows = metrics(result, "all", "all")
    for group in ["month", "region", "sza_bin", "cot_bin"]:
        for value, subset in result.groupby(group, observed=True):
            if len(subset) >= 30:
                metric_rows.extend(metrics(subset, group, str(value)))
    metric_frame = pd.DataFrame(metric_rows)
    result.to_csv(OUT / "omuvb_2019_pixel_comparison.csv.gz", index=False, compression="gzip")
    metric_frame.to_csv(OUT / "omuvb_2019_validation_metrics.csv", index=False)
    overall = metric_frame[(metric_frame.group == "all") & (metric_frame.value == "all")]
    summary = {
        "granules": len(paths), "qa_extracted_pixels": len(extracted), "merra_paired_pixels": len(result),
        "scope": "China, OMI overpass, SZA < 80 deg, fatal/suspicious/missing and row-anomaly pixels excluded",
        "interpretation": "independent satellite-product consistency test; OMUVB is not direct ground truth",
        "cloud_mapping": "OMUVB effective COT mapped to cloud_fraction=1 and reff=10 um",
        "overall": overall.to_dict(orient="records"),
    }
    (OUT / "omuvb_2019_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
