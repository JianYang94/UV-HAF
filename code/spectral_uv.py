"""Spectral emulator utilities for reconstructing erythemally weighted UV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "data" / "derived" / "allsky_spectral_pca_emulator.npz"
BWF = PROJECT_ROOT / "data" / "derived" / "bwf_action_spectra_1nm.csv"
DAYS = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def interp_grid(source, source_lat, source_lon, target_lat, target_lon):
    """Bilinearly interpolate a field with periodic longitude."""
    order = np.argsort(source_lat)
    source_lat, source = source_lat[order], source[..., order, :]
    source_lon = np.mod(source_lon, 360)
    lon_order = np.argsort(source_lon)
    source_lon, source = source_lon[lon_order], source[..., lon_order]
    extended_lon = np.r_[source_lon, source_lon[0] + 360]
    extended = np.concatenate([source, source[..., :1]], axis=-1)
    wrapped_target_lon = np.mod(target_lon, 360)
    leading = source.shape[:-2]
    intermediate = np.empty((*leading, source.shape[-2], wrapped_target_lon.size))
    for lead in np.ndindex(leading):
        for i in range(source.shape[-2]):
            intermediate[lead + (i, slice(None))] = np.interp(
                wrapped_target_lon, extended_lon, extended[lead + (i, slice(None))]
            )
    result = np.empty((*leading, target_lat.size, target_lon.size))
    for lead in np.ndindex(leading):
        for j in range(wrapped_target_lon.size):
            result[lead + (slice(None), j)] = np.interp(
                target_lat, source_lat, intermediate[lead + (slice(None), j)]
            )
    return result


def features(frame):
    """Build the predictor matrix expected by the spectral emulator."""
    solar_zenith_angle = np.deg2rad(frame["sza"])
    mu = np.maximum(np.cos(solar_zenith_angle), 0.03)
    ozone = frame["ozone"] / 350.0
    aod = frame["aod"]
    ssa = frame["ssa"]
    alpha = frame["alpha"]
    return np.column_stack([
        mu, 1 / mu, ozone, aod, ssa, alpha / 1.25,
        frame["albedo"], frame["pressure"] / 1000.0,
        np.log1p(frame["cot"]), frame["reff"] / 20.0,
        frame["cloud_fraction"], frame["water_cm"] / 3.0,
        1 + 0.033 * np.cos(2 * np.pi * frame["day"] / 365.0),
        ozone / mu, aod / mu, aod * 0.55**alpha, aod * (1 - ssa),
        frame["cloud_fraction"] * np.log1p(frame["cot"]) / mu,
    ])


def predict_weighted(predictors, action_spectrum_column=1):
    """Predict irradiance and integrate it against a biological action spectrum."""
    model = np.load(MODEL, allow_pickle=False)
    standardized = (predictors - model["input_mean"]) / model["input_scale"]
    combinations = [
        tuple(map(int, text.split(","))) for text in model["combos"].astype(str)
    ]
    design = np.column_stack([
        np.ones(len(standardized)),
        *[np.prod(standardized[:, columns], axis=1) for columns in combinations],
    ])
    scores = design @ model["coefficients"]
    spectra = np.exp(model["mean_log_spectrum"] + scores @ model["pca_basis"])
    wavelength = model["wavelength"]
    action_spectrum = np.loadtxt(
        BWF, delimiter=",", skiprows=1, usecols=action_spectrum_column
    )
    curve = action_spectrum / action_spectrum[np.where(wavelength == 300)[0][0]]
    return np.trapezoid(spectra * curve, wavelength, axis=1)


def predict_erythema(predictors):
    """Predict irradiance weighted by the CIE erythema action spectrum."""
    return predict_weighted(predictors, 1)
