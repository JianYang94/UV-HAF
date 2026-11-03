"""Train a physical OMUVB triangular-bandpass transfer layer and revalidate.

The base PCA emulator predicts 1-nm monochromatic spectra. OMUVB reports four
0.55-nm-FWHM triangular-slit quantities centered at x.1 nm. Direct 0.1-nm
uvspec runs are used to learn only that transfer ratio, not an empirical
satellite bias correction.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

from direct_omuvb_uvspec_check import run_case


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "omuvb_2019_validation"
SOURCE = VALIDATION / "omuvb_2019_pixel_comparison.csv.gz"
WAVELENGTHS = (305, 310, 324, 380)


def choose_cases(frame: pd.DataFrame, per_cot_bin: int) -> pd.DataFrame:
    bins = pd.cut(frame.cloud_optical_thickness_550, [-np.inf, 1, 10, 30, np.inf],
                  labels=["<1", "1-10", "10-30", ">30"])
    pieces = []
    for label in bins.cat.categories:
        group = frame[bins.eq(label)].sort_values(
            ["sza_deg", "ozone_du", "cloud_optical_thickness_550", "latitude", "longitude"]
        )
        count = min(per_cot_bin, len(group))
        pieces.append(group.iloc[np.linspace(0, len(group) - 1, count, dtype=int)])
    return pd.concat(pieces).drop_duplicates().reset_index().rename(columns={"index": "source_index"})


def base_features(frame: pd.DataFrame) -> np.ndarray:
    mu = np.maximum(np.cos(np.radians(frame.sza_deg.to_numpy(float))), 0.03)
    day = frame.day_of_year.to_numpy(float)
    ozone = frame.ozone_du.to_numpy(float) / 350
    return np.column_stack([
        mu, 1 / mu, ozone, ozone / mu,
        np.log1p(frame.cloud_optical_thickness_550.to_numpy(float)),
        frame.aod550.to_numpy(float), frame.ssa550.to_numpy(float),
        frame.angstrom.to_numpy(float) / 1.25,
        frame.surface_albedo.to_numpy(float), frame.pressure_hpa.to_numpy(float) / 1000,
        frame.water_mm.to_numpy(float) / 30,
        np.sin(2 * np.pi * day / 365), np.cos(2 * np.pi * day / 365),
    ])


def poly2(z: np.ndarray, combos: list[tuple[int, ...]]) -> np.ndarray:
    return np.column_stack([np.ones(len(z)), *[np.prod(z[:, combo], axis=1) for combo in combos]])


def diagnostic(observed: np.ndarray, predicted: np.ndarray) -> dict:
    relative = (predicted / observed - 1) * 100
    return {
        "n": len(observed),
        "normalized_mean_bias_percent": float((predicted.sum() / observed.sum() - 1) * 100),
        "median_bias_percent": float(np.median(relative)),
        "median_absolute_error_percent": float(np.median(np.abs(relative))),
        "p95_absolute_error_percent": float(np.quantile(np.abs(relative), 0.95)),
    }


def train_transfer(training: pd.DataFrame, direct: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = training.merge(direct, on="source_index", how="inner", suffixes=("", "_direct"))
    x = base_features(joined)
    full_x = base_features(full)
    holdout = np.arange(len(joined)) % 4 == 0
    fit = ~holdout
    mean, scale = x[fit].mean(axis=0), x[fit].std(axis=0)
    scale[scale == 0] = 1
    combos = [c for degree in (1, 2) for c in combinations_with_replacement(range(x.shape[1]), degree)]
    p = poly2((x - mean) / scale, combos)
    p_full = poly2((full_x - mean) / scale, combos)
    summary = {"direct_cases": len(joined), "holdout_cases": int(holdout.sum()), "wavelengths": {}}
    coefficients = []
    for wavelength in WAVELENGTHS:
        raw = joined[f"uvspec_{wavelength}"].to_numpy(float)
        band = joined[f"uvspec_tri_{wavelength}.1"].to_numpy(float)
        target = np.log(band / raw)
        best = None
        for ridge in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
            penalty = np.eye(p.shape[1]) * ridge
            penalty[0, 0] = 0
            coef = np.linalg.solve(p[fit].T @ p[fit] + penalty, p[fit].T @ target[fit])
            pred = np.exp(p[holdout] @ coef)
            error = np.median(np.abs((pred / np.exp(target[holdout]) - 1) * 100))
            if best is None or error < best[0]:
                best = error, ridge, coef
        _, ridge, _ = best
        penalty = np.eye(p.shape[1]) * ridge
        penalty[0, 0] = 0
        coef = np.linalg.solve(p.T @ p + penalty, p.T @ target)
        coefficients.append(coef)
        predicted_ratio_holdout = np.exp(p[holdout] @ coef)
        actual_ratio_holdout = np.exp(target[holdout])
        summary["wavelengths"][str(wavelength)] = {
            "ridge": ridge,
            "bandpass_ratio_range": [float(np.min(np.exp(target))), float(np.max(np.exp(target)))],
            "transfer_holdout": diagnostic(actual_ratio_holdout, predicted_ratio_holdout),
            "direct_bandpass_vs_omuvb_all": diagnostic(
                joined[f"observed_E_{wavelength}"].to_numpy(float), band),
        }
        full[f"bandpass_ratio_{wavelength}"] = np.exp(p_full @ coef)
        full[f"bandpass_predicted_E_{wavelength}"] = (
            full[f"predicted_E_{wavelength}"] * full[f"bandpass_ratio_{wavelength}"]
        )
    np.savez_compressed(
        VALIDATION / "omuvb_bandpass_transfer.npz", input_mean=mean, input_scale=scale,
        combos=np.array([",".join(map(str, c)) for c in combos]), coefficients=np.column_stack(coefficients),
        wavelengths=np.array(WAVELENGTHS),
    )
    return full, summary


def external_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = frame.copy()
    frame["cot_bin"] = pd.cut(frame.cloud_optical_thickness_550, [-np.inf, 1, 10, 30, np.inf],
                              labels=["<1", "1-10", "10-30", ">30"])
    groups = [("all", "all", frame)] + [("cot_bin", str(v), g) for v, g in frame.groupby("cot_bin", observed=True)]
    for group, value, subset in groups:
        for wavelength in WAVELENGTHS:
            item = diagnostic(subset[f"observed_E_{wavelength}"].to_numpy(float),
                              subset[f"bandpass_predicted_E_{wavelength}"].to_numpy(float))
            rows.append({"group": group, "value": value, "wavelength_nm": wavelength, **item})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvspec", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cygwin-bin", required=True, type=Path)
    parser.add_argument("--per-cot-bin", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    full = pd.read_csv(SOURCE)
    selected = choose_cases(full, args.per_cot_bin)
    env = os.environ.copy()
    env["PATH"] = str(args.cygwin_bin.resolve()) + os.pathsep + env.get("PATH", "")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, row, number, args.uvspec.resolve(), args.data, env, "atlas3"): number
            for number, row in enumerate(selected.itertuples(index=False), 1)
        }
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 32 == 0 or count == len(futures):
                print(f"direct bandpass cases {count}/{len(futures)}", flush=True)
    direct = pd.DataFrame(rows)
    direct.to_csv(VALIDATION / "omuvb_bandpass_direct_cases.csv", index=False)
    adjusted, summary = train_transfer(selected, direct, full)
    metrics = external_metrics(adjusted)
    metrics.to_csv(VALIDATION / "omuvb_bandpass_adjusted_metrics.csv", index=False)
    adjusted.to_csv(VALIDATION / "omuvb_2019_bandpass_adjusted_pixels.csv.gz", index=False, compression="gzip")
    summary["adjusted_external_metrics"] = metrics.to_dict(orient="records")
    (VALIDATION / "omuvb_bandpass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
