"""Week-3 all-sky spectral PCA emulator with independent Sobol holdout.

Each parameter row is evaluated as paired clear/cloudy 1-D columns and mixed
by cloud fraction.  Full 280-400 nm spectra are checkpointed before PCA and
proxy fitting, so the radiative-transfer work is restart-safe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spectral_pca_emulator"
DESIGN = ROOT / "data" / "sobol_parameters_10240.csv"
BWF = ROOT / "data" / "bwf_action_spectra_1nm.csv"
PARAMETERS = [
    "sza_deg", "ozone_du", "aod550", "ssa550", "angstrom_alpha",
    "surface_albedo", "surface_pressure_hpa", "cloud_optical_thickness_550",
    "cloud_effective_radius_um", "cloud_fraction", "precipitable_water_cm",
    "day_of_year",
]
SATELLITE_WAVELENGTHS = [305, 310, 324, 380]
ENDPOINTS = {
    "erythema": "cie_erythema_energy_norm298",
    "vitamin_d": "vitamin_d_young2021_blue_shift5",
    "plant_fc": "plant_flint_caldwell_energy_norm300",
    "plant_cg_alt": "plant_caldwell_green_energy_norm300",
    "aquatic_cullen": "aquatic_cullen_phaeodactylum_per_W_m2",
    "aquatic_neale_alt": "aquatic_neale_antarctic_raw_per_mW_m2",
}


def common_input(data: Path, row: dict) -> list[str]:
    alpha = float(row["angstrom_alpha"])
    beta = float(row["aod550"]) * 0.55**alpha
    d = data.as_posix()
    return [
        f"data_files_path {d}", f"atmosphere_file {d}/atmmod/afglus.dat",
        f"source solar {d}/solar_flux/atlas_plus_modtran", "wavelength 280 400",
        "mol_abs_param crs", "spline 280 400 1.0", f"sza {row['sza_deg']:.8f}",
        f"day_of_year {int(round(row['day_of_year']))}",
        f"mol_modify O3 {row['ozone_du']:.8f} DU",
        f"mol_modify H2O {float(row['precipitable_water_cm']) * 10:.8f} MM",
        f"pressure {row['surface_pressure_hpa']:.8f}",
        f"albedo {row['surface_albedo']:.8f}", "aerosol_default",
        f"aerosol_angstrom {alpha:.8f} {beta:.10f}",
        f"aerosol_modify ssa set {row['ssa550']:.8f}",
        "rte_solver disort", "number_of_streams 8", "zout sur",
        "output_user lambda edir edn eup", "quiet",
    ]


def execute(executable: Path, env: dict, lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    process = subprocess.run(
        [str(executable)], input="\n".join(lines) + "\n", text=True,
        capture_output=True, env=env,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"uvspec returned {process.returncode}")
    rows = []
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append(tuple(float(value) for value in parts[:4]))
        except ValueError:
            continue
    wavelength = np.array([row[0] for row in rows])
    global_down = np.array([(row[1] + row[2]) / 1000 for row in rows])
    if len(wavelength) != 121 or not np.array_equal(wavelength, np.arange(280, 401)):
        raise RuntimeError(
            f"Unexpected wavelength grid ({len(wavelength)} points); "
            f"stdout={process.stdout[:500]!r}; stderr={process.stderr[:500]!r}"
        )
    if not np.isfinite(global_down).all() or (global_down < 0).any():
        raise RuntimeError("Invalid spectrum")
    return wavelength, global_down


def run_pair(executable: Path, data: Path, env: dict, row: dict, cloud_dir: Path) -> dict:
    lines = common_input(data, row)
    wavelength, clear = execute(executable, env, lines)
    cloud_file = cloud_dir / f"{row['sample_id']}.dat"
    reff = float(row["cloud_effective_radius_um"])
    cloud_file.write_text(
        f"# z_km  LWC_g_m3  reff_um\n3.0 0.0 0.0\n2.0 0.1 {reff:.8f}\n1.0 0.1 {reff:.8f}\n",
        encoding="utf-8",
    )
    try:
        cloud_reference = cloud_file.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        cloud_reference = cloud_file.resolve().as_posix()
    cloudy_lines = lines[:-1] + [
        f"wc_file 1D {cloud_reference}",
        f"wc_modify tau set {row['cloud_optical_thickness_550']:.8f}", "quiet",
    ]
    cloudy_wavelength, cloudy = execute(executable, env, cloudy_lines)
    if not np.array_equal(wavelength, cloudy_wavelength):
        raise RuntimeError("Clear/cloudy wavelength mismatch")
    fraction = float(row["cloud_fraction"])
    allsky = (1 - fraction) * clear + fraction * cloudy
    output = {"sample_id": row["sample_id"], **{key: row[key] for key in PARAMETERS}}
    output.update({f"E_{int(w):03d}": value for w, value in zip(wavelength, allsky)})
    return output


def run_design(frame: pd.DataFrame, path: Path, executable: Path, data: Path,
               env: dict, workers: int, cloud_dir: Path) -> pd.DataFrame:
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done_ids = set(done.sample_id.astype(str)) if len(done) else set()
    pending = [row._asdict() for row in frame.itertuples(index=False)
               if str(row.sample_id) not in done_ids]
    print(f"{path.name}: {len(done_ids)} complete, {len(pending)} pending", flush=True)
    start, buffer = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_pair, executable, data, env, row, cloud_dir): row["sample_id"]
                   for row in pending}
        for count, future in enumerate(as_completed(futures), 1):
            case_id = futures[future]
            try:
                buffer.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed {case_id}: {exc}") from exc
            if len(buffer) >= 16 or count == len(pending):
                pd.DataFrame(buffer).to_csv(path, mode="a", header=not path.exists(), index=False)
                buffer.clear()
            if count % 128 == 0 or count == len(pending):
                rate = count / max(time.time() - start, 1e-9)
                print(f"  completed {count}/{len(pending)} pairs ({rate:.2f} pairs/s)", flush=True)
    result = pd.read_csv(path)
    if result.sample_id.duplicated().any():
        result = result.drop_duplicates("sample_id", keep="last")
        result.to_csv(path, index=False)
    return result


def physical_features(frame: pd.DataFrame) -> np.ndarray:
    sza = np.radians(frame.sza_deg.to_numpy(float))
    mu = np.maximum(np.cos(sza), 0.03)
    aod = frame.aod550.to_numpy(float)
    ssa = frame.ssa550.to_numpy(float)
    alpha = frame.angstrom_alpha.to_numpy(float)
    ozone = frame.ozone_du.to_numpy(float) / 350
    cot = frame.cloud_optical_thickness_550.to_numpy(float)
    cf = frame.cloud_fraction.to_numpy(float)
    day = frame.day_of_year.to_numpy(float)
    return np.column_stack([
        mu, 1 / mu, ozone, aod, ssa, alpha / 1.25,
        frame.surface_albedo.to_numpy(float), frame.surface_pressure_hpa.to_numpy(float) / 1000,
        np.log1p(cot), frame.cloud_effective_radius_um.to_numpy(float) / 20,
        cf, frame.precipitable_water_cm.to_numpy(float) / 3,
        1 + 0.033 * np.cos(2 * np.pi * day / 365),
        ozone / mu, aod / mu, aod * 0.55**alpha, aod * (1 - ssa),
        cf * np.log1p(cot) / mu,
    ])


def poly2(z: np.ndarray, combos: list[tuple[int, ...]]) -> np.ndarray:
    return np.column_stack([np.ones(len(z)), *[np.prod(z[:, c], axis=1) for c in combos]])


def dose_metrics(spectra: np.ndarray, wavelength: np.ndarray) -> tuple[list[str], np.ndarray]:
    bwf = pd.read_csv(BWF)
    at_300 = int(np.where(wavelength == 300)[0][0])
    names = ["uv_280_400_W_m2", "uvb_280_315_W_m2"]
    outputs = [np.trapezoid(spectra, wavelength, axis=1)]
    mask = wavelength <= 315
    outputs.append(np.trapezoid(spectra[:, mask], wavelength[mask], axis=1))
    for endpoint, column in ENDPOINTS.items():
        curve = bwf[column].to_numpy(float).copy()
        if endpoint == "aquatic_neale_alt":
            curve *= 1000
        curve = np.nan_to_num(curve / curve[at_300], nan=0.0)
        names.append(f"dose_{endpoint}")
        outputs.append(np.trapezoid(spectra * curve, wavelength, axis=1))
    return names, np.column_stack(outputs)


def relative_diagnostics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    relative = (predicted / actual - 1) * 100
    return {
        "bias_percent": float(np.mean(relative)),
        "rmse_percent": float(np.sqrt(np.mean(relative**2))),
        "median_absolute_error_percent": float(np.median(np.abs(relative))),
        "p95_absolute_error_percent": float(np.quantile(np.abs(relative), 0.95)),
        "max_absolute_error_percent": float(np.max(np.abs(relative))),
    }


def fit_pca_proxy(training: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    wavelength = np.arange(280, 401, dtype=float)
    spectral_columns = [f"E_{int(w):03d}" for w in wavelength]
    train_spectrum = training[spectral_columns].to_numpy(float)
    holdout_spectrum = holdout[spectral_columns].to_numpy(float)
    floor = 1e-10
    train_log = np.log(np.maximum(train_spectrum, floor))
    mean_log = train_log.mean(axis=0)
    _, singular, vt = np.linalg.svd(train_log - mean_log, full_matrices=False)
    explained = singular**2 / np.sum(singular**2)

    pca_diagnostics = {}
    chosen = None
    for components in range(1, 31):
        basis = vt[:components]
        projected = (np.log(np.maximum(holdout_spectrum, floor)) - mean_log) @ basis.T
        reconstructed = np.exp(mean_log + projected @ basis)
        names, actual_doses = dose_metrics(holdout_spectrum, wavelength)
        _, reconstructed_doses = dose_metrics(reconstructed, wavelength)
        dose_error = max(relative_diagnostics(actual_doses[:, i], reconstructed_doses[:, i])["p95_absolute_error_percent"]
                         for i in range(len(names)))
        wavelengths_error = max(
            relative_diagnostics(holdout_spectrum[:, int(w - 280)], reconstructed[:, int(w - 280)])["p95_absolute_error_percent"]
            for w in SATELLITE_WAVELENGTHS
        )
        pca_diagnostics[str(components)] = {
            "cumulative_explained_variance": float(explained[:components].sum()),
            "maximum_p95_dose_error_percent": dose_error,
            "maximum_p95_satellite_wavelength_error_percent": wavelengths_error,
        }
        if chosen is None and dose_error < 0.5 and wavelengths_error < 1.0:
            chosen = components
    if chosen is None:
        chosen = 30
    basis = vt[:chosen]
    scores = (train_log - mean_log) @ basis.T

    x = physical_features(training)
    x_holdout = physical_features(holdout)
    rng = np.random.default_rng(20261013)
    indices = rng.permutation(len(training))
    tune_size = max(256, len(training) // 5)
    tune_idx, fit_idx = indices[:tune_size], indices[tune_size:]
    mean_x, scale_x = x[fit_idx].mean(axis=0), x[fit_idx].std(axis=0)
    scale_x[scale_x == 0] = 1
    combos = [c for degree in (1, 2) for c in combinations_with_replacement(range(x.shape[1]), degree)]
    p_fit = poly2((x[fit_idx] - mean_x) / scale_x, combos)
    p_tune = poly2((x[tune_idx] - mean_x) / scale_x, combos)
    p_holdout = poly2((x_holdout - mean_x) / scale_x, combos)
    best = None
    for ridge in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        penalty = np.eye(p_fit.shape[1]) * ridge
        penalty[0, 0] = 0
        coef = np.linalg.solve(p_fit.T @ p_fit + penalty, p_fit.T @ scores[fit_idx])
        tune_spectrum = np.exp(mean_log + (p_tune @ coef) @ basis)
        _, actual = dose_metrics(train_spectrum[tune_idx], wavelength)
        _, predicted = dose_metrics(tune_spectrum, wavelength)
        score = float(np.mean([relative_diagnostics(actual[:, i], predicted[:, i])["rmse_percent"]
                               for i in range(actual.shape[1])]))
        if best is None or score < best[0]:
            best = score, ridge
    _, ridge = best
    penalty = np.eye(p_fit.shape[1]) * ridge
    penalty[0, 0] = 0
    p_all = poly2((x - mean_x) / scale_x, combos)
    coef = np.linalg.solve(p_all.T @ p_all + penalty, p_all.T @ scores)
    predicted_spectrum = np.exp(mean_log + (p_holdout @ coef) @ basis)

    wavelength_diagnostics = {
        str(w): relative_diagnostics(holdout_spectrum[:, w - 280], predicted_spectrum[:, w - 280])
        for w in SATELLITE_WAVELENGTHS
    }
    dose_names, actual_dose = dose_metrics(holdout_spectrum, wavelength)
    _, predicted_dose = dose_metrics(predicted_spectrum, wavelength)
    dose_diagnostics = {
        name: relative_diagnostics(actual_dose[:, i], predicted_dose[:, i])
        for i, name in enumerate(dose_names)
    }
    validation = holdout[["sample_id", *PARAMETERS]].copy()
    for w in SATELLITE_WAVELENGTHS:
        validation[f"actual_E_{w}"] = holdout_spectrum[:, w - 280]
        validation[f"pred_E_{w}"] = predicted_spectrum[:, w - 280]
    for i, name in enumerate(dose_names):
        validation[f"actual_{name}"] = actual_dose[:, i]
        validation[f"pred_{name}"] = predicted_dose[:, i]
    validation.to_csv(OUT / "spectral_proxy_holdout_predictions.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        OUT / "allsky_spectral_pca_emulator.npz",
        wavelength=wavelength, mean_log_spectrum=mean_log, pca_basis=basis,
        input_mean=mean_x, input_scale=scale_x,
        combos=np.array([",".join(map(str, c)) for c in combos]), coefficients=coef,
        parameters=np.array(PARAMETERS), floor=np.array([floor]),
    )
    summary = {
        "training_cases": len(training), "independent_sobol_holdout_cases": len(holdout),
        "spectrum": "all-sky global downward irradiance, 280-400 nm at 1 nm",
        "cloud_method": "independent-column clear/cloudy mixture by cloud fraction",
        "pca": {"chosen_components": chosen,
                "cumulative_explained_variance": float(explained[:chosen].sum()),
                "component_diagnostics": pca_diagnostics},
        "proxy": {"physical_feature_count": x.shape[1],
                  "polynomial_feature_count": p_all.shape[1], "ridge": ridge},
        "holdout_satellite_wavelength_diagnostics": wavelength_diagnostics,
        "holdout_dose_diagnostics": dose_diagnostics,
        "gates": {
            "wavelength_median_absolute_error_lt_15pct": all(v["median_absolute_error_percent"] < 15 for v in wavelength_diagnostics.values()),
            "dose_median_absolute_error_lt_15pct": all(v["median_absolute_error_percent"] < 15 for v in dose_diagnostics.values()),
        },
    }
    (OUT / "spectral_pca_emulator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvspec", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cygwin-bin", required=True, type=Path)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-holdout", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cloud_dir = OUT / "cloud_profiles"
    cloud_dir.mkdir(exist_ok=True)
    design = pd.read_csv(DESIGN)
    training_design = design[design.split.eq("train")].head(args.n_train).drop(columns="split")
    holdout_design = design[design.split.eq("holdout")].head(args.n_holdout).drop(columns="split")
    training_design.to_csv(OUT / "spectral_training_design.csv", index=False)
    holdout_design.to_csv(OUT / "spectral_holdout_design.csv", index=False)
    env = os.environ.copy()
    env["PATH"] = str(args.cygwin_bin.resolve()) + os.pathsep + env.get("PATH", "")
    training = run_design(
        training_design, OUT / "spectral_training_uvspec.csv", args.uvspec.resolve(),
        args.data, env, args.workers, cloud_dir,
    )
    holdout = run_design(
        holdout_design, OUT / "spectral_holdout_uvspec.csv", args.uvspec.resolve(),
        args.data, env, args.workers, cloud_dir,
    )
    summary = fit_pca_proxy(training, holdout)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
