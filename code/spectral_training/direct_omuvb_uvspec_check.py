"""Small direct-uvspec audit of the OMUVB comparison, including x.1 nm bands."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "omuvb_2019_validation" / "omuvb_2019_pixel_comparison.csv.gz"
OUT = ROOT / "omuvb_2019_validation"


def triangular_band(wave: np.ndarray, flux: np.ndarray, center: float, fwhm: float = 0.55) -> float:
    """Convolve with the triangular OMUVB slit function (FWHM 0.55 nm)."""
    weight = np.maximum(0.0, 1.0 - np.abs(wave - center) / fwhm)
    return float(np.trapezoid(flux * weight, wave) / np.trapezoid(weight, wave))


def cie_erythema(wave: np.ndarray) -> np.ndarray:
    weight = np.ones_like(wave, dtype=float)
    middle = (wave > 298) & (wave <= 328)
    longwave = wave > 328
    weight[middle] = 10 ** (0.094 * (298 - wave[middle]))
    weight[longwave] = 10 ** (0.015 * (139 - wave[longwave]))
    return weight


def run_case(row, number: int, executable: Path, data: Path, env: dict, solar: str) -> dict:
    alpha = float(row.angstrom)
    beta = float(row.aod550) * 0.55**alpha
    cloud = OUT / "direct_check_clouds" / f"cloud_{number:03d}.dat"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("# z_km LWC_g_m3 reff_um\n3 0 0\n2 0.1 10\n1 0.1 10\n", encoding="utf-8")
    try:
        cloud_reference = cloud.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        cloud_reference = cloud.resolve().as_posix()
    common = [
        f"data_files_path {data.as_posix()}", f"atmosphere_file {data.as_posix()}/atmmod/afglus.dat",
        f"source solar {data.as_posix()}/solar_flux/{solar}", "wavelength 280 400",
        "mol_abs_param crs", "spline 280 400 0.1", f"sza {row.sza_deg:.8f}",
        f"day_of_year {int(row.day_of_year)}", f"mol_modify O3 {row.ozone_du:.8f} DU",
        f"mol_modify H2O {row.water_mm:.8f} MM", f"pressure {row.pressure_hpa:.8f}",
        f"albedo {row.surface_albedo:.8f}", "aerosol_default",
        f"aerosol_angstrom {alpha:.8f} {beta:.10f}", f"aerosol_modify ssa set {row.ssa550:.8f}",
        "rte_solver disort", "number_of_streams 8", "zout sur", "output_user lambda edir edn eup",
    ]
    if row.cloud_optical_thickness_550 > 0:
        common += [f"wc_file 1D {cloud_reference}",
                   f"wc_modify tau set {row.cloud_optical_thickness_550:.8f}"]
    process = subprocess.run([str(executable)], input="\n".join(common + ["quiet"]) + "\n",
                             text=True, capture_output=True, env=env)
    if process.returncode:
        raise RuntimeError(process.stderr)
    parsed = []
    for line in process.stdout.splitlines():
        try:
            wavelength, direct, diffuse, upward = map(float, line.split()[:4])
            parsed.append((wavelength, (direct + diffuse) / 1000))
        except (ValueError, IndexError):
            pass
    wave = np.array([x[0] for x in parsed])
    flux = np.array([x[1] for x in parsed])
    if not len(wave):
        raise RuntimeError(f"No numeric uvspec output; stdout={process.stdout[:1000]!r}; stderr={process.stderr[:1000]!r}")
    output = {"sample": number,
              "source_index": getattr(row, "source_index", getattr(row, "index", number)),
              "sza_deg": row.sza_deg,
              "cot": row.cloud_optical_thickness_550}
    for wavelength in [305, 305.1, 310, 310.1, 324, 324.1, 380, 380.1]:
        index = int(np.argmin(np.abs(wave - wavelength)))
        output[f"uvspec_{wavelength:g}"] = flux[index]
    for wavelength in [305, 310, 324, 380]:
        output[f"uvspec_tri_{wavelength}.1"] = triangular_band(wave, flux, wavelength + 0.1)
    output["uvspec_erythema_0p1_W_m2"] = float(np.trapezoid(flux * cie_erythema(wave), wave))
    integer = np.isclose(wave, np.round(wave), atol=1e-6)
    output["uvspec_erythema_1nm_W_m2"] = float(
        np.trapezoid(flux[integer] * cie_erythema(wave[integer]), wave[integer])
    )
    if hasattr(row, "observed_overpass_erythemal_rate_W_m2"):
        output["omi_erythema_W_m2"] = row.observed_overpass_erythemal_rate_W_m2
    for wavelength in [305, 310, 324, 380]:
        output[f"omi_{wavelength}"] = getattr(row, f"observed_E_{wavelength}")
        output[f"proxy_{wavelength}"] = getattr(row, f"predicted_E_{wavelength}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvspec", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cygwin-bin", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--solar", default="atlas_plus_modtran")
    args = parser.parse_args()
    frame = pd.read_csv(SOURCE)
    frame = frame.sort_values(["sza_deg", "cloud_optical_thickness_550", "ozone_du"]).reset_index()
    selected = frame.iloc[np.linspace(0, len(frame) - 1, args.samples, dtype=int)]
    env = os.environ.copy()
    env["PATH"] = str(args.cygwin_bin.resolve()) + os.pathsep + env.get("PATH", "")
    rows = []
    for number, row in enumerate(selected.itertuples(index=False), 1):
        rows.append(run_case(row, number, args.uvspec.resolve(), args.data, env, args.solar))
        print(f"direct uvspec {number}/{len(selected)}", flush=True)
    result = pd.DataFrame(rows)
    summary_rows = []
    for wavelength in [305, 310, 324, 380]:
        observed = result[f"omi_{wavelength}"].to_numpy()
        for source in [f"proxy_{wavelength}", f"uvspec_{wavelength}", f"uvspec_{wavelength}.1",
                       f"uvspec_tri_{wavelength}.1"]:
            predicted = result[source].to_numpy()
            relative = (predicted / observed - 1) * 100
            summary_rows.append({"wavelength_nm": wavelength, "source": source, "n": len(result),
                                 "median_bias_percent": np.median(relative),
                                 "median_absolute_error_percent": np.median(np.abs(relative)),
                                 "normalized_mean_bias_percent": (predicted.sum() / observed.sum() - 1) * 100})
    tag = args.solar.replace(".", "_")
    result.to_csv(OUT / f"direct_uvspec_24_pixel_audit_{tag}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / f"direct_uvspec_24_pixel_audit_summary_{tag}.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
