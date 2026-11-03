"""Augment the global spectral emulator with OMUVB-domain cloudy RT cases."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import train_allsky_spectral_pca_emulator as spectral
import validate_omuvb_2019_emulator as satellite
import calibrate_omuvb_bandpass as bandpass


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "spectral_pca_emulator"
SOURCE = ROOT / "omuvb_2019_validation" / "omuvb_2019_pixel_comparison.csv.gz"
OUT = ROOT / "spectral_pca_emulator_omuvb_targeted"


def choose_design(frame: pd.DataFrame, per_bin: int) -> pd.DataFrame:
    bins = pd.cut(frame.cloud_optical_thickness_550, [-np.inf, 1, 10, 30, np.inf],
                  labels=["<1", "1-10", "10-30", ">30"])
    pieces = []
    for label in bins.cat.categories:
        group = frame[bins.eq(label)].sort_values(
            ["sza_deg", "ozone_du", "cloud_optical_thickness_550", "latitude", "longitude"]
        )
        chosen = group.iloc[np.linspace(0, len(group) - 1, min(per_bin, len(group)), dtype=int)].copy()
        chosen["cot_stratum"] = str(label)
        pieces.append(chosen)
    chosen = pd.concat(pieces).drop_duplicates().reset_index(drop=True)
    order = np.arange(len(chosen))
    chosen["split"] = np.where(order % 4 == 0, "holdout", "train")
    chosen["sample_id"] = [f"OMI_{split.upper()}_{i:04d}" for i, split in enumerate(chosen.split)]
    chosen["angstrom_alpha"] = chosen.angstrom
    chosen["surface_pressure_hpa"] = chosen.pressure_hpa
    chosen["cloud_effective_radius_um"] = 10.0
    chosen["cloud_fraction"] = 1.0
    chosen["precipitable_water_cm"] = chosen.water_mm / 10
    return chosen[["sample_id", *spectral.PARAMETERS, "split", "cot_stratum"]]


def run_cloud(row: dict, executable: Path, data: Path, env: dict, cloud_dir: Path) -> dict:
    lines = spectral.common_input(data, row)
    cloud = cloud_dir / f"{row['sample_id']}.dat"
    cloud.write_text("# z_km LWC_g_m3 reff_um\n3 0 0\n2 0.1 10\n1 0.1 10\n", encoding="utf-8")
    reference = cloud.resolve().relative_to(Path.cwd().resolve()).as_posix()
    cloudy = lines[:-1] + [f"wc_file 1D {reference}",
                           f"wc_modify tau set {row['cloud_optical_thickness_550']:.8f}", "quiet"]
    wavelength, flux = spectral.execute(executable, env, cloudy)
    output = {"sample_id": row["sample_id"], **{key: row[key] for key in spectral.PARAMETERS}}
    output.update({f"E_{int(w):03d}": value for w, value in zip(wavelength, flux)})
    return output


def run_design(design: pd.DataFrame, output: Path, executable: Path, data: Path,
               env: dict, cloud_dir: Path, workers: int) -> pd.DataFrame:
    done = pd.read_csv(output) if output.exists() else pd.DataFrame()
    done_ids = set(done.sample_id) if len(done) else set()
    pending = [row._asdict() for row in design.itertuples(index=False) if row.sample_id not in done_ids]
    print(f"target cases: {len(done_ids)} complete, {len(pending)} pending", flush=True)
    buffer, start = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_cloud, row, executable, data, env, cloud_dir): row["sample_id"]
                   for row in pending}
        for count, future in enumerate(as_completed(futures), 1):
            buffer.append(future.result())
            if len(buffer) >= 16 or count == len(pending):
                pd.DataFrame(buffer).to_csv(output, mode="a", header=not output.exists(), index=False)
                buffer.clear()
            if count % 64 == 0 or count == len(pending):
                print(f"target uvspec {count}/{len(pending)} ({count/max(time.time()-start,1e-9):.2f}/s)", flush=True)
    return pd.read_csv(output)


def apply_bandpass(frame: pd.DataFrame) -> pd.DataFrame:
    transfer = np.load(ROOT / "omuvb_2019_validation" / "omuvb_bandpass_transfer.npz")
    x = bandpass.base_features(frame)
    z = (x - transfer["input_mean"]) / transfer["input_scale"]
    combos = [tuple(map(int, item.split(","))) for item in transfer["combos"].astype(str)]
    p = bandpass.poly2(z, combos)
    for index, wavelength in enumerate(transfer["wavelengths"].astype(int)):
        ratio = np.exp(p @ transfer["coefficients"][:, index])
        frame[f"bandpass_ratio_{wavelength}"] = ratio
        frame[f"bandpass_predicted_E_{wavelength}"] = frame[f"predicted_E_{wavelength}"] * ratio
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvspec", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cygwin-bin", required=True, type=Path)
    parser.add_argument("--per-cot-bin", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cloud_dir = OUT / "cloud_profiles"
    cloud_dir.mkdir(exist_ok=True)
    source = pd.read_csv(SOURCE)
    design = choose_design(source, args.per_cot_bin)
    design.to_csv(OUT / "omuvb_target_design.csv", index=False)
    env = os.environ.copy()
    env["PATH"] = str(args.cygwin_bin.resolve()) + os.pathsep + env.get("PATH", "")
    target = run_design(design, OUT / "omuvb_target_uvspec.csv", args.uvspec.resolve(),
                        args.data, env, cloud_dir, args.workers)
    target_train = target[target.sample_id.str.startswith("OMI_TRAIN")]
    target_holdout = target[target.sample_id.str.startswith("OMI_HOLDOUT")]
    base_train = pd.read_csv(BASE / "spectral_training_uvspec.csv")
    base_holdout = pd.read_csv(BASE / "spectral_holdout_uvspec.csv")
    training = pd.concat([base_train, target_train], ignore_index=True)
    holdout = pd.concat([base_holdout, target_holdout], ignore_index=True)
    spectral.OUT = OUT
    summary = spectral.fit_pca_proxy(training, holdout)
    satellite.MODEL = OUT / "allsky_spectral_pca_emulator.npz"
    targeted = satellite.predict(source)
    targeted = apply_bandpass(targeted)
    external = bandpass.external_metrics(targeted)
    external.to_csv(OUT / "targeted_omuvb_external_metrics.csv", index=False)
    targeted.to_csv(OUT / "targeted_omuvb_pixel_predictions.csv.gz", index=False, compression="gzip")
    summary["target_training_cases"] = len(target_train)
    summary["target_holdout_cases"] = len(target_holdout)
    summary["targeted_external_metrics"] = external.to_dict(orient="records")
    (OUT / "targeted_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target_training": len(target_train), "target_holdout": len(target_holdout),
                      "external": external.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
