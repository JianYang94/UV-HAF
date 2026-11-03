"""Generate week-1/2 UV-HAF phase-0 tables, Sobol samples, and diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SOURCES = DATA / "sources"


NEALE_TEXT = """
285 0.005884451 286 0.004062648 287 0.002677548 288 0.00197644 289 0.001498352 290 0.001204038 291 0.000992771 292 0.000860736 293 0.000770988 294 0.000700626 295 0.000651776 296 0.000604756 297 0.000566807 298 0.000529635 299 0.000490644 300 0.000454373 301 0.000424517 302 0.000397003 303 0.000372411 304 0.000351546 305 0.000331987 306 0.00031573 307 0.000302785 308 0.000288171 309 0.000273054 310 0.0002575 311 0.000242339 312 0.000230505 313 0.000220775 314 0.00021312 315 0.000203323 316 0.000193486 317 0.000183979 318 0.000174847 319 0.000165648 320 0.000157207 321 0.000149623 322 0.000142818 323 0.000138364 324 0.000134975 325 0.000130971 326 0.000125806 327 0.000121236 328 0.000116946 329 0.000113388 330 0.000110131 331 0.000106231 332 0.000102182 333 9.81651E-05 334 9.43668E-05 335 9.07972E-05 336 8.70311E-05 337 8.39905E-05 338 8.06621E-05 339 7.77808E-05 340 7.51841E-05 341 7.28929E-05 342 7.04809E-05 343 6.82274E-05 344 6.5904E-05 345 6.34755E-05 346 6.14428E-05 347 5.95661E-05 348 5.79094E-05 349 5.57904E-05 350 5.37826E-05 351 5.16283E-05 352 4.95207E-05 353 4.74959E-05 354 4.5602E-05 355 4.35307E-05 356 4.1496E-05 357 3.95426E-05 358 3.75827E-05 359 3.57518E-05 360 3.40196E-05 361 3.22549E-05 362 3.0413E-05 363 2.83748E-05 364 2.64465E-05 365 2.47222E-05 366 2.3041E-05 367 2.13668E-05 368 1.94721E-05 369 1.76113E-05 370 1.60244E-05 371 1.44427E-05 372 1.27607E-05 373 1.14266E-05 374 1.03421E-05 375 9.2667E-06 376 9.34084E-06 377 8.3744E-06 378 8.30159E-06 379 7.50436E-06 380 7.63445E-06 381 7.66953E-06 382 6.78319E-06 383 7.34112E-06 384 7.79552E-06 385 7.95608E-06 386 7.46451E-06 387 6.19093E-06 388 6.86774E-06 389 7.37354E-06 390 7.7271E-06 391 5.72018E-06 392 7.56097E-06 393 6.84926E-06 394 6.54209E-06 395 7.44746E-06
"""


def cie_erythema(wl: float) -> float:
    if 250 <= wl <= 298:
        return 1.0
    if 298 < wl < 328:
        return 10 ** (0.094 * (298 - wl))
    if 328 <= wl <= 400:
        return 10 ** (0.015 * (139 - wl))
    return 0.0


def flint_caldwell(wl: float) -> float:
    if not 275 <= wl <= 390:
        return 0.0
    value = math.exp(
        4.688272 * math.exp(-math.exp(0.1703411 * (wl - 307.867) / 1.15))
        + ((390 - wl) / 121.7557 - 4.183832)
    )
    at_300 = math.exp(
        4.688272 * math.exp(-math.exp(0.1703411 * (300 - 307.867) / 1.15))
        + ((390 - 300) / 121.7557 - 4.183832)
    )
    return value / at_300


def caldwell_green_photon(wl: float) -> float:
    if not 280 <= wl <= 313.3:
        return 0.0
    return 2.618 * (1 - (wl / 313.3) ** 2) * math.exp((300 - wl) / 31.08)


def caldwell_green_energy(wl: float) -> float:
    raw = caldwell_green_photon(wl)
    raw_300 = caldwell_green_photon(300)
    return 0.0 if raw == 0 else (raw * wl / 300) / raw_300


def parse_neale() -> dict[int, float]:
    values = [float(v) for v in NEALE_TEXT.split()]
    return {int(values[i]): values[i + 1] for i in range(0, len(values), 2)}


def parse_two_column_spectrum(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        wavelength = float(parts[0])
        if abs(wavelength * 10 - round(wavelength * 10)) > 1e-6:
            continue
        result[round(wavelength * 10)] = float(parts[1])
    return result


def parse_temis_action_spectra(path: Path) -> dict[int, tuple[float, float, float]]:
    result: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, wavelength, erythema, dna, vitamin_d = parts
        result[round(float(wavelength) * 10)] = (float(erythema), float(dna), float(vitamin_d))
    return result


def write_bwf_table() -> None:
    neale = parse_neale()
    cullen = parse_two_column_spectrum(SOURCES / "doserates.Cullen_phaerodactylum.txt")
    cullen_300 = cullen[3000]
    temis = parse_temis_action_spectra(SOURCES / "TEMIS_action_v20.dat")
    path = DATA / "bwf_action_spectra_1nm.csv"
    fields = [
        "wavelength_nm",
        "cie_erythema_energy_norm298",
        "plant_flint_caldwell_energy_norm300",
        "plant_caldwell_green_photon_raw",
        "plant_caldwell_green_energy_norm300",
        "aquatic_neale_antarctic_raw_per_mW_m2",
        "aquatic_neale_antarctic_norm300",
        "aquatic_cullen_phaeodactylum_raw_per_mW_m2",
        "aquatic_cullen_phaeodactylum_per_W_m2",
        "aquatic_cullen_phaeodactylum_norm300",
        "vitamin_d_cie174_temis",
        "vitamin_d_young2021_blue_shift5",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for wl in range(280, 401):
            nraw = neale.get(wl)
            craw = cullen.get(wl * 10)
            vitamin_original = temis[wl * 10][2]
            vitamin_shifted = temis.get((wl + 5) * 10, (0.0, 0.0, 0.0))[2]
            writer.writerow(
                {
                    "wavelength_nm": wl,
                    "cie_erythema_energy_norm298": f"{cie_erythema(wl):.10g}",
                    "plant_flint_caldwell_energy_norm300": f"{flint_caldwell(wl):.10g}",
                    "plant_caldwell_green_photon_raw": f"{caldwell_green_photon(wl):.10g}",
                    "plant_caldwell_green_energy_norm300": f"{caldwell_green_energy(wl):.10g}",
                    "aquatic_neale_antarctic_raw_per_mW_m2": "" if nraw is None else f"{nraw:.10g}",
                    "aquatic_neale_antarctic_norm300": "" if nraw is None else f"{nraw / neale[300]:.10g}",
                    "aquatic_cullen_phaeodactylum_raw_per_mW_m2": "" if craw is None else f"{craw:.10g}",
                    "aquatic_cullen_phaeodactylum_per_W_m2": "" if craw is None else f"{craw * 1000:.10g}",
                    "aquatic_cullen_phaeodactylum_norm300": "" if craw is None else f"{craw / cullen_300:.10g}",
                    "vitamin_d_cie174_temis": f"{vitamin_original:.10g}",
                    "vitamin_d_young2021_blue_shift5": f"{vitamin_shifted:.10g}",
                }
            )

    qa = {
        "grid_nm": [280, 400, 1],
        "rows": 121,
        "cie_erythema_max_abs_difference_vs_temis_integer_grid": max(
            abs(cie_erythema(w) - temis[w * 10][0]) for w in range(280, 401)
        ),
        "cullen_phaeodactylum_integer_points_present": sum((w * 10) in cullen for w in range(280, 401)),
        "cullen_phaeodactylum_native_range_nm": [287, 389],
        "cullen_weight_at_300_per_mW_m2": cullen_300,
        "vitamin_d_shift_rule": "young2021(lambda) = TEMIS_CIE174(lambda + 5 nm); zero outside shifted source support",
        "vitamin_d_original_integer_points_present": sum((w * 10) in temis for w in range(280, 401)),
        "primary_missing_values_policy": "Values outside a source's native support are blank for Cullen and zero for the explicitly shifted vitamin-D curve.",
    }
    (DATA / "bwf_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    wl = np.arange(280, 401)
    curves = [
        ("Human: CIE erythema", "#1f77b4", [cie_erythema(x) for x in wl]),
        ("Human benefit: vitamin D, 5 nm blue shift", "#d62728", [temis.get((int(x) + 5) * 10, (0, 0, np.nan))[2] or np.nan for x in wl]),
        ("Plant: Flint-Caldwell 2003", "#2ca02c", [flint_caldwell(x) for x in wl]),
        ("Aquatic: Cullen Phaeodactylum", "#9467bd", [cullen.get(int(x) * 10, np.nan) / cullen_300 for x in wl]),
        ("Aquatic alt.: Neale Antarctic", "#8c564b", [neale.get(int(x), np.nan) / neale[300] for x in wl]),
    ]
    write_logline_svg(FIGURES / "bwf_action_spectra.svg", wl, curves)


def write_logline_svg(path: Path, x: np.ndarray, curves: list[tuple[str, str, list[float]]]) -> None:
    width, height = 1000, 610
    left, right, top, bottom = 95, 35, 65, 75
    pw, ph = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 280.0, 400.0, -5.0, math.log10(40)
    sx = lambda v: left + (v - xmin) / (xmax - xmin) * pw
    sy = lambda v: top + (ymax - math.log10(v)) / (ymax - ymin) * ph
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:13px}.label{font-size:16px}.title{font-size:21px;font-weight:600}</style>')
    for xt in range(280, 401, 20):
        xx = sx(xt); parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+ph}" stroke="#ddd"/>')
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{top+ph+24}" text-anchor="middle">{xt}</text>')
    for p in range(-5, 2):
        yy = sy(10**p); parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{yy+5:.1f}" text-anchor="end">10^{p}</text>')
    parts.append(f'<line x1="{sx(315):.1f}" y1="{top}" x2="{sx(315):.1f}" y2="{top+ph}" stroke="#777" stroke-dasharray="6 5"/>')
    for name, color, values in curves:
        pts = []
        for xx, value in zip(x, values):
            if value is not None and np.isfinite(value) and value >= 1e-5:
                pts.append(f'{sx(float(xx)):.1f},{sy(float(value)):.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round"/>')
    parts.append(f'<text class="title" x="{width/2}" y="32" text-anchor="middle">Selected biological weighting functions (1 nm)</text>')
    parts.append(f'<text class="label" x="{left+pw/2}" y="{height-20}" text-anchor="middle">Wavelength (nm)</text>')
    parts.append(f'<text class="label" transform="translate(25 {top+ph/2}) rotate(-90)" text-anchor="middle">Relative spectral weight (normalized at 298/300 nm)</text>')
    lx, ly = 610, 85
    for i, (name, color, _) in enumerate(curves):
        yy = ly + i*25; parts.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+28}" y2="{yy}" stroke="{color}" stroke-width="3"/><text class="tick" x="{lx+36}" y="{yy+5}">{name}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def scale_sobol(u: np.ndarray) -> dict[str, np.ndarray]:
    cos_sza = 0.0348995 + u[:, 0] * (1.0 - 0.0348995)
    sza = np.degrees(np.arccos(cos_sza))
    ozone = 150 + u[:, 1] * 400
    aod = np.expm1(u[:, 2] * math.log(4.0))
    ssa = 0.75 + u[:, 3] * 0.25
    angstrom = u[:, 4] * 2.5
    albedo = 0.02 + u[:, 5] * 0.93
    pressure = 500 + u[:, 6] * 550
    cot = np.expm1(u[:, 7] * math.log(151.0))
    reff = 5 + u[:, 8] * 25
    cloud_fraction = u[:, 9]
    water = np.exp(math.log(0.1) + u[:, 10] * (math.log(7.0) - math.log(0.1)))
    doy = np.floor(1 + u[:, 11] * 365).clip(1, 365).astype(int)
    return {
        "sza_deg": sza,
        "ozone_du": ozone,
        "aod550": aod,
        "ssa550": ssa,
        "angstrom_alpha": angstrom,
        "surface_albedo": albedo,
        "surface_pressure_hpa": pressure,
        "cloud_optical_thickness_550": cot,
        "cloud_effective_radius_um": reff,
        "cloud_fraction": cloud_fraction,
        "precipitable_water_cm": water,
        "day_of_year": doy,
    }


def sobol_points(n: int, dimensions: int, seed: int) -> np.ndarray:
    """Generate a digitally shifted Sobol sequence for dimensions 1..12."""
    params = [
        None,
        (1, 0, [1]), (2, 1, [1, 3]), (3, 1, [1, 3, 1]),
        (3, 2, [1, 1, 1]), (4, 1, [1, 3, 5, 13]), (4, 4, [1, 1, 5, 5]),
        (5, 2, [1, 3, 3, 9, 7]), (5, 4, [1, 1, 5, 11, 27]),
        (5, 7, [1, 1, 7, 13, 3]), (5, 11, [1, 1, 5, 1, 15]),
        (5, 13, [1, 1, 1, 13, 31]),
    ]
    bits = 32
    directions = np.zeros((dimensions, bits), dtype=np.uint32)
    for j in range(bits):
        directions[0, j] = np.uint32(1 << (31 - j))
    for dim in range(1, dimensions):
        s, a, initial = params[dim]
        for j in range(s):
            directions[dim, j] = np.uint32(initial[j] << (31 - j))
        for j in range(s, bits):
            value = directions[dim, j - s] ^ (directions[dim, j - s] >> np.uint32(s))
            for k in range(1, s):
                if (a >> (s - 1 - k)) & 1:
                    value ^= directions[dim, j - k]
            directions[dim, j] = value
    rng = np.random.default_rng(seed)
    shift = rng.integers(0, 2**32, size=dimensions, dtype=np.uint32)
    state = np.zeros(dimensions, dtype=np.uint32)
    points = np.empty((n, dimensions), dtype=np.float64)
    points[0] = (state ^ shift).astype(np.float64) / 2**32
    for i in range(1, n):
        c = (i & -i).bit_length() - 1
        state ^= directions[:, c]
        points[i] = (state ^ shift).astype(np.float64) / 2**32
    return points


def sobol_block(power: int, seed: int, split: str, start_id: int) -> list[dict[str, object]]:
    mapped = scale_sobol(sobol_points(2**power, 12, seed))
    rows = []
    for i in range(2**power):
        row = {"sample_id": f"{split}_{start_id + i:05d}", "split": split}
        for key, values in mapped.items():
            value = values[i]
            row[key] = int(value) if key == "day_of_year" else round(float(value), 8)
        rows.append(row)
    return rows


def write_sobol() -> None:
    rows = sobol_block(13, 20261013, "train", 1) + sobol_block(11, 20261014, "holdout", 1)
    path = DATA / "sobol_parameters_10240.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric = [k for k in rows[0] if k not in {"sample_id", "split"}]
    train = np.array([[float(row[k]) for k in numeric] for row in rows if row["split"] == "train"])
    corr = np.corrcoef(train, rowvar=False)
    np.fill_diagonal(corr, np.nan)
    diagnostic = {
        "design": "digitally shifted Sobol",
        "dimensions": 12,
        "training_samples": 8192,
        "holdout_samples": 2048,
        "training_seed": 20261013,
        "holdout_seed": 20261014,
        "max_abs_pairwise_pearson_r_training": float(np.nanmax(np.abs(corr))),
        "note": "Uniformity is in the declared transformed coordinates; day_of_year is discretized after sampling.",
    }
    (DATA / "sobol_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")

    labels = ["SZA", "O3", "AOD", "SSA", "Angstrom", "albedo", "pressure", "COT", "r_eff", "cloud frac.", "water", "DOY"]
    write_histogram_svg(FIGURES / "sobol_parameter_distributions.svg", train, labels)


def write_histogram_svg(path: Path, data: np.ndarray, labels: list[str]) -> None:
    width, height = 1120, 780
    cols, rows, bins = 4, 3, 24
    margin_x, margin_y, gap_x, gap_y = 55, 70, 28, 44
    cell_w = (width - 2*margin_x - (cols-1)*gap_x) / cols
    cell_h = (height - 2*margin_y - (rows-1)*gap_y) / rows
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:600}.small{font-size:12px}.panel{font-size:14px;font-weight:600}</style>']
    parts.append(f'<text class="title" x="{width/2}" y="32" text-anchor="middle">Digitally shifted Sobol training design: marginal distributions (n=8,192)</text>')
    for idx, label in enumerate(labels):
        row, col = divmod(idx, cols)
        x0 = margin_x + col*(cell_w+gap_x); y0 = margin_y + row*(cell_h+gap_y)
        counts, edges = np.histogram(data[:, idx], bins=bins)
        maxc = max(counts); bar_w = cell_w/bins
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="#f7f9fb" stroke="#bbb"/>')
        for b, count in enumerate(counts):
            bh = count/maxc*(cell_h-28)
            parts.append(f'<rect x="{x0+b*bar_w+0.5:.1f}" y="{y0+cell_h-bh:.1f}" width="{max(bar_w-1,0.5):.1f}" height="{bh:.1f}" fill="#2374ab"/>')
        parts.append(f'<text class="panel" x="{x0+cell_w/2:.1f}" y="{y0-8:.1f}" text-anchor="middle">{label}</text>')
        parts.append(f'<text class="small" x="{x0:.1f}" y="{y0+cell_h+17:.1f}">{edges[0]:.3g}</text><text class="small" x="{x0+cell_w:.1f}" y="{y0+cell_h+17:.1f}" text-anchor="end">{edges[-1]:.3g}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_bwf_table()
    write_sobol()
    print("Generated phase-0 BWF and Sobol deliverables.")


if __name__ == "__main__":
    main()
