"""Regenerate and verify every data-driven figure used by the manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops


JOBS = (
    ("plot_figure2.py", "Figure2"),
    ("plot_figure3.py", "Figure3"),
    ("plot_figure4.py", "Figure4"),
    ("plot_figure5.py", "Figure5"),
    ("plot_extended_data_figure1.py", "Extended_Data_Figure_1"),
    ("plot_extended_data_figure2.py", "Extended_Data_Figure_2"),
    ("plot_extended_data_figure3.py", "Extended_Data_Figure_3"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-against", type=Path, default=None)
    return parser.parse_args()


def check_pdf(path: Path):
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-4096:]:
        raise RuntimeError(f"Invalid PDF output: {path}")


def verify_manifest(root: Path):
    manifest = root / "data_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    checked = 0
    for row in rows:
        path = root / row["path"]
        expected = row.get("sha256", "").strip().upper()
        if not path.is_file() or not expected:
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise RuntimeError(f"Source-data checksum mismatch: {path}")
        checked += 1
    if checked == 0:
        raise RuntimeError("No source-data checksums were verified")
    print(f"Verified {checked} source-data checksums", flush=True)


def compare_png(actual: Path, expected: Path):
    with Image.open(actual) as a, Image.open(expected) as e:
        a.load()
        e.load()
        if a.size != e.size:
            raise RuntimeError(
                f"PNG dimensions differ for {actual.name}: {a.size} != {e.size}"
            )
        difference = ImageChops.difference(a.convert("RGBA"), e.convert("RGBA"))
        if difference.getbbox() is not None:
            raise RuntimeError(f"PNG pixels differ from committed figure: {actual.name}")


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    output = (args.output_dir or root / "figures").resolve()
    reference = args.verify_against.resolve() if args.verify_against else None
    output.mkdir(parents=True, exist_ok=True)
    code = root / "code"
    verify_manifest(root)

    for script, stem in JOBS:
        command = [
            sys.executable,
            str(code / script),
            "--repo-root",
            str(root),
            "--output-dir",
            str(output),
        ]
        print(f"Running {script}", flush=True)
        subprocess.run(command, check=True)
        png = output / f"{stem}.png"
        pdf = output / f"{stem}.pdf"
        if not png.is_file() or not pdf.is_file():
            raise RuntimeError(f"Missing output for {stem}")
        check_pdf(pdf)
        if reference is not None:
            compare_png(png, reference / png.name)

    supplementary = code / "plot_supplementary_figure_s1.py"
    print(f"Running {supplementary.name}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(supplementary),
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
    )
    stem = "Supplementary_Figure_S1"
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    if not png.is_file() or not pdf.is_file():
        raise RuntimeError(f"Missing output for {stem}")
    check_pdf(pdf)
    if reference is not None:
        compare_png(png, reference / png.name)

    for suffix in ("svg", "png", "pdf"):
        schematic = root / "figures" / f"Figure1.{suffix}"
        if not schematic.is_file():
            raise RuntimeError(f"Missing Figure 1 schematic: {schematic}")
    check_pdf(root / "figures" / "Figure1.pdf")
    print("All final figure files were generated and verified.")


if __name__ == "__main__":
    main()
