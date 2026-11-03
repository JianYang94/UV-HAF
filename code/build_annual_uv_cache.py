"""Populate per-year SSP2-4.5 UV caches for an inclusive year range."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uv_reconstruction as base

start, end = map(int, sys.argv[1:3])
scenario = sys.argv[3] if len(sys.argv) > 3 else "ssp245"
for year in range(start, end + 1):
    base.daily_dose(scenario, (year, year))
    print(f"YEAR_COMPLETE {scenario} {year}", flush=True)
