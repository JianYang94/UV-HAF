"""Load and standardise the gridded Water Hazard Index used by CHS.

The publication calculations use the released 0.5 degree derivative. When the
source GeoTIFF is available locally, this module can reproduce that derivative
by area-average resampling it to the analysis grid. The Water Hazard Index is a
fixed spatial baseline and is not included in the CRITIC indicator weighting.
"""
from __future__ import annotations

import os

import numpy as np

from . import grid

WHI_PATH = os.environ.get("UV_HAF_WHI_PATH", "")
RELEASED_WHI = os.path.join(
    os.path.dirname(__file__), "released", "whi_field_0p5deg.npz"
)


def has_whi():
    """Return whether the source raster or released derivative is available."""
    return os.path.exists(WHI_PATH) or os.path.exists(RELEASED_WHI)


def _resample_to_grid(path, G, resampling=None):
    """Area-average a geographic GeoTIFF onto the 0.5 degree analysis grid."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    resampling = resampling or Resampling.average
    lon, lat, resolution = G["lon"], G["lat"], G["res"]
    transform = from_origin(
        lon[0] - resolution / 2,
        lat[-1] + resolution / 2,
        resolution,
        resolution,
    )
    destination = np.full((lat.size, lon.size), np.nan, dtype="float32")
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=source.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination[::-1, :]


def _raw_whi_on_grid(G):
    if os.path.exists(WHI_PATH):
        return _resample_to_grid(WHI_PATH, G)
    if os.path.exists(RELEASED_WHI):
        with np.load(RELEASED_WHI, allow_pickle=False) as released:
            return released["whi"].astype(float)
    raise FileNotFoundError(
        "No Water Hazard Index is available. Set UV_HAF_WHI_PATH or provide "
        f"the released derivative at {RELEASED_WHI}."
    )


def load_whi_field(G=None):
    """Return the standardised baseline, land coverage and unscaled WHI field."""
    G = G or grid.build()
    whi = _raw_whi_on_grid(G)
    land = G["land2d"]
    valid = land & np.isfinite(whi)
    coverage = float(G["area2d"][valid].sum() / G["area2d"][land].sum())
    median = float(np.nanmedian(whi[valid]))
    filled = np.where(land & np.isfinite(whi), whi, median)
    values = filled[land]
    baseline = (filled - values.mean()) / (values.std() or 1.0) * grid.B_STD
    baseline = np.where(land, baseline, np.nan)
    return baseline, coverage, whi
