"""Adapter layer: every real dataset is wrapped by a loader that returns the
common data model (types.py). When a real loader is unavailable, the synthetic
generator provides the same structures so downstream code is identical.

NOTE: the official crater DFSAR/OHRC files are supplied by the organisers at
the event. The real-mode branches below read GeoTIFF directly and document the
native-format adapter as a clearly marked TODO[VERIFY], so real-data
uncertainty never blocks the synthetic pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from ..types import DEMProduct, Grid, OpticalProduct, RadarProduct, SunGeometry


def _read_grid(src) -> Grid:
    return Grid(
        transform=src.transform,
        height=src.height,
        width=src.width,
        res_m=abs(src.transform.a),
        crs=src.crs.to_proj4() if src.crs else Grid.centered(1, 1, 1).crs,
    )


def load_dfsar(path: str, band: Literal["L", "S"]) -> RadarProduct:
    """Load a DFSAR product as full Stokes (4,H,W).

    Expected GeoTIFF layouts (real mode):
      * 4-band GeoTIFF already holding S0,S1,S2,S3, OR
      * multi-pol channels we convert to Stokes.

    TODO[VERIFY]: ISSDC PRADAN ships ISDA (XML label + raw/derived). Confirm the
    native polarimetric layout (quad-pol covariance vs hybrid-pol Stokes) at the
    event and adapt the channel->Stokes mapping here. Synthetic mode bypasses
    this entirely.
    """
    with rasterio.open(path) as src:
        grid = _read_grid(src)
        data = src.read().astype(np.float32)
    if data.shape[0] == 4:
        stokes = data
    else:
        # TODO[VERIFY]: build Stokes from per-pol channels for the real layout.
        raise ValueError(
            f"DFSAR {band}: expected 4 Stokes bands, got {data.shape[0]}. "
            "Implement the channel->Stokes adapter for the supplied format."
        )
    return RadarProduct(grid=grid, stokes=stokes, band=band)


def load_ohrc(path: str) -> OpticalProduct:
    with rasterio.open(path) as src:
        grid = _read_grid(src)
        refl = src.read(1).astype(np.float32)
        if src.nodata is not None:
            refl = np.where(refl == src.nodata, np.nan, refl)
    return OpticalProduct(grid=grid, reflectance=refl)


def load_dem(path: str | None) -> DEMProduct | None:
    """Load a DEM GeoTIFF. None -> caller uses the synthetic fallback."""
    if path is None:
        return None
    with rasterio.open(path) as src:
        grid = _read_grid(src)
        elev = src.read(1).astype(np.float32)
    return DEMProduct(grid=grid, elevation=elev)


def load_sun_geometry(path: str | None) -> SunGeometry | None:
    """Load a solar azimuth/elevation time series (CSV: az_deg,el_deg).
    None -> caller uses the analytic fallback.

    TODO[VERIFY]: for precise illumination, swap in spiceypy + NAIF kernels.
    """
    if path is None:
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return SunGeometry(azimuth_deg=arr[:, 0], elevation_deg=arr[:, 1])
