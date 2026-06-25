"""Illumination modelling: map Permanently Shadowed Regions (PSRs) and the
extreme 'doubly-shadowed' interiors from a DEM + solar geometry via horizon
ray-casting.

For each solar azimuth (low polar sun) we cast horizons across the DEM and mark
lit pixels; the time-fraction lit is the illumination map. PSRs never receive
direct sun; doubly-shadowed pixels receive neither direct sun nor meaningful
secondary (scattered) light from nearby lit terrain — the coldest, most
ice-favourable, hardest-to-observe environments.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..types import DEMProduct, SunGeometry


def _lit_for_direction(z: np.ndarray, res_m: float, azimuth_deg: float,
                       sun_elev_deg: float) -> np.ndarray:
    """Boolean lit-mask for a single sun direction, via horizon scanning.

    Rotate the DEM so the sun direction runs along +columns, then for each row
    track the projected horizon: cell j is lit iff z[j] >= H_j where
    H_j = max(H_{j-1}, z[j-1]) - res*tan(elev).
    """
    rot = ndimage.rotate(z, azimuth_deg, reshape=False, order=1, mode="nearest")
    tan_e = np.tan(np.radians(max(sun_elev_deg, 0.05)))
    drop = res_m * tan_e
    n = rot.shape[1]
    lit = np.zeros_like(rot, dtype=np.float32)
    horizon = np.full(rot.shape[0], -np.inf, dtype=np.float64)
    for j in range(n):
        lit[:, j] = (rot[:, j] >= horizon).astype(np.float32)
        horizon = np.maximum(horizon, rot[:, j]) - drop
    # rotate the lit mask back to the original frame
    back = ndimage.rotate(lit, -azimuth_deg, reshape=False, order=1, mode="nearest")
    return (back >= 0.5).astype(np.float32)


def horizon_illumination(dem: DEMProduct, sun: SunGeometry) -> np.ndarray:
    """Illumination fraction (0..1): fraction of sampled sun positions a pixel
    is directly lit."""
    z = dem.elevation.astype(np.float64)
    res = dem.grid.res_m
    acc = np.zeros(z.shape, dtype=np.float32)
    n = len(sun.azimuth_deg)
    for az, el in zip(sun.azimuth_deg, sun.elevation_deg):
        acc += _lit_for_direction(z, res, float(az), float(el))
    return (acc / max(n, 1)).astype(np.float32)


def secondary_illumination(illum: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Proxy for scattered/secondary light: smoothed direct illumination — a
    pixel surrounded by lit terrain receives some bounced light."""
    return ndimage.gaussian_filter(illum.astype(np.float64), sigma=sigma).astype(np.float32)


def psr_mask(illum: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """Permanently Shadowed Region: never (effectively) directly lit."""
    return (illum <= thresh).astype(bool)


def doubly_shadowed_mask(illum: np.ndarray, secondary_thresh: float = 0.05) -> np.ndarray:
    """Doubly-shadowed: no direct sun AND negligible secondary illumination."""
    secondary = secondary_illumination(illum)
    return ((illum <= 0.01) & (secondary <= secondary_thresh)).astype(bool)


def illumination_stack(dem: DEMProduct, sun: SunGeometry) -> dict[str, np.ndarray]:
    illum = horizon_illumination(dem, sun)
    return {
        "illumination_frac": illum,
        "psr_mask": psr_mask(illum).astype(np.float32),
        "doubly_shadowed_mask": doubly_shadowed_mask(illum).astype(np.float32),
    }
