"""Terrain correction for polar SAR geometry.

Steep crater walls + side-looking radar -> strongly varying local incidence
angle, and CPR itself depends on incidence angle, so raw comparisons across a
crater are apples-to-oranges. We compute a per-pixel local incidence angle from
the DEM and normalise CPR before classification.
"""
from __future__ import annotations

import numpy as np


def incidence_angle(elevation: np.ndarray, res_m: float,
                    look_azimuth_deg: float = 0.0,
                    nominal_inc_deg: float = 35.0) -> np.ndarray:
    """Local incidence angle (deg) = nominal incidence modified by the terrain
    slope projected into the radar look direction."""
    gy, gx = np.gradient(elevation.astype(np.float64), res_m)
    az = np.radians(look_azimuth_deg)
    # slope component in the look (range) direction
    slope_range = gx * np.cos(az) + gy * np.sin(az)
    local = np.degrees(np.radians(nominal_inc_deg) - np.arctan(slope_range))
    return np.clip(local, 5.0, 85.0).astype(np.float32)


def normalize_cpr(cpr: np.ndarray, incidence: np.ndarray,
                  ref_inc_deg: float = 35.0) -> np.ndarray:
    """Normalise CPR to a reference incidence angle. CPR rises with incidence;
    we apply a mild cosine-ratio correction so cross-crater CPR is comparable."""
    cos_ref = np.cos(np.radians(ref_inc_deg))
    cos_loc = np.cos(np.radians(incidence))
    corr = cos_loc / (cos_ref + 1e-6)
    return (cpr * corr).astype(np.float32)
