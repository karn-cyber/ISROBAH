"""Polarimetric speckle filtering.

Critical detail: filter the Stokes vector (proportional to the coherency
matrix), NOT the CPR/DOP ratios — ratios of noisy quantities are biased.
We apply a refined-Lee-style adaptive filter to each Stokes channel using the
local statistics of total power S0, so edges (crater rims) are preserved.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def refined_lee(stokes: np.ndarray, window: int = 7, looks: float = 8.0) -> np.ndarray:
    """Adaptive Lee filter applied coherently across the 4 Stokes channels.

    The adaptive weight is derived from S0 (total power) and applied to all
    channels so polarimetric ratios stay consistent.
    """
    s0 = stokes[0].astype(np.float64)
    mean = ndimage.uniform_filter(s0, size=window)
    mean_sq = ndimage.uniform_filter(s0**2, size=window)
    var = np.maximum(mean_sq - mean**2, 0.0)
    # multiplicative speckle model: Cu^2 = 1/looks
    cu2 = 1.0 / looks
    ci2 = var / (mean**2 + 1e-9)
    w = np.clip(1.0 - cu2 / (ci2 + 1e-9), 0.0, 1.0)  # Lee weight

    out = np.empty_like(stokes)
    for c in range(stokes.shape[0]):
        ch = stokes[c].astype(np.float64)
        ch_mean = ndimage.uniform_filter(ch, size=window)
        out[c] = (ch_mean + w * (ch - ch_mean)).astype(np.float32)
    out[0] = np.maximum(out[0], 1e-6)
    return out
