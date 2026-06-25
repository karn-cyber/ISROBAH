"""Geometry validity masks: layover / radar-shadow pixels are flagged so they
can be reported as low-confidence rather than silently trusted."""
from __future__ import annotations

import numpy as np


def layover_shadow(elevation: np.ndarray, res_m: float,
                   incidence: np.ndarray) -> np.ndarray:
    """Boolean mask of geometry-compromised pixels.

    Layover where the local slope toward the radar exceeds the incidence angle;
    radar-shadow where the slope away exceeds (90 - incidence).
    """
    gy, gx = np.gradient(elevation.astype(np.float64), res_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    layover = slope > incidence
    shadow = slope > (90.0 - incidence)
    return (layover | shadow).astype(bool)
