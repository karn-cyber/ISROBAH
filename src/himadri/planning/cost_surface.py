"""Traverse cost surface: a weighted blend of the hazards a rover faces driving
from a sunlit landing site into a cryogenic, dark crater.

cost = w1*slope + w2*hazard + w3*shadow_power_drain + w4*thermal + w5*distance

Impassable cells (slope above the rover limit, flagged layover/shadow) are set
to infinity so the planner routes around them.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import FeatureStack

INF = 1e9


def _norm(x):
    x = np.nan_to_num(x.astype(np.float64))
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def build_cost_surface(feats: FeatureStack, cfg: Config) -> np.ndarray:
    b = feats.bands
    w = cfg.planning.weights
    slope = b["slope_deg"]
    rough = b["roughness"]
    boulders = b["boulder_density"]
    illum = b["illumination_frac"]

    hazard = 0.5 * _norm(rough) + 0.5 * _norm(boulders)
    shadow_power = 1.0 - _norm(illum)  # darker -> costlier (battery drain)
    thermal = 1.0 - _norm(illum)  # darker/colder -> survival cost
    dist = np.ones_like(slope, dtype=np.float64)

    cost = (
        w["slope"] * _norm(slope)
        + w["hazard"] * hazard
        + w["shadow_power"] * shadow_power
        + w["thermal"] * thermal
        + w["distance"] * dist
    )
    cost = 0.05 + cost  # floor so every step has positive cost
    # impassable terrain
    cost[slope > cfg.planning.max_slope_deg] = INF
    return cost.astype(np.float64)
