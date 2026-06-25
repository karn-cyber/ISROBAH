"""Landing-site selection via a multi-criteria suitability surface.

A scientifically perfect site is useless if you cannot land or survive there.
We score every candidate pixel on: low slope, low boulder/roughness hazard,
illumination (power budget), proximity to the ice target, and comms
line-of-sight to Earth — then rank well-separated local optima.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import Config
from ..types import FeatureStack, Grid, IceResult


def _norm(x, invert=False):
    x = np.nan_to_num(x.astype(np.float64))
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    n = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    return 1 - n if invert else n


def suitability(feats: FeatureStack, ice: IceResult, cfg: Config) -> np.ndarray:
    b = feats.bands
    w = cfg.landing.weights
    slope = b["slope_deg"]
    rough = b["roughness"]
    illum = b["illumination_frac"]

    # distance (m) to high-confidence ice
    target = ice.probability > cfg.detection.target_prob
    if target.any():
        dist = ndimage.distance_transform_edt(~target) * feats.grid.res_m
    else:
        dist = np.full(slope.shape, 1e6, dtype=np.float64)
    # comms LOS proxy: prefer higher ground with sky view (use illumination's
    # smoothed field as a stand-in for horizon openness toward Earth)
    comms = ndimage.gaussian_filter(illum.astype(np.float64), 4)

    score = (
        w["slope"] * _norm(slope, invert=True)
        + w["roughness"] * _norm(rough, invert=True)
        + w["illumination"] * _norm(illum)
        + w["proximity"] * _norm(dist, invert=True)
        + w["comms"] * _norm(comms)
    )
    # hard constraint: too-steep terrain is unlandable
    score[slope > cfg.landing.max_slope_deg] = 0.0
    # cannot land inside permanent shadow (no power, no visibility)
    score[b["psr_mask"] > 0.5] *= 0.15
    return score.astype(np.float32)


def rank_sites(suit: np.ndarray, grid: Grid, cfg: Config):
    import geopandas as gpd
    from shapely.geometry import Point

    s = suit.copy()
    sites = []
    sep_px = int(cfg.landing.min_separation_m / grid.res_m)
    for _ in range(cfg.landing.n_sites):
        idx = np.argmax(s)
        r, c = np.unravel_index(idx, s.shape)
        if s[r, c] <= 0:
            break
        x, y = grid.rc_to_xy(r, c)
        sites.append({"rank": len(sites) + 1, "row": int(r), "col": int(c),
                      "score": float(s[r, c]), "geometry": Point(x, y)})
        r0, r1 = max(0, r - sep_px), min(s.shape[0], r + sep_px)
        c0, c1 = max(0, c - sep_px), min(s.shape[1], c + sep_px)
        s[r0:r1, c0:c1] = -1  # suppress neighbourhood
    gdf = gpd.GeoDataFrame(sites, crs=grid.crs) if sites else gpd.GeoDataFrame(
        columns=["rank", "row", "col", "score", "geometry"], crs=grid.crs)
    return gdf


def select_landing(feats: FeatureStack, ice: IceResult, cfg: Config):
    suit = suitability(feats, ice, cfg)
    gdf = rank_sites(suit, feats.grid, cfg)
    return suit, gdf
