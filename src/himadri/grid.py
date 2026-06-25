"""Grid / CRS helpers: build the common grid and reproject products onto it.

In synthetic mode every product is generated already on the common grid, so
reprojection is a no-op pass-through. In real mode `coregister.to_grid`
(preprocess) uses these helpers to warp DFSAR / OHRC / DEM onto one grid.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .types import Grid


def build_grid(cfg: Config) -> Grid:
    return Grid.centered(cfg.grid.height, cfg.grid.width, cfg.grid.res_m)


def reproject_to_grid(
    src: np.ndarray,
    src_grid: Grid,
    dst_grid: Grid,
    resampling: str = "bilinear",
) -> np.ndarray:
    """Reproject a (H,W) or (B,H,W) array from src_grid to dst_grid.

    Uses rasterio.warp when grids differ; returns the array unchanged when the
    grids already match (the synthetic-mode common case).
    """
    same = (
        src_grid.crs == dst_grid.crs
        and src_grid.shape == dst_grid.shape
        and np.allclose(list(src_grid.transform), list(dst_grid.transform))
    )
    if same:
        return src

    from rasterio.warp import Resampling, reproject

    rs = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }[resampling]

    squeeze = src.ndim == 2
    arr = src[None] if squeeze else src
    out = np.empty((arr.shape[0], dst_grid.height, dst_grid.width), dtype=np.float32)
    for b in range(arr.shape[0]):
        reproject(
            source=np.ascontiguousarray(arr[b].astype(np.float32)),
            destination=out[b],
            src_transform=src_grid.transform,
            src_crs=src_grid.crs,
            dst_transform=dst_grid.transform,
            dst_crs=dst_grid.crs,
            resampling=rs,
        )
    return out[0] if squeeze else out
