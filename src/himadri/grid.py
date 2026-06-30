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


def lonlat_to_polar_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Project a selenographic lon/lat (deg) to lunar south-polar stereographic
    metres. Used to centre the analysis grid on a named crater."""
    from pyproj import Transformer

    from .types import LUNAR_SOUTH_POLAR_STEREO

    geo = "+proj=longlat +a=1737400 +b=1737400 +no_defs"
    tr = Transformer.from_crs(geo, LUNAR_SOUTH_POLAR_STEREO, always_xy=True)
    x, y = tr.transform(lon_deg, lat_deg)
    return float(x), float(y)


def grid_centered_on_lonlat(lon_deg: float, lat_deg: float, half_width_m: float,
                            res_m: float) -> Grid:
    """A square Grid of half-width `half_width_m` centred on a crater's lon/lat."""
    from affine import Affine

    cx, cy = lonlat_to_polar_xy(lon_deg, lat_deg)
    n = int(round(2 * half_width_m / res_m))
    x0 = cx - half_width_m
    y0 = cy + half_width_m
    transform = Affine(res_m, 0, x0, 0, -res_m, y0)
    return Grid(transform=transform, height=n, width=n, res_m=res_m)


def grid_from_dataset(path: str, max_dim: int = 512) -> Grid:
    """Build a Grid matching a raster's footprint, downsampled so the longer
    side is at most `max_dim` pixels (keeps uploaded full-tile DEMs tractable)."""
    import rasterio
    from affine import Affine

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        t = src.transform
        res = max(abs(t.a), abs(t.e))
        crs = src.crs.to_proj4() if src.crs else Grid.centered(1, 1, 1).crs
    scale = max(1.0, max(h, w) / max_dim)
    new_res = res * scale
    new_h = int(round(h / scale))
    new_w = int(round(w / scale))
    transform = Affine(new_res, t.b, t.c, t.d, -new_res, t.f)
    return Grid(transform=transform, height=new_h, width=new_w, res_m=new_res, crs=crs)


def warp_to_grid(path: str, grid: Grid, resampling: str = "bilinear",
                 indexes=None) -> np.ndarray:
    """Warp a raster file onto `grid`, reading only the needed region via a
    WarpedVRT (memory-safe for multi-GB polar DEMs / OHRC strips)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    rs = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear,
          "cubic": Resampling.cubic}[resampling]
    dst_crs = CRS.from_proj4(grid.crs)
    with rasterio.open(path) as src:
        src_crs = src.crs or dst_crs  # assume already on grid CRS if undefined
        with WarpedVRT(src, crs=dst_crs, src_crs=src_crs, transform=grid.transform,
                       width=grid.width, height=grid.height, resampling=rs) as vrt:
            data = vrt.read(indexes=indexes).astype(np.float32)
            nod = src.nodata
    if nod is not None:
        data = np.where(data == nod, np.nan, data)
    return data


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
