"""GeoTIFF / vector writers. Every raster is written with the scene's CRS and
transform so the outputs open correctly in QGIS / ArcGIS."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from ..types import Grid


def _crs(grid: Grid) -> CRS:
    return CRS.from_proj4(grid.crs)


def write_geotiff(path: str | Path, arr: np.ndarray, grid: Grid, nodata=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    count = arr.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=count,
        dtype="float32",
        crs=_crs(grid),
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        for b in range(count):
            dst.write(arr[b].astype("float32"), b + 1)
    return path


def write_json(path: str | Path, obj: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))
    return path


def write_vector(gdf, path: str | Path, layer: str | None = None) -> Path:
    """Write a GeoDataFrame to GeoPackage. Falls back to GeoJSON if the GPKG
    driver is unavailable in the local GDAL build."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_file(path, layer=layer, driver="GPKG")
        return path
    except Exception:
        alt = path.with_suffix(".geojson")
        gdf.to_file(alt, driver="GeoJSON")
        return alt
