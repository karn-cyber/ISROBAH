"""Co-registration: bring DFSAR / OHRC / DEM onto one common polar grid.

In synthetic mode all products are generated on the common grid (no-op). In
real mode each product is reprojected via grid.reproject_to_grid using the DEM
as the geometric backbone.
"""
from __future__ import annotations

from ..grid import reproject_to_grid
from ..types import Grid, OpticalProduct, RadarProduct


def radar_to_grid(radar: RadarProduct, grid: Grid) -> RadarProduct:
    stokes = reproject_to_grid(radar.stokes, radar.grid, grid, "bilinear")
    return RadarProduct(grid=grid, stokes=stokes, band=radar.band, meta=radar.meta)


def optical_to_grid(optical: OpticalProduct, grid: Grid) -> OpticalProduct:
    refl = reproject_to_grid(optical.reflectance, optical.grid, grid, "bilinear")
    return OpticalProduct(grid=grid, reflectance=refl, meta=optical.meta)
