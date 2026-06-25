"""Common data model — the contract every pipeline stage speaks.

All rasters share one Grid: a lunar south-polar stereographic projection,
fixed resolution, identical shape and affine transform. This guarantees
radar, optical and DEM layers line up pixel-for-pixel before any fusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from affine import Affine

# Lunar south-polar stereographic. The Moon has no standard EPSG code that
# pyproj ships universally, so we define an explicit proj4 string on a
# spherical Moon (IAU mean radius 1737.4 km). This avoids EPSG ambiguity and
# works identically across rasterio / pyproj / geopandas.
LUNAR_SOUTH_POLAR_STEREO = (
    "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)


@dataclass
class Grid:
    """Shared geospatial grid for every raster in the pipeline."""

    transform: Affine
    height: int
    width: int
    res_m: float
    crs: str = LUNAR_SOUTH_POLAR_STEREO

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @classmethod
    def centered(cls, height: int, width: int, res_m: float) -> "Grid":
        """A grid centred on the origin (south pole), pixel size res_m."""
        x0 = -(width / 2.0) * res_m
        y0 = (height / 2.0) * res_m
        transform = Affine(res_m, 0.0, x0, 0.0, -res_m, y0)
        return cls(transform=transform, height=height, width=width, res_m=res_m)

    def pixel_area_m2(self) -> float:
        return self.res_m * self.res_m

    def rc_to_xy(self, row: float, col: float) -> tuple[float, float]:
        x, y = self.transform * (col + 0.5, row + 0.5)
        return x, y


@dataclass
class RadarProduct:
    """Full-Stokes DFSAR product for a single band (L or S)."""

    grid: Grid
    stokes: np.ndarray  # (4, H, W): S0, S1, S2, S3  (float32)
    band: str = "L"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.stokes.ndim == 3 and self.stokes.shape[0] == 4, (
            f"stokes must be (4,H,W), got {self.stokes.shape}"
        )


@dataclass
class OpticalProduct:
    grid: Grid
    reflectance: np.ndarray  # (H, W) float32, NaN inside permanent shadow
    meta: dict = field(default_factory=dict)


@dataclass
class DEMProduct:
    grid: Grid
    elevation: np.ndarray  # (H, W) float32, metres


@dataclass
class SunGeometry:
    """Solar positions over (a representative sample of) a lunar day."""

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray


@dataclass
class FeatureStack:
    grid: Grid
    bands: dict[str, np.ndarray]  # name -> (H, W) float32

    def stack(self, names: list[str]) -> np.ndarray:
        """Return a (N, H, W) array in the given band order."""
        return np.stack([self.bands[n] for n in names], axis=0)

    def feature_names(self) -> list[str]:
        return list(self.bands.keys())


@dataclass
class IceResult:
    grid: Grid
    probability: np.ndarray  # (H, W) 0..1
    uncertainty: np.ndarray  # (H, W) 0..1
    target_mask: np.ndarray  # (H, W) bool
    meta: dict = field(default_factory=dict)


@dataclass
class VolumeResult:
    total_m3: float
    lower_m3: float
    upper_m3: float
    mean_ice_fraction: float
    per_depth_bin: list[dict]
    meta: dict = field(default_factory=dict)


@dataclass
class Scene:
    """Everything loaded/synthesised for one crater, on a common grid."""

    grid: Grid
    radar_l: RadarProduct
    radar_s: RadarProduct
    optical: OpticalProduct
    dem: DEMProduct
    sun: SunGeometry
    truth: dict = field(default_factory=dict)  # synthetic ground-truth rasters
