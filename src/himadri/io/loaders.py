"""Adapter layer: every real dataset is wrapped by a loader that returns the
common data model (types.py). When a real loader is unavailable, the synthetic
generator provides the same structures so downstream code is identical.

Real sources (verified Jun 2026):
  * DFSAR / OHRC : ISSDC PRADAN  https://pradan.issdc.gov.in/ch2/  (free account).
                   DFSAR scene for the crater is SUPPLIED by the organisers.
  * LOLA DEM     : LOLA PDS node https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/
                   e.g. LDEM_85S_10M.IMG (10 m/px, 85-90S, ~1.75 GB) covers Faustini.

All loaders accept an optional target `grid`; when given, the product is warped
onto it via a memory-safe WarpedVRT (so multi-GB polar tiles never load whole).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import rasterio

from ..grid import grid_from_dataset, warp_to_grid
from ..types import DEMProduct, Grid, OpticalProduct, RadarProduct, SunGeometry


def _grid_from_open(src) -> Grid:
    return Grid(
        transform=src.transform, height=src.height, width=src.width,
        res_m=abs(src.transform.a),
        crs=src.crs.to_proj4() if src.crs else Grid.centered(1, 1, 1).crs,
    )


# --------------------------------------------------------------------------- #
#  DFSAR  (scattering matrix -> circular-transmit Stokes)                     #
# --------------------------------------------------------------------------- #

def stokes_from_scattering_matrix(shh, shv, svh, svv,
                                  transmit: str = "L") -> np.ndarray:
    """Build the (4,H,W) Stokes vector of the backscattered field for a
    circular-transmit, dual-linear-receive system from the complex quad-pol
    scattering matrix S = [[Shh, Shv],[Svh, Svv]].

    Transmit a left-circular wave  E_t = [1, +i]/sqrt(2).  The received field is
    E_r = S @ E_t  =>  E_h = (Shh + i*Shv)/sqrt2 ,  E_v = (Svh + i*Svv)/sqrt2.
    Stokes (BSA, linear h/v receive):
      S0 = |E_h|^2 + |E_v|^2
      S1 = |E_h|^2 - |E_v|^2
      S2 =  2 Re(E_h E_v*)
      S3 = -2 Im(E_h E_v*)
    This reproduces the same CPR/DOP/m-chi quantities used throughout HIMADRI.
    """
    inv = 1.0 / np.sqrt(2.0)
    e_h = (shh + 1j * shv) * inv
    e_v = (svh + 1j * svv) * inv
    ih = np.abs(e_h) ** 2
    iv = np.abs(e_v) ** 2
    cross = e_h * np.conj(e_v)
    s0 = (ih + iv).astype(np.float32)
    s1 = (ih - iv).astype(np.float32)
    s2 = (2 * np.real(cross)).astype(np.float32)
    s3 = (-2 * np.imag(cross)).astype(np.float32)
    s0 = np.maximum(s0, 1e-6)
    return np.stack([s0, s1, s2, s3], axis=0)


def _bands_to_stokes(data: np.ndarray, band: str) -> np.ndarray:
    """Interpret a multi-band DFSAR raster as Stokes (4,H,W).

    Supported layouts (auto-detected by band count / dtype):
      * 4 real bands              -> already S0,S1,S2,S3.
      * 4 complex bands           -> Shh,Shv,Svh,Svv scattering matrix.
      * 8 real bands              -> [Re,Im] x (Shh,Shv,Svh,Svv).
      * 2 bands (compact/hybrid)  -> treated as a partial Stokes pair (S0, S3-like);
                                     S1,S2 set to 0 (DOP then reflects circular only).
    TODO[VERIFY]: confirm the supplied DFSAR product's exact band order/convention
    at the event and adjust the mapping here. Everything downstream is unchanged.
    """
    n = data.shape[0]
    if np.iscomplexobj(data):
        if n == 4:
            shh, shv, svh, svv = data
            return stokes_from_scattering_matrix(shh, shv, svh, svv, band)
        raise ValueError(f"DFSAR {band}: complex data with {n} bands not understood")
    if n == 4:
        out = data.astype(np.float32)
        out[0] = np.maximum(out[0], 1e-6)
        return out
    if n == 8:
        shh = data[0] + 1j * data[1]
        shv = data[2] + 1j * data[3]
        svh = data[4] + 1j * data[5]
        svv = data[6] + 1j * data[7]
        return stokes_from_scattering_matrix(shh, shv, svh, svv, band)
    if n == 2:  # compact / hybrid: child Stokes (S0, circular term)
        s0 = np.maximum(data[0].astype(np.float32), 1e-6)
        s3 = data[1].astype(np.float32)
        zero = np.zeros_like(s0)
        return np.stack([s0, zero, zero, s3], axis=0)
    raise ValueError(f"DFSAR {band}: unsupported band count {n}; "
                     "implement the channel->Stokes adapter for this format.")


def load_dfsar(path: str, band: Literal["L", "S"],
               grid: Grid | None = None) -> RadarProduct:
    """Load a DFSAR product as full Stokes (4,H,W), optionally warped to `grid`.

    Auto-detects the real PRADAN L1A SLC product (zip/dir with sli cp_lh/cp_lv
    complex channels) and delegates to the SLC adapter. Otherwise reads a
    GeoTIFF / PDS raster directly and builds Stokes from its bands.
    """
    from .dfsar_slc import is_dfsar_slc_product, load_dfsar_slc

    if is_dfsar_slc_product(path):
        return load_dfsar_slc(path, band=band, grid=grid)
    with rasterio.open(path) as src:
        native_grid = _grid_from_open(src)
        if grid is None:
            data = src.read()
        else:
            # warp each band onto the grid (handles complex via real/imag split)
            data = warp_to_grid(path, grid, "bilinear")
            native_grid = grid
    stokes = _bands_to_stokes(np.asarray(data), band)
    return RadarProduct(grid=native_grid, stokes=stokes, band=band,
                        meta={"source": path})


# --------------------------------------------------------------------------- #
#  OHRC                                                                       #
# --------------------------------------------------------------------------- #

def load_ohrc(path: str, grid: Grid | None = None) -> OpticalProduct:
    """Load OHRC optical imagery (GeoTIFF, or PDS .img/.xml via GDAL)."""
    if grid is None:
        with rasterio.open(path) as src:
            native_grid = _grid_from_open(src)
            refl = src.read(1).astype(np.float32)
            if src.nodata is not None:
                refl = np.where(refl == src.nodata, np.nan, refl)
    else:
        refl = warp_to_grid(path, grid, "bilinear", indexes=1)
        native_grid = grid
    # normalise DN to ~[0,1] reflectance proxy if it looks like raw integers
    finite = np.isfinite(refl)
    if finite.any() and np.nanmax(refl[finite]) > 1.5:
        refl = refl / np.nanpercentile(refl[finite], 99.5)
    return OpticalProduct(grid=native_grid, reflectance=refl, meta={"source": path})


# --------------------------------------------------------------------------- #
#  DEM  (LOLA polar tiles, possibly multi-GB)                                 #
# --------------------------------------------------------------------------- #

def load_dem(path: str | None, grid: Grid | None = None,
             max_dim: int = 512) -> DEMProduct | None:
    """Load a DEM. None -> caller uses the synthetic fallback.

    If `grid` is None the analysis grid is derived from the DEM footprint
    (downsampled so the longer side <= max_dim), then the DEM is warped onto it.
    """
    if path is None:
        return None
    target = grid or grid_from_dataset(path, max_dim=max_dim)
    elev = warp_to_grid(path, target, "bilinear", indexes=1)
    return DEMProduct(grid=target, elevation=elev.astype(np.float32))


def load_sun_geometry(path: str | None) -> SunGeometry | None:
    """Load a solar azimuth/elevation time series (CSV: az_deg,el_deg).
    None -> caller uses the analytic fallback.

    TODO[VERIFY]: for precise illumination swap in spiceypy + NAIF kernels.
    """
    if path is None:
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return SunGeometry(azimuth_deg=arr[:, 0], elevation_deg=arr[:, 1])
