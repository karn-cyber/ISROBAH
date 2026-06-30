"""Real Chandrayaan-2 DFSAR Level-1A SLC adapter.

The supplied product (PRADAN `ncxl` zip/dir) stores, per the DFSAR User Manual:
  * data/.../*_sli_xx_cp_lh_*.tif , *_sli_xx_cp_lv_*.tif  — slant-range
    single-look COMPLEX (ComplexLSB8 -> 2 float32 bands I,Q), the two
    compact-pol receive channels. ONLY these carry phase -> the only data
    usable for CPR/DOP/m-chi.
  * geometry/.../*_g_sli_*.csv — Lat,Lon,SlantRange,Incidence on a regular
    grid sampled every `interval` pixels (from the geometry XML).
  * (SRI map-projected channels are amplitude-only -> no phase -> unused here.)

This module reads the complex channels (block-wise, straight from the zip via
GDAL /vsizip — no 4 GB extraction), multilooks them by the geometry grid
interval to build the Stokes vector, then geocodes the Stokes grid onto a lunar
south-polar stereographic grid using the geometry CSV as control points.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger

from ..types import Grid, RadarProduct
from .writers import _crs  # noqa: F401  (reused CRS helper)


def is_dfsar_slc_product(path: str) -> bool:
    p = Path(path)
    names: list[str] = []
    if p.suffix.lower() == ".zip" and p.exists():
        try:
            names = zipfile.ZipFile(p).namelist()
        except Exception:
            return False
    elif p.is_dir():
        names = [str(f) for f in p.rglob("*")]
    return any("_sli_" in n and "_cp_lh_" in n for n in names)


def _members(path: str) -> dict[str, str]:
    """Locate the SLI lh/lv tifs and g_sli csv; return GDAL-openable paths."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        names = zipfile.ZipFile(p).namelist()
        pref = lambda m: f"/vsizip/{p}/{m}"
    else:
        names = [str(f.relative_to(p)) for f in p.rglob("*")]
        pref = lambda m: str(p / m)
    out = {}
    for n in names:
        if re.search(r"_sli_.*_cp_lh_.*\.tif$", n):
            out["lh"] = pref(n)
        elif re.search(r"_sli_.*_cp_lv_.*\.tif$", n):
            out["lv"] = pref(n)
        elif re.search(r"_g_sli_.*\.csv$", n):
            out["csv"] = pref(n) if p.suffix.lower() != ".zip" else n
            out["csv_zip"] = str(p)
    return out


def _read_geometry_grid(path: str, members: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (lat, lon, n_records, n_samples) for the SLI geometry grid."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        import io
        with zipfile.ZipFile(p) as zf:
            raw = zf.read(members["csv"]).decode()
        arr = np.loadtxt(io.StringIO(raw), delimiter=",", skiprows=1, usecols=(0, 1))
    else:
        arr = np.loadtxt(members["csv"], delimiter=",", skiprows=1, usecols=(0, 1))
    lat, lon = arr[:, 0], arr[:, 1]
    return lat, lon, len(lat), 0


def _multilook_stokes(lh_path: str, lv_path: str, looks: int,
                      block_lines: int = 8192) -> np.ndarray:
    """Block-read the two complex channels and multilook by `looks`x`looks`
    into Stokes (4, nrec, nsamp). Memory-safe (one block at a time)."""
    with rasterio.open(lh_path) as dlh, rasterio.open(lv_path) as dlv:
        H, W = dlh.shape
        nrec, nsamp = H // looks, W // looks
        Wc = nsamp * looks
        acc = {k: np.zeros((nrec, nsamp), np.float64) for k in ("c11", "c22", "re", "im")}
        block = (block_lines // looks) * looks
        row = 0
        for r0 in range(0, nrec * looks, block):
            r1 = min(r0 + block, nrec * looks)
            win = ((r0, r1), (0, Wc))
            Elh = dlh.read(1, window=win).astype(np.float32) + 1j * dlh.read(2, window=win).astype(np.float32)
            Elv = dlv.read(1, window=win).astype(np.float32) + 1j * dlv.read(2, window=win).astype(np.float32)
            nr = (r1 - r0) // looks
            def ml(x):
                return x[: nr * looks].reshape(nr, looks, nsamp, looks).mean(axis=(1, 3))
            acc["c11"][row:row + nr] = ml(np.abs(Elh) ** 2)
            acc["c22"][row:row + nr] = ml(np.abs(Elv) ** 2)
            c12 = ml(Elh * np.conj(Elv))
            acc["re"][row:row + nr] = np.real(c12)
            acc["im"][row:row + nr] = np.imag(c12)
            row += nr
    s0 = np.maximum(acc["c11"] + acc["c22"], 1e-12)
    s1 = acc["c11"] - acc["c22"]
    s2 = 2 * acc["re"]
    s3 = -2 * acc["im"]
    return np.stack([s0, s1, s2, s3], 0).astype(np.float32)


def _auto_grid(x: np.ndarray, y: np.ndarray, max_dim: int = 1600,
               min_res: float = 20.0) -> Grid:
    from affine import Affine
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    span = max(xmax - xmin, ymax - ymin)
    res = max(span / max_dim, min_res)
    w = int(np.ceil((xmax - xmin) / res)) + 1
    h = int(np.ceil((ymax - ymin) / res)) + 1
    transform = Affine(res, 0, xmin, 0, -res, ymax)
    return Grid(transform=transform, height=h, width=w, res_m=res)


def load_dfsar_slc(path: str, band: str = "L", grid: Grid | None = None,
                   looks: int = 32) -> RadarProduct:
    """Read the real DFSAR SLC product -> geocoded Stokes RadarProduct."""
    from pyproj import Transformer
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    from ..types import LUNAR_SOUTH_POLAR_STEREO

    mem = _members(path)
    if "lh" not in mem or "lv" not in mem:
        raise ValueError("DFSAR SLC product missing sli cp_lh/cp_lv channels")
    logger.info("DFSAR SLC: multilooking complex channels (32x32)…")
    stokes_sl = _multilook_stokes(mem["lh"], mem["lv"], looks)  # (4, nrec, nsamp)
    _, nrec, nsamp = stokes_sl.shape

    lat, lon, n, _ = _read_geometry_grid(path, mem)
    # the CSV is a regular (records x samples) grid; infer columns from counts
    ncol = n // nrec if n % nrec == 0 else int(round(n / nrec))
    lat = lat[: nrec * ncol].reshape(nrec, ncol)
    lon = lon[: nrec * ncol].reshape(nrec, ncol)
    c = min(nsamp, ncol)
    lat, lon = lat[:, :c], lon[:, :c]
    stokes_sl = stokes_sl[:, :, :c]

    tr = Transformer.from_crs("+proj=longlat +a=1737400 +b=1737400 +no_defs",
                              LUNAR_SOUTH_POLAR_STEREO, always_xy=True)
    X, Y = tr.transform(lon.ravel(), lat.ravel())
    pts = np.column_stack([X, Y])
    target = grid or _auto_grid(np.array(X), np.array(Y))
    logger.info(f"DFSAR SLC: geocoding -> {target.width}x{target.height} @ {target.res_m:.0f} m/px")

    cols, rows = np.meshgrid(np.arange(target.width), np.arange(target.height))
    gx, gy = target.transform * (cols + 0.5, rows + 0.5)
    tri = Delaunay(pts)
    out = np.full((4, target.height, target.width), np.nan, np.float32)
    for b in range(4):
        interp = LinearNDInterpolator(tri, stokes_sl[b].ravel())
        out[b] = interp(gx, gy).astype(np.float32)
    out[0] = np.where(np.isfinite(out[0]), np.maximum(out[0], 1e-9), np.nan)
    return RadarProduct(grid=target, stokes=out, band=band,
                        meta={"source": str(path), "product": "L1A-SLC", "looks": looks})
