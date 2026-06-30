"""Public ancillary-data fetcher (verified Jun 2026).

Downloads LOLA polar DEM tiles from the LOLA PDS node and (optionally) crops
them to a named crater ROI, producing a small GeoTIFF ready for HIMADRI.

The DFSAR/OHRC mission products require a (free) ISSDC PRADAN account and are
not scriptable here — see DATA.md for the exact PRADAN checklist. The DFSAR
crater scene itself is SUPPLIED by the organisers at the event.

Verified hosts:
  LOLA GDR polar IMG+LBL :  https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/
    LDEM_875S_5M  (5 m,  87.5-90 S, ~1.75 GB)
    LDEM_85S_10M  (10 m, 85-90 S,  ~1.75 GB)
    LDEM_80S_20M  (20 m, 80-90 S,  ~1.85 GB)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

LOLA_BASE = "https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG"

# LOLA polar DEM tiles (name -> resolution m/px, latitude coverage)
LOLA_TILES = {
    "LDEM_875S_5M": (5.0, "87.5-90 S"),
    "LDEM_85S_10M": (10.0, "85-90 S"),
    "LDEM_80S_20M": (20.0, "80-90 S"),
}

# Named south-polar craters: approximate centre (lon_deg_east, lat_deg, radius_m)
# and the recommended LOLA tile that covers them.
CRATERS = {
    "faustini":     {"lon": 75.8,  "lat": -87.18, "radius_m": 21000, "tile": "LDEM_85S_10M"},
    "shackleton":   {"lon": 129.2, "lat": -89.66, "radius_m": 10500, "tile": "LDEM_875S_5M"},
    "de_gerlache":  {"lon": 272.0, "lat": -88.5,  "radius_m": 16500, "tile": "LDEM_875S_5M"},
    "cabeus":       {"lon": 324.5, "lat": -84.9,  "radius_m": 49000, "tile": "LDEM_80S_20M"},
    "haworth":      {"lon": 4.4,   "lat": -87.45, "radius_m": 25500, "tile": "LDEM_85S_10M"},
    "shoemaker":    {"lon": 44.9,  "lat": -88.1,  "radius_m": 25500, "tile": "LDEM_875S_5M"},
    "nobile":       {"lon": 53.5,  "lat": -85.2,  "radius_m": 36500, "tile": "LDEM_80S_20M"},
}


def tile_urls(tile: str) -> tuple[str, str]:
    return f"{LOLA_BASE}/{tile}.IMG", f"{LOLA_BASE}/{tile}.LBL"


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes a partial download; --fail surfaces HTTP errors.
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)


def download_tile(tile: str, outdir: str | Path) -> Path:
    """Download a LOLA polar tile (.IMG + detached .LBL). Returns the .LBL path
    (GDAL reads the detached label for projection/geometry)."""
    if tile not in LOLA_TILES:
        raise ValueError(f"unknown tile {tile}; choose from {list(LOLA_TILES)}")
    outdir = Path(outdir)
    img_url, lbl_url = tile_urls(tile)
    lbl = outdir / f"{tile}.LBL"
    img = outdir / f"{tile}.IMG"
    _curl(lbl_url, lbl)
    _curl(img_url, img)
    return lbl


def crop_to_crater(lbl_path: str | Path, crater: dict, res_m: float | None,
                   out_tif: str | Path) -> Path:
    """Crop/resample a LOLA tile to a crater ROI -> a small GeoTIFF on the
    lunar south-polar stereographic grid HIMADRI expects."""
    from ..grid import grid_centered_on_lonlat, warp_to_grid
    from .writers import write_geotiff

    res = res_m or 10.0
    grid = grid_centered_on_lonlat(crater["lon"], crater["lat"],
                                   crater["radius_m"] * 1.2, res)
    elev = warp_to_grid(str(lbl_path), grid, "bilinear", indexes=1)
    return write_geotiff(out_tif, elev, grid)
