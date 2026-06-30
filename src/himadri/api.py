"""HIMADRI FastAPI backend.

React renders; Python does all the science. This wraps the existing pipeline
(`himadri.pipeline` + detect/volume/landing/planning) behind a small REST API:

  POST /api/run            run the pipeline for a set of knobs -> summary JSON
  GET  /api/layer          render a result/feature raster as a PNG (server-side)
  GET  /api/scatter        downsampled CPR-vs-DOP points for the disambiguation plot
  POST /api/upload         stage real DFSAR/OHRC/DEM files for a real-data run
  GET  /api/health

Computed runs are cached in-memory keyed by their parameters, so the React UI
stays responsive when only display knobs change.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline as P
from .config import Config
from .detect import classifier, fusion, pseudolabels, uncertainty
from .landing.site_selection import select_landing
from .planning import cost_surface, energy as energy_mod, planner
from .synth.generate import generate_scene
from .types import FeatureStack
from .validate import metrics as M
from .volume.dielectric import estimate_volume

app = FastAPI(title="HIMADRI API", version="1.0.0")

# ---- in-memory caches ------------------------------------------------------
_BASE: dict = {}     # base scene+feats keyed by (source,seed,grid,real-sig)
_RUNS: dict = {}     # full run arrays keyed by run_id

# Layer -> (matplotlib colormap, label). Light/scientific palette.
LAYERS = {
    "ice_probability": ("YlGnBu", "Ice probability"),
    "ice_uncertainty": ("OrRd", "Uncertainty"),
    "target_mask": ("BuGn", "High-confidence target"),
    "cpr_L": ("Spectral_r", "CPR (L-band)"),
    "dop_L": ("PuBuGn", "DOP (L-band)"),
    "mchi_volume_L": ("viridis", "m-χ volume power (L)"),
    "ls_ratio": ("RdBu_r", "L/S ratio (depth)"),
    "illumination_frac": ("bone", "Illumination fraction"),
    "slope_deg": ("YlOrBr", "Slope (°)"),
    "roughness": ("YlOrRd", "Roughness"),
    "doubly_shadowed_mask": ("Greys", "Doubly-shadowed"),
    "landing_suitability": ("YlGn", "Landing suitability"),
    "s0_L": ("gray", "Radar backscatter S0 (L-band)"),
    "radar_valid": ("Greens", "Radar swath coverage"),
}
LAYER_RANGE = {
    "cpr_L": (0, 2), "dop_L": (0, 1), "ls_ratio": (0.6, 1.6),
    "slope_deg": (0, 45), "illumination_frac": (0, 1),
    "ice_probability": (0, 1), "ice_uncertainty": (0, 1), "target_mask": (0, 1),
}


class RunParams(BaseModel):
    source: str = "synthetic"          # synthetic | real
    seed: int = 42
    grid: int = 224
    cpr_min: float = 1.0
    dop_max: float = 0.13
    fusion_k: float = 4.0
    target_prob: float = 0.6
    unc_max: float = 0.6
    battery_Wh: float = 2000.0
    max_slope_deg: float = 25.0
    proximity_w: float = 0.30
    n_mc: int = 800
    upload_token: Optional[str] = None


def _hash(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _get_base(p: RunParams):
    key = _hash(p.source, p.seed, p.grid, p.upload_token or "")
    if key in _BASE:
        return _BASE[key]
    cfg = Config()
    cfg.seed = p.seed
    cfg.grid.height = cfg.grid.width = p.grid
    if p.source == "real":
        token_dir = Path(tempfile.gettempdir()) / "himadri_api_uploads" / (p.upload_token or "")
        cfg.mode = "real"
        cfg.paths.dem = _find(token_dir, [".lbl", ".tif", ".tiff", ".img"], "dem")
        cfg.paths.dfsar_l = _find(token_dir, [".tif", ".tiff", ".xml", ".img"], "dfsar_l")
        cfg.paths.dfsar_s = _find(token_dir, [".tif", ".tiff", ".xml", ".img"], "dfsar_s")
        cfg.paths.ohrc = _find(token_dir, [".tif", ".tiff", ".xml", ".img"], "ohrc")
        if not cfg.paths.dem or not cfg.paths.dfsar_l:
            raise HTTPException(400, "real run needs at least a DEM and L-band DFSAR")
        scene = P.build_real_scene(cfg)
    else:
        scene = generate_scene(cfg)
    scene = P.preprocess(scene, cfg)
    feats = P.build_features(scene, cfg)
    _BASE[key] = (scene, feats)
    return scene, feats


def _find(d: Path, exts: list[str], prefix: str) -> Optional[str]:
    if not d.exists():
        return None
    files = [f for f in d.iterdir() if f.name.startswith(prefix)]
    for ext in exts:
        for f in files:
            if f.name.lower().endswith(ext):
                return str(f)
    return str(files[0]) if files else None


def _compute(p: RunParams):
    scene, feats = _get_base(p)
    cfg = Config()
    cfg.seed = p.seed
    cfg.detection.cpr_ice_min = p.cpr_min
    cfg.detection.dop_ice_max = p.dop_max
    cfg.volume.n_mc = int(p.n_mc)
    cfg.planning.battery_Wh = float(p.battery_Wh)
    cfg.planning.max_slope_deg = float(p.max_slope_deg)
    cfg.landing.weights["proximity"] = float(p.proximity_w)

    labels, weights = pseudolabels.make_pseudolabels(feats, cfg)
    radar_prob, importance, backend = classifier.train_predict(feats, labels, weights, cfg)
    rock_prob = fusion.optical_rock_probability(feats)

    eps = 1e-4
    pr = np.clip(radar_prob, eps, 1 - eps)
    logit = np.log(pr / (1 - pr)) - p.fusion_k * rock_prob
    posterior = (1 / (1 + np.exp(-logit))).astype(np.float32)
    posterior *= (feats.bands["psr_mask"] > 0.5)
    unc = uncertainty.total_uncertainty(posterior, radar_prob, rock_prob,
                                        feats.bands["geometry_mask"])
    target = (posterior > p.target_prob) & (unc < p.unc_max)

    from .types import IceResult
    ice = IceResult(grid=feats.grid, probability=posterior, uncertainty=unc,
                    target_mask=target)
    res = _plan(scene, feats, ice, cfg)

    grid = feats.grid
    summary = {
        "grid": {"h": grid.height, "w": grid.width, "res_m": grid.res_m},
        "volume": {
            "total_m3": res["vol"].total_m3, "lower_m3": res["vol"].lower_m3,
            "upper_m3": res["vol"].upper_m3, "mean_ice_fraction": res["vol"].mean_ice_fraction,
            "per_depth_bin": res["vol"].per_depth_bin, "meta": res["vol"].meta,
        },
        "energy": res["energy"],
        "route": [[int(r), int(c)] for r, c in res["route"]],
        "sites": res["sites"],
        "metrics": res["metrics"],
        "importance": sorted(importance.items(), key=lambda kv: -kv[1])[:12] if importance else [],
        "backend": backend,
        "target_pixels": int(target.sum()),
        "has_truth": bool(scene.truth),
        "layers": [{"name": n, "label": LAYERS[n][1]} for n in LAYERS],
    }

    arrays = {
        "ice_probability": posterior, "ice_uncertainty": unc,
        "target_mask": target.astype(np.float32),
        "landing_suitability": np.asarray(res["suit"]),
    }
    for n in ("cpr_L", "dop_L", "mchi_volume_L", "ls_ratio", "slope_deg",
              "roughness", "illumination_frac", "doubly_shadowed_mask"):
        if n in feats.bands:
            arrays[n] = feats.bands[n]

    # scatter sample for disambiguation
    cpr = feats.bands["cpr_L"].ravel(); dop = feats.bands["dop_L"].ravel()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cpr), size=min(4000, len(cpr)), replace=False)
    cls = scene.truth["class_map"].ravel()[idx].tolist() if "class_map" in scene.truth else None
    scatter = {
        "cpr": cpr[idx].round(3).tolist(), "dop": dop[idx].round(3).tolist(),
        "prob": posterior.ravel()[idx].round(3).tolist(), "cls": cls,
    }

    run_id = _hash(*p.model_dump().values())
    _RUNS[run_id] = {"arrays": arrays, "scatter": scatter}
    summary["run_id"] = run_id
    return summary


def _plan(scene, feats, ice, cfg):
    vol = estimate_volume(ice, feats, cfg)
    suit, sites = select_landing(feats, ice, cfg)
    cost = cost_surface.build_cost_surface(feats, cfg)
    start = P._pick_start(sites, feats); goal = P._pick_goal(ice)
    route_info = planner.dash_and_return(cost, feats.bands["illumination_frac"], start, goal, cfg)
    eprof = energy_mod.energy_profile(route_info.get("route", []), cost,
                                      feats.bands["illumination_frac"],
                                      feats.bands["slope_deg"], feats.grid, cfg)
    site_rc = [{"row": int(r["row"]), "col": int(r["col"]), "score": float(r["score"])}
               for _, r in sites.iterrows()] if len(sites) else []
    metrics = None
    if scene.truth:
        metrics = M.evaluate(ice.probability, ice.target_mask, scene.truth, feats, cfg,
                             vol_est=vol, vol_truth=scene.truth.get("volume_m3"))
    return {"vol": vol, "suit": np.asarray(suit), "sites": site_rc,
            "route": route_info.get("route", []), "energy": eprof, "metrics": metrics}


def _render_png(arr: np.ndarray, cmap: str, vmin=None, vmax=None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    from PIL import Image

    a = np.array(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if vmin is None:
        vmin = float(np.nanpercentile(a[finite], 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(a[finite], 98)) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = colormaps[cmap](norm(a))
    rgba[..., 3] = np.where(finite, 1.0, 0.0)  # NaN -> transparent
    img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


REAL_RUNS = {"faustini": {"dir": "data/outputs/real_faustini",
                          "lon": 77.0, "lat": -87.2, "name": "Faustini"}}
_GEO = "+proj=longlat +a=1737400 +b=1737400 +no_defs"


def _graticule_labels(transform, w, h, crs_proj4, center_lon, center_lat,
                      name, ds_mask=None):
    """Build a lat/lon graticule (polylines in row/col) + area labels for the
    crater grid, using pyproj for the polar-stereographic <-> lon/lat mapping."""
    import math

    from pyproj import Transformer

    to_geo = Transformer.from_crs(crs_proj4, _GEO, always_xy=True)
    to_xy = Transformer.from_crs(_GEO, crs_proj4, always_xy=True)
    inv = ~transform

    def xy_to_rc(x, y):
        col, row = inv * (x, y)
        return row, col

    # robust lon/lat range from a 3x3 sample of the grid
    los, las = [], []
    for r in (0, h / 2, h):
        for c in (0, w / 2, w):
            x, y = transform * (c, r)
            lo, la = to_geo.transform(x, y)
            los.append(((lo - center_lon + 180) % 360) - 180 + center_lon)
            las.append(la)
    lonmin, lonmax, latmin, latmax = min(los), max(los), min(las), max(las)

    def nice(span):
        for s in (0.1, 0.25, 0.5, 1, 2, 5, 10):
            if span / s <= 7:
                return s
        return 10

    lat_step, lon_step = nice(latmax - latmin), nice(lonmax - lonmin)

    def line(fixed_is_lat, fixed, var_min, var_max):
        pts = []
        for i in range(61):
            v = var_min + (var_max - var_min) * i / 60
            lo, la = (v, fixed) if fixed_is_lat else (fixed, v)
            x, y = to_xy.transform(lo, la)
            r, c = xy_to_rc(x, y)
            if -8 <= c <= w + 8 and -8 <= r <= h + 8:
                pts.append([round(r, 1), round(c, 1)])
        return pts

    lat_lines, lon_lines = [], []
    la = math.ceil(latmin / lat_step) * lat_step
    while la <= latmax:
        pts = line(True, la, lonmin, lonmax)
        if len(pts) >= 2:
            lab = f"{abs(la):.2f}°S" if la < 0 else f"{la:.2f}°N"
            lat_lines.append({"label": lab, "pts": pts})
        la += lat_step
    lo = math.ceil(lonmin / lon_step) * lon_step
    while lo <= lonmax:
        pts = line(False, lo, latmin, latmax)
        if len(pts) >= 2:
            lon_lines.append({"label": f"{lo % 360:.0f}°E", "pts": pts})
        lo += lon_step

    labels = []
    x, y = to_xy.transform(center_lon, center_lat)
    r, c = xy_to_rc(x, y)
    if 0 <= r <= h and 0 <= c <= w:
        labels.append({"text": name, "row": round(r, 1), "col": round(c, 1), "kind": "crater"})
    if ds_mask is not None and (ds_mask > 0.5).any():
        ys, xs = np.where(ds_mask > 0.5)
        labels.append({"text": "doubly-shadowed floor", "row": float(np.median(ys)),
                       "col": float(np.median(xs)), "kind": "psr"})
    return {"lat": lat_lines, "lon": lon_lines, "labels": labels}


@app.get("/api/real")
def real(run: str = "faustini"):
    """Serve a PRECOMPUTED real run (from `himadri run --mode real`) — no
    recompute. Reads the GeoTIFF layers + volume report from disk."""
    import json

    info = REAL_RUNS.get(run)
    if not info:
        raise HTTPException(404, f"unknown real run '{run}'")
    d = Path(info["dir"])
    if not d.exists():
        raise HTTPException(404, f"no precomputed real run at {d}; run "
                            "`himadri run --config config/real_faustini.yaml` first")
    filemap = {
        "s0_L": "features/s0_L.tif", "cpr_L": "features/cpr_L.tif",
        "dop_L": "features/dop_L.tif", "mchi_volume_L": "features/mchi_volume_L.tif",
        "ice_probability": "ice_probability.tif", "ice_uncertainty": "ice_uncertainty.tif",
        "target_mask": "ice_target_mask.tif", "landing_suitability": "landing_suitability.tif",
        "illumination_frac": "features/illumination_frac.tif",
        "doubly_shadowed_mask": "features/doubly_shadowed_mask.tif",
        "slope_deg": "features/slope_deg.tif", "roughness": "features/roughness.tif",
        "radar_valid": "features/radar_valid.tif",
    }
    arrays, grid = {}, None
    transform = crs_proj4 = None
    for key, rel in filemap.items():
        p = d / rel
        if not p.exists():
            continue
        with rasterio.open(p) as ds:
            arrays[key] = ds.read(1)
            if grid is None:
                grid = {"h": ds.height, "w": ds.width, "res_m": abs(ds.transform.a)}
                transform, crs_proj4 = ds.transform, (ds.crs.to_proj4() if ds.crs else None)

    vol = {}
    if (d / "volume_report.json").exists():
        vol = json.loads((d / "volume_report.json").read_text())

    sites = []
    for cand in ("landing_sites.gpkg", "landing_sites.geojson"):
        if (d / cand).exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(d / cand)
                for _, r in gdf.iterrows():
                    if "row" in r and "col" in r:
                        sites.append({"row": int(r["row"]), "col": int(r["col"]),
                                      "score": float(r.get("score", 0))})
            except Exception:
                pass
            break

    rv = arrays.get("radar_valid")
    coverage = float((rv > 0.5).mean()) if rv is not None else None

    route, energy = [], {"feasible": False, "waypoints": [],
                         "reason": "no traverse computed"}
    if (d / "traverse.json").exists():
        tj = json.loads((d / "traverse.json").read_text())
        route = tj.get("route", [])
        energy = tj.get("energy", energy)

    scatter = {"cpr": [], "dop": [], "prob": [], "cls": None}
    if "cpr_L" in arrays:
        cpr = arrays["cpr_L"].ravel(); dop = arrays["dop_L"].ravel()
        prob = arrays["ice_probability"].ravel()
        valid = np.isfinite(cpr) & (np.abs(cpr) > 1e-9)
        idx = np.where(valid)[0]
        if len(idx) > 4000:
            idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
        scatter = {"cpr": np.nan_to_num(cpr[idx]).round(3).tolist(),
                   "dop": np.nan_to_num(dop[idx]).round(3).tolist(),
                   "prob": np.nan_to_num(prob[idx]).round(3).tolist(), "cls": None}

    graticule = {"lat": [], "lon": [], "labels": []}
    if transform is not None and crs_proj4:
        try:
            graticule = _graticule_labels(
                transform, grid["w"], grid["h"], crs_proj4, info["lon"], info["lat"],
                info["name"], arrays.get("doubly_shadowed_mask"))
        except Exception:
            pass

    run_id = f"real_{run}"
    _RUNS[run_id] = {"arrays": arrays, "scatter": scatter}
    return {
        "graticule": graticule,
        "run_id": run_id, "grid": grid,
        "volume": {"total_m3": vol.get("total_m3", 0), "lower_m3": vol.get("lower_m3", 0),
                   "upper_m3": vol.get("upper_m3", 0),
                   "mean_ice_fraction": vol.get("mean_ice_fraction", 0),
                   "per_depth_bin": vol.get("per_depth_bin", []), "meta": vol.get("meta", {})},
        "energy": energy,
        "route": route, "sites": sites, "metrics": None, "importance": [],
        "backend": "real · precomputed", "has_truth": False,
        "target_pixels": int(np.nansum(arrays.get("target_mask", np.zeros(1)) > 0.5)),
        "coverage": coverage,
        "scene": {"dfsar": "ch2_sar_ncxl_20241005t052503010 · L1A SLC (compact-pol)",
                  "dem": "LOLA LDEM_85S_10M · 10 m", "crater": "Faustini ~87.2°S, 77°E"},
        "layers": [{"name": n, "label": LAYERS.get(n, (None, n))[1]} for n in arrays],
    }


@app.get("/api/validation")
def validation(grid: int = 224, seed: int = 42):
    """Model accuracy & training evidence — computed on the SYNTHETIC crater
    where ground truth exists (no truth exists for a real crater). Returns
    ROC/PR curves, spatially-blocked CV AUC, the CPR-only baseline comparison,
    feature importances, and volume validation incl. the Monte-Carlo histogram.
    Cached after first call."""
    s = _compute(RunParams(source="synthetic", grid=grid, seed=seed, n_mc=1500))
    return {
        "metrics": s["metrics"], "importance": s["importance"],
        "volume": s["volume"], "backend": s["backend"], "grid": s["grid"],
        "target_pixels": s["target_pixels"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/api/run")
def run(params: RunParams):
    return _compute(params)


@app.get("/api/layer")
def layer(run_id: str, name: str):
    if run_id not in _RUNS:
        raise HTTPException(404, "unknown run_id; POST /api/run first")
    arrays = _RUNS[run_id]["arrays"]
    if name not in arrays:
        raise HTTPException(404, f"layer '{name}' not available")
    cmap = LAYERS.get(name, ("viridis", name))[0]
    vmin, vmax = LAYER_RANGE.get(name, (None, None))
    png = _render_png(arrays[name], cmap, vmin, vmax)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=600"})


@app.get("/api/scatter")
def scatter(run_id: str):
    if run_id not in _RUNS:
        raise HTTPException(404, "unknown run_id")
    return _RUNS[run_id]["scatter"]


@app.post("/api/upload")
async def upload(role: str = Form(...), token: str = Form(...),
                 file: UploadFile = File(...)):
    """Stage a real-data file under a session token. role in {dem,dfsar_l,dfsar_s,ohrc}."""
    if role not in ("dem", "dfsar_l", "dfsar_s", "ohrc"):
        raise HTTPException(400, "invalid role")
    d = Path(tempfile.gettempdir()) / "himadri_api_uploads" / token
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{role}__{file.filename}"
    dest.write_bytes(await file.read())
    return {"ok": True, "role": role, "stored": dest.name}


# ---- serve the built React SPA (if present) --------------------------------
_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
