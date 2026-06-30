"""End-to-end orchestration: raw inputs -> all five outputs with one call."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from .config import Config
from .detect import classifier, fusion, pseudolabels, segmentation, uncertainty
from .features import depth, optical, polarimetry
from .illumination import psr
from .io import loaders, writers
from .landing.site_selection import select_landing
from .planning import cost_surface, energy, planner
from .preprocess import coregister, masks, speckle, terrain
from .synth.generate import generate_scene
from .types import (
    FeatureStack,
    IceResult,
    OpticalProduct,
    RadarProduct,
    Scene,
    SunGeometry,
)
from .validate import metrics as metrics_mod
from .viz import maps, report
from .volume.dielectric import estimate_volume


def default_sun(n: int = 24, elevation_deg: float = 1.5) -> SunGeometry:
    """Analytic low-polar-sun sweep used when no SPICE/ephemeris is supplied."""
    return SunGeometry(azimuth_deg=np.linspace(0, 360, n, endpoint=False),
                       elevation_deg=np.full(n, elevation_deg))


def build_real_scene(cfg: Config) -> Scene:
    """Assemble a Scene from real product files (DEM is the geometric backbone).

    The analysis grid is taken from the DEM footprint (downsampled to keep
    multi-GB polar tiles tractable); DFSAR and OHRC are warped onto it. If only
    L-band DFSAR is supplied, S-band falls back to a copy of L so the pipeline
    still runs — the L/S depth feature is then neutral (see make_pseudolabels).
    """
    if cfg.paths.dem is None:
        raise ValueError("real mode requires a DEM (LOLA polar tile); none supplied")
    logger.info("loading real products (DEM defines the common grid)")

    if cfg.grid.center_lon is not None and cfg.grid.center_lat is not None:
        # crater-focused grid: crop the big polar tile to a square ROI
        from .grid import grid_centered_on_lonlat
        grid = grid_centered_on_lonlat(cfg.grid.center_lon, cfg.grid.center_lat,
                                       cfg.grid.half_width_m, cfg.grid.res_m)
        dem = loaders.load_dem(cfg.paths.dem, grid=grid)
        logger.info(f"crater grid @ ({cfg.grid.center_lon},{cfg.grid.center_lat}): "
                    f"{grid.width}x{grid.height} @ {grid.res_m:.0f} m/px")
    else:
        max_dim = max(cfg.grid.height, cfg.grid.width)
        dem = loaders.load_dem(cfg.paths.dem, grid=None, max_dim=max_dim)
        grid = dem.grid
        logger.info(f"common grid: {grid.width}x{grid.height} @ {grid.res_m:.1f} m/px")

    if cfg.paths.dfsar_l is None:
        raise ValueError("real mode requires at least L-band DFSAR; none supplied")
    radar_l = loaders.load_dfsar(cfg.paths.dfsar_l, "L", grid=grid)
    if cfg.paths.dfsar_s:
        radar_s = loaders.load_dfsar(cfg.paths.dfsar_s, "S", grid=grid)
    else:
        logger.warning("no S-band DFSAR -> L/S depth feature neutral (L-only mode)")
        radar_s = RadarProduct(grid=grid, stokes=radar_l.stokes.copy(), band="S",
                               meta={"l_only_fallback": True})

    if cfg.paths.ohrc:
        opt = loaders.load_ohrc(cfg.paths.ohrc, grid=grid)
    else:
        logger.warning("no OHRC -> optical roughness from DEM only")
        opt = OpticalProduct(grid=grid, reflectance=np.full(grid.shape, np.nan,
                                                            dtype=np.float32))
    # track where the radar swath actually has data (real geocoded products
    # only cover part of the crater grid); used to mask detection downstream.
    valid = np.isfinite(radar_l.stokes[0]) & (np.nan_to_num(radar_l.stokes[0]) > 1e-9)
    radar_l.meta["valid"] = valid
    cover = float(valid.mean())
    logger.info(f"radar swath covers {cover*100:.0f}% of the crater grid")

    sun = loaders.load_sun_geometry(cfg.paths.sun_geometry) or default_sun()
    return Scene(grid=grid, radar_l=radar_l, radar_s=radar_s, optical=opt,
                 dem=dem, sun=sun, truth={})


def load_or_synthesize(cfg: Config) -> Scene:
    if cfg.mode == "synthetic" or cfg.paths.dfsar_l is None:
        if cfg.mode != "synthetic":
            logger.warning("real paths incomplete -> falling back to synthetic mode")
            cfg.mode = "synthetic"
        logger.info("generating synthetic doubly-shadowed crater scene")
        return generate_scene(cfg)
    return build_real_scene(cfg)


def preprocess(scene: Scene, cfg: Config) -> Scene:
    logger.info("preprocess: speckle filter + terrain context")
    scene.radar_l.stokes = speckle.refined_lee(scene.radar_l.stokes)
    scene.radar_s.stokes = speckle.refined_lee(scene.radar_s.stokes)
    return scene


def build_features(scene: Scene, cfg: Config) -> FeatureStack:
    logger.info("features: polarimetry + optical + depth + illumination")
    bands: dict[str, np.ndarray] = {}
    bands.update(polarimetry.polarimetric_stack(scene.radar_l, scene.radar_s))
    bands.update(optical.optical_stack(scene.dem, scene.optical, scene.grid.res_m))
    bands["ls_ratio"] = depth.ls_ratio(scene.radar_l, scene.radar_s)
    bands.update(psr.illumination_stack(scene.dem, scene.sun))

    # terrain correction: incidence angle + CPR normalisation + geometry mask
    inc = terrain.incidence_angle(scene.dem.elevation, scene.grid.res_m)
    bands["incidence_L"] = inc
    bands["incidence_S"] = inc
    bands["cpr_L"] = terrain.normalize_cpr(bands["cpr_L"], inc)
    bands["cpr_S"] = terrain.normalize_cpr(bands["cpr_S"], inc)
    bands["geometry_mask"] = masks.layover_shadow(
        scene.dem.elevation, scene.grid.res_m, inc).astype(np.float32)
    valid = scene.radar_l.meta.get("valid")
    bands["radar_valid"] = (valid.astype(np.float32) if valid is not None
                            else np.ones(scene.grid.shape, np.float32))
    return FeatureStack(grid=scene.grid, bands=bands)


def detect(feats: FeatureStack, cfg: Config) -> tuple[IceResult, dict, str]:
    logger.info("detect: pseudo-labels -> classifier -> fusion -> uncertainty")
    labels, weights = pseudolabels.make_pseudolabels(feats, cfg)
    radar_prob, importance, backend = classifier.train_predict(feats, labels, weights, cfg)

    if cfg.detection.use_unet and segmentation.available():
        logger.info("U-Net spatial refinement enabled")
        radar_prob, _ = segmentation.refine_with_unet(radar_prob, feats, labels, cfg)

    rock_prob = fusion.optical_rock_probability(feats)
    posterior = fusion.bayesian_fusion(radar_prob, rock_prob)
    # restrict to PSR interior (ice only survives in shadow) and to where the
    # radar swath actually has data.
    posterior = posterior * (feats.bands["psr_mask"] > 0.5)
    if "radar_valid" in feats.bands:
        posterior = posterior * (feats.bands["radar_valid"] > 0.5)

    unc = uncertainty.total_uncertainty(
        posterior, radar_prob, rock_prob, feats.bands["geometry_mask"])
    target = (posterior > cfg.detection.target_prob) & (unc < 0.6)

    ice = IceResult(grid=feats.grid, probability=posterior.astype(np.float32),
                    uncertainty=unc.astype(np.float32), target_mask=target,
                    meta={"backend": backend})
    return ice, importance, backend


def export(scene, feats, ice, vol, suitability, sites, route_info, energy_prof,
           metrics, importance, backend, cfg) -> dict:
    outdir = Path(cfg.paths.outdir)
    logger.info(f"exporting outputs -> {outdir}")
    grid = scene.grid

    writers.write_geotiff(outdir / "ice_probability.tif", ice.probability, grid)
    writers.write_geotiff(outdir / "ice_uncertainty.tif", ice.uncertainty, grid)
    writers.write_geotiff(outdir / "ice_target_mask.tif",
                          ice.target_mask.astype(np.float32), grid)
    writers.write_geotiff(outdir / "landing_suitability.tif", suitability, grid)
    feat_dir = outdir / "features"
    for name in ("cpr_L", "dop_L", "mchi_volume_L", "ls_ratio", "roughness",
                 "slope_deg", "illumination_frac", "psr_mask", "doubly_shadowed_mask",
                 "radar_valid", "s0_L"):
        if name in feats.bands:
            writers.write_geotiff(feat_dir / f"{name}.tif", feats.bands[name], grid)

    vol_obj = {
        "total_m3": vol.total_m3, "lower_m3": vol.lower_m3, "upper_m3": vol.upper_m3,
        "mean_ice_fraction": vol.mean_ice_fraction, "per_depth_bin": vol.per_depth_bin,
        "meta": vol.meta,
    }
    writers.write_json(outdir / "volume_report.json", vol_obj)

    if len(sites):
        writers.write_vector(sites, outdir / "landing_sites.gpkg", layer="landing_sites")

    route_paths = export_traverse(route_info, energy_prof, grid, outdir)
    # persist the route + energy profile as JSON so the API/UI can replay it
    route_rc = [[int(r), int(c)] for r, c in route_info.get("route", [])]
    writers.write_json(outdir / "traverse.json",
                       {"route": route_rc, "energy": energy_prof})

    figs = maps.make_maps(feats, ice, suitability, sites,
                          route_info.get("route", []), vol, scene.truth, outdir)
    imp_sorted = sorted(importance.items(), key=lambda kv: -kv[1])[:12] if importance else None
    report.render_report({
        "mode": cfg.mode, "depth_m": cfg.volume.depth_m, "vol": vol,
        "metrics": metrics, "energy": energy_prof, "figs": figs,
        "importance": imp_sorted, "backend": backend,
    }, outdir)

    if metrics is not None:
        writers.write_json(outdir / "metrics.json", metrics)

    return {
        "outdir": str(outdir),
        "ice_probability": str(outdir / "ice_probability.tif"),
        "volume_report": str(outdir / "volume_report.json"),
        "landing_sites": str(outdir / "landing_sites.gpkg"),
        "traverse": route_paths,
        "report": str(outdir / "report.html"),
        "metrics": str(outdir / "metrics.json") if metrics is not None else None,
    }


def export_traverse(route_info, energy_prof, grid, outdir):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    route = route_info.get("route", [])
    if len(route) < 2:
        return None
    coords = [grid.rc_to_xy(r, c) for r, c in route]
    line = gpd.GeoDataFrame(
        [{"type": "traverse", "distance_m": energy_prof.get("total_distance_m"),
          "feasible": energy_prof.get("feasible"), "geometry": LineString(coords)}],
        crs=grid.crs)
    path = writers.write_vector(line, outdir / "traverse.gpkg", layer="route")
    # waypoints
    wps = energy_prof.get("waypoints", [])
    if wps:
        wp_gdf = gpd.GeoDataFrame(
            [{**w, "geometry": Point(w["x"], w["y"])} for w in wps[::5]],
            crs=grid.crs)
        writers.write_vector(wp_gdf, outdir / "traverse_waypoints.gpkg", layer="waypoints")
    return str(path)


def run_pipeline(cfg: Config) -> dict:
    np.random.seed(cfg.seed)
    scene = load_or_synthesize(cfg)
    scene = preprocess(scene, cfg)
    feats = build_features(scene, cfg)
    ice, importance, backend = detect(feats, cfg)
    vol = estimate_volume(ice, feats, cfg)

    logger.info("landing-site selection")
    suitability, sites = select_landing(feats, ice, cfg)

    cost = cost_surface.build_cost_surface(feats, cfg)
    start = _pick_start(sites, feats)
    goal = _pick_reachable_goal(ice, feats, cost, start)
    if goal is None:
        logger.warning("no doubly-shadowed floor / ice target found -> "
                       "skipping traverse (no meaningful destination)")
        route_info = {"route": []}
        energy_prof = {"feasible": False, "reason": "no ice target detected",
                       "waypoints": []}
    else:
        logger.info("rover traverse planning (dash-and-return)")
        route_info = planner.dash_and_return(
            cost, feats.bands["illumination_frac"], start, goal, cfg)
        energy_prof = energy.energy_profile(
            route_info.get("route", []), cost, feats.bands["illumination_frac"],
            feats.bands["slope_deg"], feats.grid, cfg)

    metrics = None
    if cfg.mode == "synthetic" and scene.truth:
        logger.info("validating against synthetic ground truth")
        metrics = metrics_mod.evaluate(
            ice.probability, ice.target_mask, scene.truth, feats, cfg,
            vol_est=vol, vol_truth=scene.truth.get("volume_m3"))

    out = export(scene, feats, ice, vol, suitability, sites, route_info,
                 energy_prof, metrics, importance, backend, cfg)
    out["volume"] = {"total_m3": vol.total_m3, "interval": [vol.lower_m3, vol.upper_m3]}
    out["energy"] = energy_prof
    if metrics:
        out["metrics_summary"] = {
            "roc_auc": metrics["roc_auc"], "iou": metrics["iou"],
            "rim_fp_reduction": metrics["rim_fp_reduction_fraction"],
        }
    return out


def _pick_start(sites, feats, goal=None):
    """Land at the ranked site that gives the shortest, safest dash to the
    target — i.e. the candidate nearest the goal (a real mission lands on the
    rim closest to the crater it must reach)."""
    if len(sites):
        if goal is not None:
            d = [(r["row"] - goal[0]) ** 2 + (r["col"] - goal[1]) ** 2
                 for _, r in sites.iterrows()]
            s = sites.iloc[int(np.argmin(d))]
        else:
            s = sites.iloc[0]
        return (int(s["row"]), int(s["col"]))
    score = feats.bands["illumination_frac"] - (feats.bands["slope_deg"] / 90.0)
    return tuple(np.unravel_index(np.argmax(score), score.shape))


def _pick_reachable_goal(ice: IceResult, feats: FeatureStack, cost, start):
    """The traverse destination, constrained to terrain the rover can actually
    reach. The deepest doubly-shadowed floor is often ringed by un-traversable
    (>~40°) walls, so we target the COLDEST point reachable from the landing
    site — the accessible cold-trap margin. Priority: a confident ice target →
    the doubly-shadowed floor → the wider PSR → the coldest reachable ground.
    Guarantees a feasible route while staying scientifically on-target."""
    from scipy import ndimage

    INF = 1e9
    passable = cost < INF
    if not passable[start]:
        # snap start to the nearest passable cell
        d, (ir, ic) = ndimage.distance_transform_edt(~passable, return_indices=True)
        start = (int(ir[start]), int(ic[start]))
    lbl, _ = ndimage.label(passable)
    comp = (lbl == lbl[start]) if lbl[start] > 0 else passable
    illum = feats.bands["illumination_frac"]

    candidates = []
    if ice.target_mask.any():
        candidates.append(comp & ice.target_mask)
    for key in ("doubly_shadowed_mask", "psr_mask"):
        m = feats.bands.get(key)
        if m is not None:
            candidates.append(comp & (m > 0.5))
    candidates.append(comp)  # last resort: coldest reachable ground
    for sel in candidates:
        if sel.any():
            cand = np.where(sel, illum, np.inf)  # coldest (least-lit) reachable
            return tuple(int(v) for v in np.unravel_index(np.argmin(cand), cand.shape))
    return None


def _pick_goal(ice: IceResult, feats: FeatureStack | None = None, min_prob: float = 0.3):
    """The traverse destination. PS-8's objective is to *reach the
    doubly-shadowed crater floor* to access subsurface ice — so we target the
    floor itself (the coldest, most ice-favourable ground), independent of
    whether this particular radar swath detected ice. If high-confidence ice
    exists, we aim for it directly."""
    if ice.target_mask.any():
        ys, xs = np.where(ice.target_mask)
        return (int(np.median(ys)), int(np.median(xs)))
    if feats is not None:
        for key in ("doubly_shadowed_mask", "psr_mask"):
            mask = feats.bands.get(key)
            if mask is not None and (mask > 0.5).any():
                # the flattest (most reachable) point on the cold floor
                slope = feats.bands.get("slope_deg")
                cand = np.where(mask > 0.5, slope if slope is not None else 0.0, np.inf)
                return tuple(int(v) for v in np.unravel_index(np.argmin(cand), cand.shape))
    if float(np.nanmax(ice.probability)) >= min_prob:
        return tuple(int(v) for v in np.unravel_index(np.nanargmax(ice.probability), ice.probability.shape))
    return None
