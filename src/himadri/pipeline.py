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
from .types import FeatureStack, IceResult, Scene
from .validate import metrics as metrics_mod
from .viz import maps, report
from .volume.dielectric import estimate_volume


def load_or_synthesize(cfg: Config) -> Scene:
    if cfg.mode == "synthetic" or cfg.paths.dfsar_l is None:
        if cfg.mode != "synthetic":
            logger.warning("real paths incomplete -> falling back to synthetic mode")
            cfg.mode = "synthetic"
        logger.info("generating synthetic doubly-shadowed crater scene")
        return generate_scene(cfg)

    # real mode (best-effort adapter; synthetic remains the default)
    logger.info("loading real DFSAR/OHRC/DEM products")
    from .grid import build_grid

    grid = build_grid(cfg)
    radar_l = coregister.radar_to_grid(loaders.load_dfsar(cfg.paths.dfsar_l, "L"), grid)
    radar_s = coregister.radar_to_grid(loaders.load_dfsar(cfg.paths.dfsar_s, "S"), grid)
    opt = coregister.optical_to_grid(loaders.load_ohrc(cfg.paths.ohrc), grid)
    dem = loaders.load_dem(cfg.paths.dem)
    if dem is None:
        raise ValueError("real mode requires a DEM; none supplied")
    sun = loaders.load_sun_geometry(cfg.paths.sun_geometry) or generate_scene(cfg).sun
    return Scene(grid=grid, radar_l=radar_l, radar_s=radar_s, optical=opt,
                 dem=dem, sun=sun, truth={})


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
    # restrict to PSR interior (ice only survives in shadow)
    posterior = posterior * (feats.bands["psr_mask"] > 0.5)

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
                 "slope_deg", "illumination_frac", "psr_mask", "doubly_shadowed_mask"):
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

    logger.info("rover traverse planning (dash-and-return)")
    cost = cost_surface.build_cost_surface(feats, cfg)
    start = _pick_start(sites, feats)
    goal = _pick_goal(ice)
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


def _pick_start(sites, feats):
    if len(sites):
        s = sites.iloc[0]
        return (int(s["row"]), int(s["col"]))
    # fallback: brightest low-slope pixel
    score = feats.bands["illumination_frac"] - (feats.bands["slope_deg"] / 90.0)
    return tuple(np.unravel_index(np.argmax(score), score.shape))


def _pick_goal(ice: IceResult):
    if ice.target_mask.any():
        ys, xs = np.where(ice.target_mask)
        # centroid of the largest target cluster (most confident region)
        return (int(np.median(ys)), int(np.median(xs)))
    return tuple(np.unravel_index(np.argmax(ice.probability), ice.probability.shape))
