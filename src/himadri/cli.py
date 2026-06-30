"""HIMADRI command-line interface (typer)."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

from .config import Config

app = typer.Typer(add_completion=False, help="HIMADRI — lunar subsurface-ice "
                  "detection, landing & rover traverse (ISRO BAH 2026 PS-8).")

DEFAULT_CFG = "config/config.yaml"


def _load(config: str | None) -> Config:
    path = config if config and Path(config).exists() else None
    if path is None and config:
        logger.warning(f"config {config} not found; using built-in defaults")
    return Config.load(path)


@app.command()
def synth(out: str = typer.Option("data/synthetic", help="output dir"),
          config: str = typer.Option(DEFAULT_CFG)):
    """Generate a synthetic doubly-shadowed crater scene (+ ground truth)."""
    from .io import writers
    from .synth.generate import generate_scene

    cfg = _load(config)
    scene = generate_scene(cfg)
    out_p = Path(out)
    writers.write_geotiff(out_p / "dfsar_L_stokes.tif", scene.radar_l.stokes, scene.grid)
    writers.write_geotiff(out_p / "dfsar_S_stokes.tif", scene.radar_s.stokes, scene.grid)
    writers.write_geotiff(out_p / "ohrc.tif", scene.optical.reflectance, scene.grid)
    writers.write_geotiff(out_p / "dem.tif", scene.dem.elevation, scene.grid)
    writers.write_geotiff(out_p / "truth_ice_mask.tif",
                          scene.truth["ice_mask"].astype("float32"), scene.grid)
    writers.write_geotiff(out_p / "truth_ice_fraction.tif",
                          scene.truth["ice_fraction"], scene.grid)
    logger.info(f"synthetic scene written to {out_p} "
                f"(truth volume {scene.truth['volume_m3']:,.0f} m³)")


@app.command()
def run(config: str = typer.Option(DEFAULT_CFG),
        mode: str = typer.Option(None, help="override mode: synthetic|real"),
        outdir: str = typer.Option(None, help="override output dir")):
    """Run the full pipeline end-to-end (default mode: synthetic)."""
    from .pipeline import run_pipeline

    cfg = _load(config)
    if mode:
        cfg.mode = mode
    if outdir:
        cfg.paths.outdir = outdir
    out = run_pipeline(cfg)
    logger.info("=== DONE ===")
    typer.echo(f"\nOutputs in: {out['outdir']}")
    typer.echo(f"Report:     {out['report']}")
    typer.echo(f"Volume:     {out['volume']['total_m3']:,.0f} m³ "
               f"[{out['volume']['interval'][0]:,.0f}, {out['volume']['interval'][1]:,.0f}]")
    if out.get("metrics_summary"):
        m = out["metrics_summary"]
        typer.echo(f"ROC-AUC:    {m['roc_auc']:.3f} · IoU {m['iou']:.2f} · "
                   f"rim-FP reduction {m['rim_fp_reduction']*100:.0f}%")
    e = out.get("energy") or {}
    if e.get("total_distance_m") is not None:
        typer.echo(f"Traverse:   {'FEASIBLE' if e.get('feasible') else 'CHECK'} · "
                   f"{e['total_distance_m']} m · peak {e['peak_energy_Wh']} Wh · "
                   f"dark {e['dark_time_min']} min")
    else:
        typer.echo(f"Traverse:   {e.get('reason', 'no route found')}")


@app.command()
def features(config: str = typer.Option(DEFAULT_CFG)):
    """Build and persist the feature stack only."""
    from .io import writers
    from .pipeline import build_features, load_or_synthesize, preprocess

    cfg = _load(config)
    scene = preprocess(load_or_synthesize(cfg), cfg)
    feats = build_features(scene, cfg)
    out = Path(cfg.paths.outdir) / "features"
    for name, arr in feats.bands.items():
        writers.write_geotiff(out / f"{name}.tif", arr, feats.grid)
    logger.info(f"{len(feats.bands)} feature bands -> {out}")


@app.command(name="fetch-dem")
def fetch_dem(
    crater: str = typer.Option("faustini", help="named crater (faustini, shackleton, "
                               "de_gerlache, cabeus, haworth, shoemaker, nobile)"),
    out: str = typer.Option("data/raw", help="download directory"),
    crop: bool = typer.Option(True, help="crop to the crater ROI -> small GeoTIFF"),
    res_m: float = typer.Option(10.0, help="output resolution for the crop"),
    download: bool = typer.Option(True, help="set --no-download to only print URLs"),
):
    """Fetch a public LOLA polar DEM for a crater and crop it to a HIMADRI grid.

    The DFSAR/OHRC mission products need an ISSDC PRADAN account (see DATA.md);
    only the DEM is scriptable. Example:  himadri fetch-dem --crater faustini
    """
    from .io import fetch

    crater = crater.lower()
    if crater not in fetch.CRATERS:
        logger.error(f"unknown crater '{crater}'. choices: {list(fetch.CRATERS)}")
        raise typer.Exit(1)
    info = fetch.CRATERS[crater]
    tile = info["tile"]
    img_url, lbl_url = fetch.tile_urls(tile)
    res, cov = fetch.LOLA_TILES[tile]
    typer.echo(f"Crater {crater}: lon={info['lon']} lat={info['lat']} "
               f"radius~{info['radius_m']/1000:.0f} km")
    typer.echo(f"Recommended LOLA tile: {tile}  ({res:.0f} m/px, {cov})")
    typer.echo(f"  IMG: {img_url}")
    typer.echo(f"  LBL: {lbl_url}")
    if not download:
        typer.echo("\n(--no-download) Set config paths.dem to the downloaded .LBL "
                   "or cropped .tif and run: himadri run --mode real")
        return
    logger.info(f"downloading {tile} (~1.7-1.9 GB; resumable)…")
    lbl = fetch.download_tile(tile, out)
    result = str(lbl)
    if crop:
        out_tif = Path(out) / f"dem_{crater}_{int(res_m)}m.tif"
        logger.info(f"cropping to {crater} ROI -> {out_tif}")
        result = str(fetch.crop_to_crater(lbl, info, res_m, out_tif))
    typer.echo(f"\nDEM ready: {result}")
    typer.echo(f"Set in config.yaml:  paths.dem: {result}")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000),
          reload: bool = typer.Option(False)):
    """Launch the FastAPI backend + React frontend (single server).

    Serves the built React SPA (web/dist) and the /api endpoints. Build the
    frontend first with:  cd web && npm install && npm run build
    """
    try:
        import uvicorn  # noqa: F401
    except Exception:
        logger.error("uvicorn not installed: pip install 'fastapi[all]' uvicorn")
        raise typer.Exit(1)
    import uvicorn

    logger.info(f"HIMADRI API + UI on http://{host}:{port}")
    uvicorn.run("himadri.api:app", host=host, port=port, reload=reload)


@app.command(name="dfsar-quicklook")
def dfsar_quicklook(
    product: str = typer.Argument(..., help="path to the DFSAR SLC zip or extracted dir"),
    out: str = typer.Option("data/outputs/real", help="output dir"),
    looks: int = typer.Option(32, help="multilook factor (matches geometry grid)"),
):
    """Process a REAL DFSAR L1A SLC into geocoded CPR/DOP/m-χ products + a PNG.

    Works with no DEM: geocodes onto an auto polar-stereographic grid from the
    product's own geometry. Example:
      himadri dfsar-quicklook ch2_sar_ncxl_...zip
    """
    import numpy as np

    from .features import polarimetry as pol
    from .io import writers
    from .io.dfsar_slc import load_dfsar_slc

    radar = load_dfsar_slc(product, band="L", looks=looks)
    grid = radar.grid
    st = radar.stokes
    bands = {
        "cpr_L": pol.cpr(st), "dop_L": pol.dop(st), "s0_L": st[0],
        "mchi_volume_frac_L": pol.volume_fraction(st),
    }
    outdir = Path(out)
    for name, arr in bands.items():
        writers.write_geotiff(outdir / f"{name}.tif", arr, grid)
    logger.info(f"geocoded polarimetric products -> {outdir}")

    # quicklook PNG (CPR with ice-like pixels highlighted)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cpr = bands["cpr_L"]; dop = bands["dop_L"]
        valid = np.isfinite(cpr)
        ice_like = valid & (cpr > 1.0) & (dop < 0.4)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
        im0 = ax[0].imshow(np.where(valid, cpr, np.nan), cmap="Spectral_r", vmin=0, vmax=1.6)
        ax[0].set_title("CPR (real DFSAR, geocoded)"); plt.colorbar(im0, ax=ax[0], shrink=0.7)
        ax[1].imshow(np.where(valid, dop, np.nan), cmap="PuBuGn", vmin=0, vmax=1)
        ax[1].imshow(np.where(ice_like, 1.0, np.nan), cmap="autumn", alpha=0.9)
        ax[1].set_title(f"DOP + ice-like (CPR>1 & DOP<0.4): {int(ice_like.sum())} px")
        for a in ax: a.axis("off")
        fig.tight_layout(); fig.savefig(outdir / "quicklook.png", dpi=110); plt.close(fig)
        logger.info(f"quicklook -> {outdir/'quicklook.png'}")
    except Exception as e:
        logger.warning(f"PNG quicklook skipped: {e}")

    n = int(np.isfinite(bands['cpr_L']).sum())
    typer.echo(f"\nGeocoded grid: {grid.width}x{grid.height} @ {grid.res_m:.0f} m/px")
    typer.echo(f"Valid pixels: {n:,}")
    typer.echo(f"Outputs in: {outdir}")


@app.command()
def dashboard(config: str = typer.Option(DEFAULT_CFG)):
    """Launch the optional Streamlit dashboard (if installed)."""
    try:
        import streamlit  # noqa: F401
    except Exception:
        logger.error("streamlit not installed: pip install streamlit")
        raise typer.Exit(1)
    import subprocess

    dash = Path(__file__).parent / "viz" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dash), "--",
                    "--config", config])


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {message}",
               level="INFO")
    app()


if __name__ == "__main__":
    main()
