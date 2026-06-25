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
        typer.echo("Traverse:   no route found (check landing/goal connectivity)")


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
