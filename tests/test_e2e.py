import time
from pathlib import Path

from himadri.config import Config
from himadri.pipeline import run_pipeline


def test_end_to_end(tmp_path):
    cfg = Config()
    cfg.grid.height = 192
    cfg.grid.width = 192
    cfg.volume.n_mc = 400
    cfg.paths.outdir = str(tmp_path / "out")

    t0 = time.time()
    out = run_pipeline(cfg)
    elapsed = time.time() - t0

    # all PRD §5 outputs exist
    for key in ("ice_probability", "volume_report", "report"):
        assert Path(out[key]).exists()
    assert Path(out["outdir"], "ice_uncertainty.tif").exists()
    assert Path(out["outdir"], "metrics.json").exists()

    m = out["metrics_summary"]
    assert m["roc_auc"] >= 0.90
    assert m["rim_fp_reduction"] >= 0.70
    assert elapsed < 600  # < 10 min on CPU


def test_determinism(tmp_path):
    cfg = Config()
    cfg.grid.height = 160
    cfg.grid.width = 160
    cfg.volume.n_mc = 300
    cfg.paths.outdir = str(tmp_path / "a")
    o1 = run_pipeline(cfg)
    cfg.paths.outdir = str(tmp_path / "b")
    o2 = run_pipeline(cfg)
    assert abs(o1["volume"]["total_m3"] - o2["volume"]["total_m3"]) < 1e-6
