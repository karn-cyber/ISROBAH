# HIMADRI 🏔️❄️
### Hybrid Ice Mapping And Detection using Radar Intelligence
**ISRO BAH 2026 · Problem Statement 8** — Detection & characterization of subsurface ice in lunar south-polar *doubly-shadowed* craters, with landing-site selection and rover-traverse planning.

---

## The one idea that wins this

> **A high radar circular-polarisation ratio (CPR) is ambiguous — rough rocky terrain mimics ice.**

Most teams threshold `CPR > 1` and call it ice. That floods the crater rim — *exactly where you must not land or drive* — with false positives. HIMADRI **disambiguates**: it fuses CPR with the **degree of polarisation (DOP)**, the **m-χ scattering decomposition**, the **dual-frequency L/S depth signature**, and **optical roughness**, then reports a **calibrated probability with per-pixel uncertainty**.

Ice and rock *collide* on the CPR axis but *separate* on DOP, on the decomposition, and across L/S bands — so we classify in that richer space.

On the built-in synthetic crater (with known ground truth) this produces:

| Metric | Target | HIMADRI |
|---|---|---|
| Detection ROC-AUC | ≥ 0.90 | **0.998** |
| Rim false-positives vs CPR-only baseline | ≥ 70% fewer | **~100% fewer** |
| Ice-volume error (top 5 m) | within ±25% | **~5%**, interval contains truth |
| End-to-end runtime (CPU) | < 10 min | **~30 s** |
| Rover traverse | safe + within energy/thermal budget | **feasible dash-and-return** |

---

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

himadri run                 # full pipeline on synthetic data (default)
open data/outputs/report.html
```

That single command produces all five deliverables in `data/outputs/`:

1. `ice_probability.tif` + `ice_uncertainty.tif` + `ice_target_mask.tif` — disambiguated detection
2. `volume_report.json` — ice volume (top 5 m) ± bound, per depth bin
3. `landing_sites.gpkg` — ranked, safe landing sites
4. `traverse.gpkg` (+ `traverse_waypoints.gpkg`) — power/thermal-aware rover route
5. `report.html` — figures + numbers + method summary (open in any browser)

…plus `metrics.json` validating against synthetic ground truth, and `features/*.tif` (CPR, DOP, m-χ volume, L/S ratio, roughness, slope, illumination, PSR & doubly-shadowed masks) — all GeoTIFFs that open directly in QGIS/ArcGIS.

### Other commands
```bash
himadri synth      --out data/synthetic     # write the synthetic scene + ground truth as GeoTIFFs
himadri features   --config config/config.yaml
himadri dashboard  --config config/config.yaml   # optional Streamlit UI (pip install streamlit)
pytest -q                                    # 16 acceptance tests
```

---

## Why this runs *today*, before the real data arrives

The official crater DFSAR/OHRC files are handed out **at the event**. HIMADRI ships a **physically-grounded synthetic generator** (`synth/generate.py`) that builds a doubly-shadowed crater with **known ground truth**, so the whole pipeline is built, tested and *measured* before real data exists. At the event you point `config.yaml` at the real files (`mode: real`) and the **adapter layer** (`io/loaders.py`) feeds them through the identical downstream code.

The synthetic ice obeys the real physics we exploit: **high CPR + low DOP + volume-dominated + strong-in-L/weak-in-S** for ice; **high CPR + high DOP + double-bounce + rough optics** for rock. CPR alone cannot tell them apart — which is the entire point.

---

## Pipeline

```
DFSAR(L+S, full-Stokes)      OHRC optical            LOLA DEM + sun ephemeris
        │                         │                          │
   speckle-filter            roughness / boulders     PSR & doubly-shadowed
   build Stokes              GLCM / slope             mapping (horizon ray-cast)
        └────────────┬────────────┴───────────┬──────────────┘
                     ▼   co-register to common polar grid  ▼
  Features: CPR · DOP · χ · m-χ(SB/DB/Vol) · L/S ratio · roughness · illumination
                     │
        physics pseudo-labels → gradient-boosted classifier (+ optional U-Net)
                     │
        Bayesian radar × optical fusion → ICE PROBABILITY + UNCERTAINTY
                     │
        ├─► dielectric/abundance inversion (Monte-Carlo) → VOLUME (top 5 m) ± bound
        ├─► multi-criteria suitability → ranked SAFE LANDING SITE
        └─► cost surface → A* → power/thermal DASH-AND-RETURN TRAVERSE
                     ▼
            GIS decision layers + HTML report
```

## What makes it defensible to judges

- **Scientific robustness** — detection requires *converging* evidence (polarimetry **and** thermal/PSR environment **and** L/S depth consistency **and** low optical roughness), not one threshold.
- **Honest uncertainty** — every pixel and the volume estimate ship with calibrated confidence; physical constants (regolith density, porosity, loss tangent, mixing coefficients) are **priors**, Monte-Carlo propagated — never magic numbers.
- **Provable improvement** — the CPR-only baseline is implemented *in the codebase* so we quantify exactly how many rim false-positives we kill.
- **Portable & reproducible** — pure open-source, CPU-only, seeded/deterministic, config-driven. Optional upgrades (`xgboost`, `torch`+U-Net, `streamlit`) auto-detect; the pipeline never depends on them.

## Config

Everything is driven by [`config/config.yaml`](config/config.yaml): grid, detection thresholds (`cpr_ice_min`, `dop_ice_max`), volume priors, landing weights, and rover energy/thermal budget. Switch `mode: real` and set `paths:` to ingest the supplied crater data.

## Layout

```
src/himadri/
  synth/        synthetic doubly-shadowed crater + ground truth
  io/           adapter loaders + GeoTIFF/vector writers
  preprocess/   speckle filter, terrain correction, geometry masks, co-registration
  features/     polarimetry (CPR/DOP/χ/m-χ), optical roughness, L/S depth
  illumination/ PSR & doubly-shadowed mapping (horizon ray-casting)
  detect/       pseudo-labels → classifier → fusion → uncertainty (+ optional U-Net)
  volume/       dielectric/abundance inversion + Monte-Carlo volume
  landing/      multi-criteria suitability + site ranking
  planning/     cost surface, A*/D*-Lite planner, energy/thermal model
  viz/          figures, HTML report, optional Streamlit dashboard
  validate/     metrics + CPR-only baseline comparison
  pipeline.py   orchestration   ·   cli.py   typer CLI
```

> **Data-source note.** External archive URLs/formats (ISSDC PRADAN, PDS/USGS LOLA, NAIF SPICE) change and several need registration; every such reference is marked `TODO[VERIFY]` in the code. Synthetic mode is the default and never blocks on them.
# ISROBAH
