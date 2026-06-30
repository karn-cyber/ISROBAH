# HIMADRI — Real Data Guide (ISRO BAH 2026 · PS-8)

How to feed **real** Chandrayaan-2 + LOLA data into HIMADRI: where each input
comes from, how much you need, and exactly how it maps into the pipeline.

> **Sources verified June 2026.** Endpoints/filenames were checked live; the
> LOLA URLs below are confirmed downloadable. PRADAN requires a free account so
> its per-file links can't be hard-coded — the checklist tells you what to click.

---

## TL;DR

| Input | Source | Format | How much | Required? |
|---|---|---|---|---|
| **DFSAR L-band** (full/quad-pol) | **Supplied by organisers** · also ISSDC PRADAN | PDS4/ISDA (XML + binary) or GeoTIFF | ~0.3–2 GB / scene | **Yes** |
| **DFSAR S-band** | Same | Same | similar | Optional (enables L/S depth) |
| **OHRC** optical | ISSDC PRADAN | PDS4 `.zip` → `.img`+`.xml`, ~0.28 m/px | ~1–4 GB / strip | Optional (rim/boulders) |
| **LOLA DEM** | **NASA LOLA PDS node** (scriptable) | PDS3 `.IMG`+`.LBL` (or GeoTIFF) | ~1.7–1.9 GB full tile → ~tens of MB cropped | **Yes** |
| Sun geometry | NAIF SPICE (optional) | SPICE kernels | tens of MB | Optional (analytic fallback) |

**Minimum viable real run:** 1 DFSAR L-band scene + 1 LOLA DEM tile. Total to
download yourself ≈ **2 GB** (the DEM); the DFSAR scene is handed to you.

---

## 1. LOLA DEM — scripted (do this now, before the event)

The DEM is the **geometric backbone**: it defines the analysis grid, drives
illumination/PSR mapping, slope, terrain correction, landing and traverse.

**One command** downloads the right tile for a crater and crops it to a small,
ready-to-use GeoTIFF:

```bash
himadri fetch-dem --crater faustini          # downloads + crops to a GeoTIFF
himadri fetch-dem --crater faustini --no-download   # just print the URLs
```

Verified tiles (LOLA GDR polar node — `https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/`):

| Tile | Res | Coverage | Size | Use for |
|---|---|---|---|---|
| `LDEM_875S_5M`  | 5 m  | 87.5–90°S | ~1.75 GB | Shackleton, de Gerlache, Shoemaker |
| `LDEM_85S_10M`  | 10 m | 85–90°S   | ~1.75 GB | **Faustini**, Haworth |
| `LDEM_80S_20M`  | 20 m | 80–90°S   | ~1.85 GB | Cabeus, Nobile, wider context |

Built-in crater registry (centre lon/lat + recommended tile): `faustini`,
`shackleton`, `de_gerlache`, `cabeus`, `haworth`, `shoemaker`, `nobile`.
The `.IMG` needs its detached `.LBL` for projection — `fetch-dem` grabs both and
GDAL reads the pair.

> The "Faustini doubly-shadowed crater" in PS-8 sits inside the Faustini PSR
> (~87.2°S, 75.8°E) → `LDEM_85S_10M` (10 m). The 5 m product's 87.5°S cap just
> misses Faustini's centre.

---

## 2. DFSAR + OHRC — ISSDC PRADAN (account required)

Portal: **https://pradan.issdc.gov.in/ch2/** → create a free account → browse
Chandrayaan-2.

**DFSAR checklist**
- The crater scene is **supplied by the organisers** — start from that file.
- If pulling your own (PRADAN → SAR table): match **CentreLatitude/Longitude**
  to your crater. For **Faustini (~87.2°S, ~77°E)** the centred scene is the
  `20241005t052503010` acquisition (CentreLon ≈ 77).
- **Pick the right product level (confirmed from the DFSAR User Manual v1.0):**

  | Level | Filename code | Pixels | Use for polarimetry? |
  |---|---|---|---|
  | **L1A SLC** (single-look complex, per-pol TIFFs RH/RV, +Grid CSV) | **`ncxl`** (larger, ~4 GB) | complex I/Q, **phase preserved** | ✅ **DOWNLOAD THIS** |
  | L1B GRI (ground-range) | `nrxl` (~1 GB) | 2-byte amplitude, no phase | ❌ can't do DOP/m-χ |
  | L2 SRI (map-projected) | — | 2-byte amplitude | ❌ amplitude only |

  → For Faustini: **`ch2_sar_ncxl_20241005t052503010_d_cp_d18.zip` (SLC, ~4 GB)**.
- **`cp` = compact-pol** (RH+RV) — exactly what the m-χ decomposition is for.
- **L1A is slant-range, not geocoded:** geocode using the companion **Grid CSV**
  (per-pixel lat/lon + incidence angle) that ships in the product.
- Also grab a **Derived polar mosaic** (`ProcessingLevel Contains Derived`) —
  CPR/DOP/inversion already computed & map-projected; great cross-check.
- DFSAR is L-band; if no S-band scene, HIMADRI runs L-only (L/S feature goes
  neutral — see §4).

**OHRC checklist**
- Tick **Optical High Resolution Camera**, pick the strip(s) over your crater's
  illuminated rim/approach (OHRC is blind inside the shadow). Files are `.zip`
  → unzip to `.img` + `.xml` (PDS4).

---

## 3. Plug it into HIMADRI

### A) Dashboard (no config editing)
```bash
himadri dashboard
```
Sidebar → **Data source → Upload real data** → drop the **DEM** and **DFSAR
L-band** (S-band + OHRC optional). The live console renders the real scene.
GeoTIFF is easiest; PDS `.img`+`.lbl` pairs can be dropped together.

### B) CLI / batch
Edit `config/config.yaml`:
```yaml
mode: real
grid:
  height: 384      # grid CAP — the real grid is derived from the DEM footprint
  width: 384
paths:
  dem:      data/raw/dem_faustini_10m.tif   # from `himadri fetch-dem`
  dfsar_l:  data/raw/<organiser_dfsar_L>.tif
  dfsar_s:  null                            # or the S-band file
  ohrc:     data/raw/<ohrc_strip>.tif       # or null
```
```bash
himadri run --mode real
```
Outputs (GeoTIFF + GeoPackage + `report.html`) land in `data/outputs/`.

---

## 4. Format adapters & honest caveats

- **DFSAR → Stokes** (`io/loaders.py::load_dfsar`). Auto-detects layout:
  4 real bands = Stokes already; 4 complex / 8 real bands = quad-pol scattering
  matrix → converted to circular-transmit Stokes via
  `stokes_from_scattering_matrix`; 2 bands = compact/hybrid child-Stokes.
  *TODO[VERIFY]:* confirm the supplied product's exact band order at the event
  and tweak the mapping — everything downstream is unchanged.
- **OHRC** (`load_ohrc`): GeoTIFF or PDS `.img`/`.xml` via GDAL; DN auto-scaled,
  nodata → NaN (shadow).
- **DEM** (`load_dem`): warps the (possibly multi-GB) tile onto the analysis
  grid via a memory-safe `WarpedVRT` — never loads the whole tile into RAM.
- **CRS**: everything is co-registered to **lunar south-polar stereographic**
  (IAU Moon sphere, R=1737.4 km), using the DEM as the backbone.
- **L-band only**: if no S-band, `ls_ratio≈1` everywhere and the depth criterion
  is dropped from the pseudo-labels automatically. Detection still runs; expect
  more rim ambiguity than with both bands.
- **No ground truth on real data**: ROC-AUC/IoU are only defined on the
  synthetic crater. On the real crater, confidence comes from converging
  physical evidence + per-pixel uncertainty (see the Validation tab).

---

## 5. Sources

- ISSDC PRADAN (Chandrayaan-2 DFSAR & OHRC): https://pradan.issdc.gov.in/ch2/
- LOLA GDR polar DEMs (PDS node): https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/
- PGDA high-res south-pole LOLA products: https://pgda.gsfc.nasa.gov/products/78
- NAIF SPICE kernels: https://naif.jpl.nasa.gov/
- DFSAR instrument/products: https://arxiv.org/abs/2104.14259
