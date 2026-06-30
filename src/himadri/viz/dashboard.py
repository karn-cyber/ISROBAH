"""HIMADRI — interactive Streamlit mission console.

A polished, live-controllable frontend for the demo: tweak detection
thresholds, fusion strength, landing weights and the rover energy budget and
watch the ice map, volume, landing site and traverse update in real time.

Run:
    himadri dashboard
    # or:  streamlit run src/himadri/viz/dashboard.py
Requires:  pip install streamlit plotly   (the core pipeline does not).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from himadri.config import Config
from himadri.detect import classifier, fusion, pseudolabels, uncertainty
from himadri.features import depth, optical, polarimetry
from himadri.illumination import psr as psr_mod
from himadri.landing.site_selection import select_landing
from himadri.planning import cost_surface, energy as energy_mod, planner
from himadri.preprocess import masks, speckle, terrain
from himadri.synth.generate import generate_scene
from himadri.types import FeatureStack, IceResult
from himadri.validate import metrics as M
from himadri.volume.dielectric import estimate_volume

ACCENT = "#00e5ff"
ICE = "#33d6a6"
ROCK = "#ff5566"

# Custom "ice glow" scale: near-zero blends into the app background (#0b0f1a),
# ice ramps through teal -> cyan -> white so targets glow on a dark field.
ICE_SCALE = [
    [0.00, "#0b0f1a"], [0.15, "#10243a"], [0.35, "#0e5e6e"],
    [0.55, "#19a7a0"], [0.78, "#33d6a6"], [1.00, "#eafff4"],
]


def _human(n: float) -> str:
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"

# --------------------------------------------------------------------------- #
#  Cached compute layers (heavy -> light, so live knobs stay responsive)      #
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_base(seed: int, h: int, w: int):
    """Scene + feature stack (synthetic). Cached for the session."""
    cfg = Config()
    cfg.seed, cfg.grid.height, cfg.grid.width = seed, h, w
    scene = generate_scene(cfg)
    scene.radar_l.stokes = speckle.refined_lee(scene.radar_l.stokes)
    scene.radar_s.stokes = speckle.refined_lee(scene.radar_s.stokes)

    bands: dict[str, np.ndarray] = {}
    bands.update(polarimetry.polarimetric_stack(scene.radar_l, scene.radar_s))
    bands.update(optical.optical_stack(scene.dem, scene.optical, scene.grid.res_m))
    bands["ls_ratio"] = depth.ls_ratio(scene.radar_l, scene.radar_s)
    bands.update(psr_mod.illumination_stack(scene.dem, scene.sun))
    inc = terrain.incidence_angle(scene.dem.elevation, scene.grid.res_m)
    bands["incidence_L"] = inc
    bands["incidence_S"] = inc
    bands["cpr_L"] = terrain.normalize_cpr(bands["cpr_L"], inc)
    bands["cpr_S"] = terrain.normalize_cpr(bands["cpr_S"], inc)
    bands["geometry_mask"] = masks.layover_shadow(
        scene.dem.elevation, scene.grid.res_m, inc).astype(np.float32)
    feats = FeatureStack(grid=scene.grid, bands=bands)
    return scene, feats


def detect_core(feats, cfg):
    """Pseudo-labels -> classifier -> radar & optical-rock probabilities."""
    labels, weights = pseudolabels.make_pseudolabels(feats, cfg)
    radar_prob, importance, backend = classifier.train_predict(feats, labels, weights, cfg)
    rock_prob = fusion.optical_rock_probability(feats)
    return radar_prob, rock_prob, importance, backend


@st.cache_data(show_spinner=False)
def get_radar_prob(seed, h, w, cpr_min, dop_max):
    """Synthetic-mode cached detection (depends on pseudo-label thresholds)."""
    _, feats = get_base(seed, h, w)
    cfg = Config()
    cfg.seed = seed
    cfg.detection.cpr_ice_min = cpr_min
    cfg.detection.dop_ice_max = dop_max
    return detect_core(feats, cfg)


def build_ice(feats, radar_prob, rock_prob, k, target_prob, unc_max):
    """Live fusion + uncertainty + target mask (cheap)."""
    posterior = fusion.bayesian_fusion(radar_prob, rock_prob, prior=0.5)
    # apply the configurable rock weight by re-deriving in logit space
    eps = 1e-4
    p = np.clip(radar_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)) - k * rock_prob
    posterior = (1 / (1 + np.exp(-logit))).astype(np.float32)
    posterior *= (feats.bands["psr_mask"] > 0.5)
    unc = uncertainty.total_uncertainty(posterior, radar_prob, rock_prob,
                                        feats.bands["geometry_mask"])
    target = (posterior > target_prob) & (unc < unc_max)
    return IceResult(grid=feats.grid, probability=posterior, uncertainty=unc,
                     target_mask=target)


def plan_core(scene, feats, ice, cfg):
    """Volume + landing + traverse + (if truth) metrics. Source-agnostic."""
    from himadri import pipeline as P

    vol = estimate_volume(ice, feats, cfg)
    suit, sites = select_landing(feats, ice, cfg)
    cost = cost_surface.build_cost_surface(feats, cfg)
    start = P._pick_start(sites, feats)
    goal = P._pick_goal(ice)
    route_info = planner.dash_and_return(cost, feats.bands["illumination_frac"],
                                         start, goal, cfg)
    eprof = energy_mod.energy_profile(route_info.get("route", []), cost,
                                      feats.bands["illumination_frac"],
                                      feats.bands["slope_deg"], feats.grid, cfg)
    site_rc = [(int(r["row"]), int(r["col"]), float(r["score"]))
               for _, r in sites.iterrows()] if len(sites) else []
    metrics = None
    if scene.truth:
        metrics = M.evaluate(ice.probability, ice.target_mask, scene.truth, feats,
                             cfg, vol_est=vol, vol_truth=scene.truth.get("volume_m3"))
    return {
        "suit": np.asarray(suit), "sites": site_rc, "route": route_info.get("route", []),
        "energy": eprof, "vol": vol, "metrics": metrics,
    }


@st.cache_data(show_spinner=False)
def get_plan(seed, h, w, cpr_min, dop_max, k, target_prob, unc_max,
             battery, slope_lim, prox_w, n_mc):
    """Synthetic-mode cached plan."""
    scene, feats = get_base(seed, h, w)
    radar_prob, rock_prob, _, _ = get_radar_prob(seed, h, w, cpr_min, dop_max)
    ice = build_ice(feats, radar_prob, rock_prob, k, target_prob, unc_max)
    cfg = _knob_cfg(seed, n_mc, battery, slope_lim, prox_w)
    return plan_core(scene, feats, ice, cfg)


def _knob_cfg(seed, n_mc, battery, slope_lim, prox_w):
    cfg = Config()
    cfg.seed = seed
    cfg.volume.n_mc = int(n_mc)
    cfg.planning.battery_Wh = float(battery)
    cfg.planning.max_slope_deg = float(slope_lim)
    cfg.landing.weights["proximity"] = float(prox_w)
    return cfg


@st.cache_resource(show_spinner=False)
def get_real_base(sig: str, dem_path: str, l_path: str, s_path: str = "",
                  ohrc_path: str = "", grid_dim: int = 384):
    """Build (scene, feats) from uploaded real product files. `sig` keys the
    cache (changes when uploads change)."""
    from himadri import pipeline as P

    cfg = Config()
    cfg.mode = "real"
    cfg.grid.height = cfg.grid.width = int(grid_dim)
    cfg.paths.dem = dem_path
    cfg.paths.dfsar_l = l_path
    cfg.paths.dfsar_s = s_path or None
    cfg.paths.ohrc = ohrc_path or None
    scene = P.build_real_scene(cfg)
    scene = P.preprocess(scene, cfg)
    feats = P.build_features(scene, cfg)
    return scene, feats


# --------------------------------------------------------------------------- #
#  Plot helpers                                                               #
# --------------------------------------------------------------------------- #

def heatmap(z, title, colorscale="Viridis", zmin=None, zmax=None, overlay_route=None,
            sites=None):
    fig = go.Figure(go.Heatmap(z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               colorbar=dict(thickness=12)))
    if overlay_route:
        rr = [p[0] for p in overlay_route]
        cc = [p[1] for p in overlay_route]
        fig.add_trace(go.Scatter(x=cc, y=rr, mode="lines", line=dict(color=ACCENT, width=3),
                                 name="traverse"))
        fig.add_trace(go.Scatter(x=[cc[0]], y=[rr[0]], mode="markers",
                                 marker=dict(color="white", size=12, symbol="circle"),
                                 name="land"))
        fig.add_trace(go.Scatter(x=[cc[-1]], y=[rr[-1]], mode="markers",
                                 marker=dict(color="#ffd166", size=12, symbol="square"),
                                 name="sun-return"))
    if sites:
        fig.add_trace(go.Scatter(
            x=[s[1] for s in sites], y=[s[0] for s in sites], mode="markers+text",
            marker=dict(color=ACCENT, size=14, symbol="star", line=dict(color="black", width=1)),
            text=[f"#{i+1}" for i in range(len(sites))], textposition="top center",
            textfont=dict(color=ACCENT), name="landing sites"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(title=title, template="plotly_dark", height=440,
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                      legend=dict(orientation="h", y=-0.08))
    return fig


# --------------------------------------------------------------------------- #
#  App                                                                        #
# --------------------------------------------------------------------------- #

def _save_uploads(files, subdir: str) -> list[str]:
    base = Path(tempfile.gettempdir()) / "himadri_uploads" / subdir
    base.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        p = base / f.name
        p.write_bytes(f.getbuffer())
        saved.append(str(p))
    return saved


def _openable(saved: list[str], prefer: list[str]) -> str | None:
    for ext in prefer:
        for p in saved:
            if p.lower().endswith(ext):
                return p
    return saved[0] if saved else None


def _real_upload_panel(s):
    """Sidebar file-drop for real products. Returns dict of openable paths + a
    cache signature, or None until the minimum (DEM + L-band DFSAR) is present."""
    s.markdown("**Real products** · drop files (GeoTIFF preferred; PDS `.img`+`.lbl` "
               "pairs accepted together).")
    dem_f = s.file_uploader("LOLA DEM (.tif / .img+.lbl)", type=["tif", "tiff", "img", "lbl"],
                            accept_multiple_files=True, key="dem")
    l_f = s.file_uploader("DFSAR L-band (Stokes or quad-pol)",
                          type=["tif", "tiff", "img", "xml", "lbl"],
                          accept_multiple_files=True, key="dl")
    s2_f = s.file_uploader("DFSAR S-band (optional)",
                           type=["tif", "tiff", "img", "xml", "lbl"],
                           accept_multiple_files=True, key="ds")
    ohrc_f = s.file_uploader("OHRC optical (optional)",
                             type=["tif", "tiff", "img", "xml"],
                             accept_multiple_files=True, key="oh")
    out: dict = {}
    sig_parts = []
    if dem_f:
        out["dem"] = _openable(_save_uploads(dem_f, "dem"), [".lbl", ".tif", ".tiff", ".img"])
        sig_parts += [f.name + str(f.size) for f in dem_f]
    if l_f:
        out["dfsar_l"] = _openable(_save_uploads(l_f, "dl"), [".tif", ".tiff", ".xml", ".img", ".lbl"])
        sig_parts += [f.name + str(f.size) for f in l_f]
    if s2_f:
        out["dfsar_s"] = _openable(_save_uploads(s2_f, "ds"), [".tif", ".tiff", ".xml", ".img", ".lbl"])
        sig_parts += [f.name + str(f.size) for f in s2_f]
    if ohrc_f:
        out["ohrc"] = _openable(_save_uploads(ohrc_f, "oh"), [".tif", ".tiff", ".xml", ".img"])
        sig_parts += [f.name + str(f.size) for f in ohrc_f]
    out["sig"] = "|".join(sig_parts)
    if not out.get("dfsar_s"):
        s.caption("No S-band → L/S depth feature runs in neutral (L-only) mode.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.parse_known_args()

    st.set_page_config(page_title="HIMADRI Mission Console", layout="wide",
                       page_icon="❄️")
    st.markdown(f"""
    <style>
      .stApp {{ background:#0b0f1a; }}
      h1,h2,h3,h4 {{ color:#e6ecff; }}
      .him-sub {{ color:#9fb0d0; margin-top:-10px; }}
      div[data-testid="stMetric"] {{ background:#121829; border:1px solid #243049;
         border-radius:12px; padding:14px 16px; }}
      div[data-testid="stMetricValue"] {{ color:{ACCENT}; font-size:1.6rem; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# ❄️ HIMADRI — Mission Console")
    st.markdown('<div class="him-sub">Hybrid Ice Mapping And Detection using Radar '
                'Intelligence · ISRO BAH 2026 · Problem Statement 8</div>',
                unsafe_allow_html=True)

    # ---- sidebar controls -------------------------------------------------
    s = st.sidebar
    s.header("⚙️ Controls")
    source = s.radio("Data source", ["Synthetic crater", "Upload real data"],
                     help="Synthetic = built-in doubly-shadowed crater with ground "
                          "truth. Real = drop in the organisers' DFSAR/OHRC + a LOLA DEM.")
    real_paths = None
    if source == "Upload real data":
        real_paths = _real_upload_panel(s)
    grid_n = s.select_slider(
        "Grid size (px)" if source == "Synthetic crater" else "Grid cap (px)",
        [128, 160, 192, 224, 256, 320, 384], value=192)
    seed = s.number_input("Random seed", 0, 9999, 42, 1)

    s.subheader("Detection")
    cpr_min = s.slider("CPR ice threshold", 0.6, 1.6, 1.0, 0.05,
                       help="Pseudo-label seed: CPR above this is ice-like.")
    dop_max = s.slider("DOP ice threshold", 0.05, 0.40, 0.13, 0.01,
                       help="Ice randomises polarisation → low DOP. The disambiguator.")
    k = s.slider("Optical-rock fusion weight", 0.0, 8.0, 4.0, 0.5,
                 help="How hard optical roughness pulls 'ice-like' radar toward rock.")
    target_prob = s.slider("Target threshold", 0.3, 0.9, 0.6, 0.05)
    unc_max = s.slider("Max uncertainty for target", 0.3, 1.0, 0.6, 0.05)

    s.subheader("Landing & rover")
    prox_w = s.slider("Landing: proximity-to-ice weight", 0.0, 0.6, 0.30, 0.05)
    slope_lim = s.slider("Rover max slope (°)", 15, 35, 25, 1)
    battery = s.slider("Rover battery (Wh)", 500, 4000, 2000, 100)
    n_mc = s.select_slider("Volume Monte-Carlo samples", [400, 800, 1500, 3000], value=800)

    if source == "Upload real data" and not (real_paths and real_paths.get("dem")
                                              and real_paths.get("dfsar_l")):
        st.info("⬆️ Upload at least a **LOLA DEM** and an **L-band DFSAR** product in "
                "the sidebar to run on real data. (S-band & OHRC optional.) "
                "Meanwhile, see the synthetic crater by switching the data source.")
        st.stop()

    with st.spinner("Computing detection, volume, landing & traverse…"):
        if source == "Synthetic crater":
            scene, feats = get_base(seed, grid_n, grid_n)
            radar_prob, rock_prob, importance, backend = get_radar_prob(
                seed, grid_n, grid_n, cpr_min, dop_max)
            ice = build_ice(feats, radar_prob, rock_prob, k, target_prob, unc_max)
            plan = get_plan(seed, grid_n, grid_n, cpr_min, dop_max, k, target_prob,
                            unc_max, float(battery), float(slope_lim), float(prox_w), int(n_mc))
        else:
            scene, feats = get_real_base(real_paths["sig"], real_paths["dem"],
                                         real_paths["dfsar_l"], real_paths.get("dfsar_s", ""),
                                         real_paths.get("ohrc", ""), int(grid_n))
            cfg = _knob_cfg(seed, n_mc, battery, slope_lim, prox_w)
            cfg.detection.cpr_ice_min = cpr_min
            cfg.detection.dop_ice_max = dop_max
            radar_prob, rock_prob, importance, backend = detect_core(feats, cfg)
            ice = build_ice(feats, radar_prob, rock_prob, k, target_prob, unc_max)
            plan = plan_core(scene, feats, ice, cfg)

    vol = plan["vol"]
    metrics = plan["metrics"]
    eprof = plan["energy"]

    # ---- headline metrics -------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ice volume · top 5 m", f"{_human(vol.total_m3)} m³",
              f"± {_human((vol.upper_m3 - vol.lower_m3) / 2)} m³")
    if metrics:
        c2.metric("Detection ROC-AUC", f"{metrics['roc_auc']:.3f}")
        rr = metrics["rim_fp_reduction_fraction"]
        c3.metric("Rim FP vs CPR-only", f"−{rr*100:.0f}%" if rr == rr else "—",
                  "fewer false positives")
    feas = eprof.get("feasible")
    c4.metric("Rover traverse", "Feasible" if feas else "Over budget",
              f"{_human(eprof.get('total_distance_m', 0))} m route")

    tabs = st.tabs(["🛰️ Detection", "🔬 Disambiguation", "🧊 Volume",
                    "🚀 Landing & Traverse", "✅ Validation"])

    # ---- tab: detection ---------------------------------------------------
    with tabs[0]:
        layers = {
            "Ice probability (posterior)": (ice.probability, ICE_SCALE, 0, 1),
            "Uncertainty": (ice.uncertainty, "Cividis", 0, 1),
            "High-confidence target": (ice.target_mask.astype(float), ICE_SCALE, 0, 1),
            "CPR (L-band)": (feats.bands["cpr_L"], "Turbo", 0, 2),
            "DOP (L-band)": (feats.bands["dop_L"], "Turbo", 0, 1),
            "m-χ volume power (L)": (feats.bands["mchi_volume_L"], "Viridis", None, None),
            "L/S ratio (depth)": (feats.bands["ls_ratio"], "RdBu", 0.5, 1.6),
            "Illumination fraction": (feats.bands["illumination_frac"], "Inferno", 0, 1),
            "Slope (°)": (feats.bands["slope_deg"], "Hot", 0, 45),
            "Doubly-shadowed mask": (feats.bands["doubly_shadowed_mask"], "Greys", 0, 1),
        }
        col = st.columns([2, 1])
        layer = col[1].selectbox("Map layer", list(layers.keys()))
        col[1].markdown("**Why uncertainty matters**")
        col[1].caption("Uncertainty is high precisely on the rocky rims — where CPR "
                       "looks ice-like but optics says rough. That is the CPR ambiguity, "
                       "flagged honestly instead of mis-called as ice.")
        z, cs, zmn, zmx = layers[layer]
        col[0].plotly_chart(heatmap(z, layer, cs, zmn, zmx), use_container_width=True)

    # ---- tab: disambiguation ---------------------------------------------
    with tabs[1]:
        st.markdown("#### Ice and rock collide on CPR — but separate on DOP")
        cpr = feats.bands["cpr_L"].ravel()
        dop = feats.bands["dop_L"].ravel()
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cpr), size=min(5000, len(cpr)), replace=False)
        fig = go.Figure()
        if "class_map" in scene.truth:
            cls = scene.truth["class_map"].ravel()
            names = {0: "flat regolith", 1: "ICE", 2: "rocky rim"}
            colors = {0: "#5566aa", 1: ICE, 2: ROCK}
            for kcls in (0, 2, 1):
                m = cls[idx] == kcls
                fig.add_trace(go.Scatter(x=cpr[idx][m], y=dop[idx][m], mode="markers",
                                         marker=dict(size=4, color=colors[kcls], opacity=0.5),
                                         name=names[kcls]))
        else:
            # real mode: no labels — colour by HIMADRI's ice probability
            prob = ice.probability.ravel()[idx]
            fig.add_trace(go.Scatter(
                x=cpr[idx], y=dop[idx], mode="markers",
                marker=dict(size=4, color=prob, colorscale=ICE_SCALE, cmin=0, cmax=1,
                            opacity=0.6, colorbar=dict(title="ice prob", thickness=12)),
                name="pixels"))
        fig.add_vline(x=cpr_min, line_dash="dash", line_color="#aaaaaa")
        fig.add_hline(y=dop_max, line_dash="dash", line_color="#aaaaaa")
        fig.add_annotation(x=cpr_min + 0.25, y=dop_max / 2, text="ICE window",
                           showarrow=False, font=dict(color=ICE))
        fig.update_layout(template="plotly_dark", height=480, xaxis_title="CPR (L-band)",
                          yaxis_title="DOP (L-band)", paper_bgcolor="#0b0f1a",
                          plot_bgcolor="#0b0f1a", xaxis_range=[0, 2.2], yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Move the CPR/DOP sliders in the sidebar to see the decision window "
                   "shift. A naive CPR>1 cut (vertical line) also captures the entire red "
                   "rocky-rim cloud — the false positives HIMADRI removes with the DOP axis.")

    # ---- tab: volume ------------------------------------------------------
    with tabs[2]:
        cc = st.columns([1.3, 1])
        bins = [f"{b['depth_top_m']:.0f}–{b['depth_bottom_m']:.0f} m" for b in vol.per_depth_bin]
        vals = [b["volume_m3"] for b in vol.per_depth_bin]
        bar = go.Figure(go.Bar(x=bins, y=vals, marker_color=ICE))
        bar.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0b0f1a",
                          plot_bgcolor="#0b0f1a", yaxis_title="Ice volume (m³)",
                          title=f"Volume by depth — total {vol.total_m3:,.0f} m³")
        cc[0].plotly_chart(bar, use_container_width=True)
        cc[1].metric("Total (top 5 m)", f"{vol.total_m3:,.0f} m³")
        cc[1].metric("90% interval", f"{vol.lower_m3:,.0f} – {vol.upper_m3:,.0f} m³")
        cc[1].metric("Mean ice fraction", f"{vol.mean_ice_fraction:.3f}")
        cc[1].metric("Effective permittivity", f"{vol.meta['mean_eps_eff']:.2f}")
        cc[1].caption(f"Monte-Carlo over abundance coefficients, porosity, density & "
                      f"loss tangent (n={vol.meta['n_mc']}). Reported ± bound, never a "
                      f"bare number.")

    # ---- tab: landing & traverse -----------------------------------------
    with tabs[3]:
        cc = st.columns(2)
        cc[0].plotly_chart(
            heatmap(plan["suit"], "Landing suitability", "YlGn", sites=plan["sites"]),
            use_container_width=True)
        cc[1].plotly_chart(
            heatmap(ice.probability, "Dash-and-return traverse", ICE_SCALE, 0, 1,
                    overlay_route=plan["route"]), use_container_width=True)
        wps = eprof.get("waypoints", [])
        if wps:
            d = [w["dist_m"] for w in wps]
            e = [w["energy_Wh"] for w in wps]
            shadow = [w["in_shadow"] for w in wps]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d, y=e, mode="lines", line=dict(color=ACCENT, width=2),
                                     name="cumulative energy"))
            sx = [d[i] for i in range(len(d)) if shadow[i]]
            sy = [e[i] for i in range(len(e)) if shadow[i]]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                                     marker=dict(color="#3344aa", size=5), name="in shadow"))
            fig.add_hline(y=battery, line_dash="dash", line_color=ROCK,
                          annotation_text="battery limit")
            fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0b0f1a",
                              plot_bgcolor="#0b0f1a", xaxis_title="distance along route (m)",
                              yaxis_title="energy used (Wh)", title="Energy / thermal profile")
            st.plotly_chart(fig, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distance", f"{eprof.get('total_distance_m', 0)} m")
        m2.metric("Time in shadow", f"{eprof.get('dark_time_min', 0)} min")
        m3.metric("Peak energy", f"{eprof.get('peak_energy_Wh', 0)} Wh")
        m4.metric("Energy margin", f"{eprof.get('energy_margin_Wh', 0)} Wh")

    # ---- tab: validation --------------------------------------------------
    with tabs[4]:
        if metrics is None:
            st.info("**Real-data mode** — no pixel-level ground truth exists for an "
                    "actual crater, so ROC-AUC/IoU aren't defined. Confidence instead "
                    "comes from converging physical evidence (CPR + DOP + m-χ + L/S + "
                    "optical), per-pixel uncertainty, and the feature importances below. "
                    "Quantitative skill is demonstrated on the synthetic crater, where "
                    "truth is known.")
        if metrics:
            cc = st.columns(2)
            cc[0].markdown("#### Detection vs synthetic ground truth")
            cc[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            cc[0].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
            cc[0].metric("IoU (target vs truth)", f"{metrics['iou']:.3f}")
            cc[1].markdown("#### CPR-only baseline vs HIMADRI")
            cc[1].metric("Rocky-rim FPs — CPR-only",
                         metrics["baseline_cpr_only"]["rim_false_positives"])
            cc[1].metric("Rocky-rim FPs — HIMADRI",
                         metrics["himadri"]["rim_false_positives"])
            rr = metrics["rim_fp_reduction_fraction"]
            cc[1].metric("Reduction", f"{rr*100:.0f}%" if rr == rr else "—")
            if "volume" in metrics:
                v = metrics["volume"]
                st.markdown("#### Volume validation")
                st.write({
                    "estimate_m3": round(v["estimate_m3"]),
                    "truth_m3": round(v["truth_m3"]),
                    "relative_error": f"{v['relative_error']*100:.1f}%",
                    "interval_contains_truth": v["interval_contains_truth"],
                })
        if importance:
            st.markdown(f"#### What the model relies on  ·  backend: `{backend}`")
            imp = sorted(importance.items(), key=lambda kv: -kv[1])[:12]
            ifig = go.Figure(go.Bar(x=[v for _, v in imp][::-1],
                                    y=[kk for kk, _ in imp][::-1], orientation="h",
                                    marker_color=ACCENT))
            ifig.update_layout(template="plotly_dark", height=380, paper_bgcolor="#0b0f1a",
                               plot_bgcolor="#0b0f1a", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(ifig, use_container_width=True)


if __name__ == "__main__":
    main()
