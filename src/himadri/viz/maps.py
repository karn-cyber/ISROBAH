"""Figure generation (deck-style dark theme). Produces PNGs for the report and
the GIS-style decision layers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "#0b0f1a",
    "axes.facecolor": "#0b0f1a",
    "savefig.facecolor": "#0b0f1a",
    "text.color": "#e6ecff",
    "axes.labelcolor": "#e6ecff",
    "xtick.color": "#9fb0d0",
    "ytick.color": "#9fb0d0",
    "axes.titlecolor": "#e6ecff",
    "font.size": 10,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _imshow(ax, arr, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_maps(feats, ice, suitability, sites_gdf, route, vol, truth, outdir) -> dict:
    outdir = Path(outdir)
    figs = {}

    # 1. detection panel
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _imshow(ax[0], ice.probability, "Ice probability (posterior)", "magma", 0, 1)
    _imshow(ax[1], ice.uncertainty, "Uncertainty", "cividis", 0, 1)
    _imshow(ax[2], ice.target_mask.astype(float), "High-confidence target", "magma", 0, 1)
    figs["detection"] = str(_save(fig, outdir / "fig_detection.png"))

    # 2. the disambiguation: CPR vs DOP coloured by truth
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cpr = feats.bands["cpr_L"].ravel()
    dop = feats.bands["dop_L"].ravel()
    cls = truth["class_map"].ravel() if "class_map" in truth else np.zeros_like(cpr)
    sub = np.random.default_rng(0).choice(len(cpr), size=min(6000, len(cpr)), replace=False)
    colors = {0: "#5566aa", 1: "#33d6a6", 2: "#ff5566"}
    names = {0: "flat regolith", 1: "ICE", 2: "rocky rim"}
    for k in (0, 2, 1):
        m = cls[sub] == k
        ax[0].scatter(cpr[sub][m], dop[sub][m], s=4, alpha=0.4,
                      c=colors[k], label=names[k])
    ax[0].axvline(1.0, ls="--", c="#aaaaaa", lw=1)
    ax[0].axhline(0.13, ls="--", c="#aaaaaa", lw=1)
    ax[0].set_xlabel("CPR (L-band)"); ax[0].set_ylabel("DOP (L-band)")
    ax[0].set_title("Why CPR alone fails: ice & rock collide on CPR,\nseparate on DOP")
    ax[0].legend(loc="upper right", framealpha=0.2)
    ax[0].set_xlim(0, 2.2); ax[0].set_ylim(0, 1)

    vol_frac = (feats.bands["mchi_volume_L"] /
                (feats.bands["mchi_volume_L"] + feats.bands["mchi_double_L"]
                 + feats.bands["mchi_single_L"] + 1e-6))
    _imshow(ax[1], vol_frac, "m-chi volume fraction (ice -> high)", "viridis", 0, 1)
    figs["disambiguation"] = str(_save(fig, outdir / "fig_disambiguation.png"))

    # 3. landing + traverse over hillshade-ish base
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    _imshow(ax[0], suitability, "Landing suitability", "YlGn", 0, None)
    if len(sites_gdf):
        rows = sites_gdf["row"].values; cols = sites_gdf["col"].values
        ax[0].scatter(cols, rows, c="#00e5ff", s=80, marker="*",
                      edgecolors="k", zorder=5)
        for _, srow in sites_gdf.iterrows():
            ax[0].annotate(f"#{int(srow['rank'])}", (srow["col"], srow["row"]),
                           color="#00e5ff", fontsize=9)
    _imshow(ax[1], ice.probability, "Traverse: landing -> ice -> sun", "magma", 0, 1)
    if route:
        rr = [p[0] for p in route]; cc = [p[1] for p in route]
        ax[1].plot(cc, rr, c="#00e5ff", lw=2, zorder=5)
        ax[1].scatter([cc[0]], [rr[0]], c="#ffffff", s=70, marker="o", zorder=6, label="land")
        ax[1].scatter([cc[-1]], [rr[-1]], c="#ffd166", s=70, marker="s", zorder=6, label="sun-return")
        ax[1].legend(loc="upper right", framealpha=0.2)
    figs["planning"] = str(_save(fig, outdir / "fig_planning.png"))

    # 4. volume per depth bin
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [f"{b['depth_top_m']:.0f}-{b['depth_bottom_m']:.0f}m" for b in vol.per_depth_bin]
    vals = [b["volume_m3"] for b in vol.per_depth_bin]
    ax.bar(bins, vals, color="#33d6a6")
    ax.set_ylabel("Ice volume (m³)")
    ax.set_title(f"Ice volume by depth — total {vol.total_m3:,.0f} m³ "
                 f"[{vol.lower_m3:,.0f}, {vol.upper_m3:,.0f}]")
    figs["volume"] = str(_save(fig, outdir / "fig_volume.png"))

    return figs
