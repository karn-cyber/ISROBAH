"""Physics-informed weak supervision.

We have no pixel-level ice labels for the real crater. So we generate
pseudo-labels from physics: pixels that satisfy ALL of {high CPR, low DOP,
volume-dominated, low optical roughness, inside PSR} become high-confidence
positive seeds; clearly rocky (double-bounce / high roughness / boulders)
pixels become negatives. A classifier trains on these seeds; the rest are
unlabelled. Sample weights encode the margin to the thresholds.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import FeatureStack


def make_pseudolabels(feats: FeatureStack, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, weights). labels: 1=ice, 0=not-ice, -1=unlabelled."""
    b = feats.bands
    cpr = b["cpr_L"]
    dop = b["dop_L"]
    vol = b["mchi_volume_L"]
    dbl = b["mchi_double_L"]
    sng = b["mchi_single_L"]
    total = vol + dbl + sng + 1e-6
    vol_frac = vol / total
    dbl_frac = dbl / total
    rough = b["roughness"]
    boulders = b["boulder_density"]
    ls = b["ls_ratio"]
    in_psr = b["psr_mask"] > 0.5

    rough_n = rough / (np.nanpercentile(rough, 95) + 1e-6)

    # POSITIVE seeds: the multi-evidence ice signature
    pos = (
        (cpr > cfg.detection.cpr_ice_min)
        & (dop < cfg.detection.dop_ice_max)
        & (vol_frac > 0.55)
        & (rough_n < 0.4)
        & (ls > 1.2)
        & in_psr
    )
    # NEGATIVE seeds: clearly rough / double-bounce / boulder-rich
    neg = (
        (dbl_frac > 0.40)
        | (rough_n > 0.6)
        | (boulders > np.nanpercentile(boulders, 85))
    ) & ~pos

    labels = np.full(cpr.shape, -1, dtype=np.int8)
    labels[neg] = 0
    labels[pos] = 1

    # weights: margin to thresholds (confident seeds weigh more)
    weights = np.zeros(cpr.shape, dtype=np.float32)
    weights[pos] = np.clip(
        (cpr[pos] - cfg.detection.cpr_ice_min)
        + (cfg.detection.dop_ice_max - dop[pos]) * 5
        + (vol_frac[pos] - 0.55),
        0.1, 3.0,
    )
    weights[neg] = np.clip(dbl_frac[neg] + rough_n[neg], 0.1, 3.0)
    return labels, weights
