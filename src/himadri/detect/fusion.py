"""Bayesian radar x optical fusion.

The radar gives an ice probability; the optics give a 'rockiness' probability
from roughness + boulder density. We treat them as independent evidence and
compute a posterior that DOWN-WEIGHTS radar ice where optics says 'rough/rocky'
— resolving the CPR ambiguity toward rock exactly where a lander/rover must not
go. This is the headline disambiguation move.
"""
from __future__ import annotations

import numpy as np

from ..types import FeatureStack


def optical_rock_probability(feats: FeatureStack) -> np.ndarray:
    """P(rocky) from optical/terrain evidence (roughness + boulders + slope)."""
    b = feats.bands
    rough = b["roughness"]
    boulders = b["boulder_density"]
    slope = b["slope_deg"]
    rn = rough / (np.nanpercentile(rough, 95) + 1e-6)
    bn = boulders / (np.nanpercentile(boulders, 95) + 1e-6)
    sn = np.clip(slope / 25.0, 0, 1)
    rock = np.clip(0.5 * rn + 0.3 * bn + 0.2 * sn, 0, 1)
    return rock.astype(np.float32)


def bayesian_fusion(radar_prob: np.ndarray, rock_prob: np.ndarray,
                    prior: float = 0.5) -> np.ndarray:
    """Posterior ice probability.

    Likelihood that a pixel is ice given radar evidence is radar_prob; optical
    rock evidence reduces it. We combine in log-odds space:
        logit(post) = logit(radar) - k * rock_prob
    so strong optical rockiness pulls the posterior down sharply.
    """
    eps = 1e-4
    p = np.clip(radar_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    k = 4.0  # rock-evidence weight
    logit_post = logit - k * rock_prob
    post = 1.0 / (1.0 + np.exp(-logit_post))
    return post.astype(np.float32)
