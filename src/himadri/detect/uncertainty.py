"""Per-pixel uncertainty quantification.

Every detection ships with a confidence — required for safe ISRU siting. We
combine three sources:
  * probabilistic ambiguity  — entropy of the posterior (max near p=0.5)
  * evidence disagreement    — radar says 'ice' but optics says 'rock'
  * geometry penalty         — layover/shadow & low-illumination pixels
"""
from __future__ import annotations

import numpy as np


def total_uncertainty(posterior: np.ndarray, radar_prob: np.ndarray,
                      rock_prob: np.ndarray, geometry_mask: np.ndarray) -> np.ndarray:
    p = np.clip(posterior, 1e-4, 1 - 1e-4)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))  # 0..1
    disagreement = np.clip(radar_prob * rock_prob, 0, 1)  # high when both fire
    geom = geometry_mask.astype(np.float32) * 0.5
    unc = np.clip(0.5 * entropy + 0.3 * disagreement + geom, 0, 1)
    return unc.astype(np.float32)
