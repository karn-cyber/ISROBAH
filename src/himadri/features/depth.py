"""Depth attribution via the dual-frequency (L/S) contrast.

Penetration depth scales with wavelength: L-band (longer lambda) probes deeper
than S-band. So a signature STRONG in L but WEAK in S is consistent with BURIED
ice; one strong in both is shallow/surface. The L/S contrast is therefore both
a depth discriminator and a confidence input — and is exactly why DFSAR
(dual-frequency), not a single-band SAR, was chosen for this problem.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_EPS = 1e-6


def ls_ratio(radar_l, radar_s, smooth: int = 3) -> np.ndarray:
    """S0_L / S0_S, smoothed. >1 indicates a buried (L-dominant) scatterer."""
    s0_l = radar_l.stokes[0]
    s0_s = radar_s.stokes[0]
    ratio = s0_l / (s0_s + _EPS)
    ratio = ndimage.uniform_filter(ratio.astype(np.float64), size=smooth)
    return ratio.astype(np.float32)


def penetration_depth_m(wavelength_cm: float, loss_tangent: float,
                        eps_r: float = 2.7) -> float:
    """Approximate microwave skin/penetration depth in dry regolith (metres).

    d ~ lambda * sqrt(eps_r) / (2*pi*loss_tangent)  (low-loss approximation).
    """
    lam_m = wavelength_cm / 100.0
    return float(lam_m * np.sqrt(eps_r) / (2 * np.pi * max(loss_tangent, 1e-4)))
