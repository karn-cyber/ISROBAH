"""Polarimetric feature extraction — the scientific core of HIMADRI.

From the full Stokes vector (S0,S1,S2,S3) we derive:
  * CPR  — circular polarisation ratio (classic, ambiguous, ice flag)
  * DOP  — degree of polarisation (the disambiguator: ice randomises -> low)
  * chi  — ellipticity angle (even/odd bounce indicator)
  * m-chi decomposition — single / double / volume scattering powers

The whole trick in one line: ice and rock COLLIDE on the CPR axis but SEPARATE
on DOP, on the decomposition, and across L/S bands.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _s(stokes: np.ndarray):
    return stokes[0], stokes[1], stokes[2], stokes[3]


def cpr(stokes: np.ndarray) -> np.ndarray:
    """CPR = sigma_SC / sigma_OC, with
    sigma_OC = (S0 + S3)/2, sigma_SC = (S0 - S3)/2."""
    s0, _, _, s3 = _s(stokes)
    sigma_oc = (s0 + s3) / 2.0
    sigma_sc = (s0 - s3) / 2.0
    return (sigma_sc / (sigma_oc + _EPS)).astype(np.float32)


def dop(stokes: np.ndarray) -> np.ndarray:
    """Degree of polarisation m = sqrt(S1^2+S2^2+S3^2)/S0  (0=random,1=pure)."""
    s0, s1, s2, s3 = _s(stokes)
    m = np.sqrt(s1**2 + s2**2 + s3**2) / (s0 + _EPS)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def chi(stokes: np.ndarray) -> np.ndarray:
    """Ellipticity angle chi = 0.5 * arcsin(S3 / (m*S0))."""
    s0, _, _, s3 = _s(stokes)
    m = dop(stokes)
    arg = np.clip(s3 / (m * s0 + _EPS), -1.0, 1.0)
    return (0.5 * np.arcsin(arg)).astype(np.float32)


def m_chi_decomposition(stokes: np.ndarray) -> dict[str, np.ndarray]:
    """Raney m-chi decomposition into single / double / volume powers.

      volume = S0 * (1 - m)
      single = (m*S0/2) * (1 + sin(2*chi))     # surface / odd-bounce
      double = (m*S0/2) * (1 - sin(2*chi))     # double-bounce / even
    True ice -> volume-dominated; rocky terrain -> double-bounce-dominated.
    """
    s0, _, _, _ = _s(stokes)
    m = dop(stokes)
    c = chi(stokes)
    sin2chi = np.sin(2 * c)
    volume = s0 * (1 - m)
    single = (m * s0 / 2.0) * (1 + sin2chi)
    double = (m * s0 / 2.0) * (1 - sin2chi)
    return {
        "single": single.astype(np.float32),
        "double": double.astype(np.float32),
        "volume": volume.astype(np.float32),
    }


def volume_fraction(stokes: np.ndarray) -> np.ndarray:
    """Fraction of total power in the volume term — a compact ice indicator."""
    d = m_chi_decomposition(stokes)
    total = d["single"] + d["double"] + d["volume"] + _EPS
    return (d["volume"] / total).astype(np.float32)


def polarimetric_stack(radar_l, radar_s) -> dict[str, np.ndarray]:
    """Compute the per-band polarimetric feature bands for L and S."""
    out: dict[str, np.ndarray] = {}
    for tag, radar in (("L", radar_l), ("S", radar_s)):
        st = radar.stokes
        out[f"cpr_{tag}"] = cpr(st)
        out[f"dop_{tag}"] = dop(st)
        out[f"chi_{tag}"] = chi(st)
        dec = m_chi_decomposition(st)
        out[f"mchi_single_{tag}"] = dec["single"]
        out[f"mchi_double_{tag}"] = dec["double"]
        out[f"mchi_volume_{tag}"] = dec["volume"]
        out[f"s0_{tag}"] = st[0].astype(np.float32)
    return out
