"""Optical / terrain features from DEM + OHRC.

OHRC is blind inside the permanent shadow, so we use it for what it is good at:
rim morphology, boulder/hazard mapping and roughness priors on the illuminated
approach terrain. These optical features DOWN-WEIGHT radar ice scores where the
surface is demonstrably rough/rocky (the fusion step), resolving the CPR
ambiguity toward 'rock'.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def slope_deg(elevation: np.ndarray, res_m: float) -> np.ndarray:
    gy, gx = np.gradient(elevation.astype(np.float64), res_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    return slope.astype(np.float32)


def roughness(elevation: np.ndarray, window: int = 5) -> np.ndarray:
    """Local terrain roughness: std of elevation in a moving window (TRI-like)."""
    e = elevation.astype(np.float64)
    mean = ndimage.uniform_filter(e, size=window)
    mean_sq = ndimage.uniform_filter(e**2, size=window)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(var).astype(np.float32)


def glcm_contrast(image: np.ndarray, window: int = 7) -> np.ndarray:
    """Texture contrast proxy. Uses a local-variance surrogate for GLCM contrast
    (robust to NaN shadow regions and far cheaper than a full GLCM per pixel)."""
    img = np.nan_to_num(image.astype(np.float64), nan=0.0)
    mean = ndimage.uniform_filter(img, size=window)
    mean_sq = ndimage.uniform_filter(img**2, size=window)
    contrast = np.maximum(mean_sq - mean**2, 0.0)
    return contrast.astype(np.float32)


def boulder_density(image: np.ndarray, window: int = 9) -> np.ndarray:
    """Boulder density via Laplacian-of-Gaussian blob response, averaged in a
    window. Bright compact features (boulders) yield high response."""
    img = np.nan_to_num(image.astype(np.float64), nan=0.0)
    log = ndimage.gaussian_laplace(img, sigma=1.2)
    resp = np.clip(-log, 0, None)  # bright blobs -> negative LoG
    thresh = resp.mean() + 2.0 * resp.std()
    blobs = (resp > thresh).astype(np.float32)
    density = ndimage.uniform_filter(blobs, size=window)
    return density.astype(np.float32)


def optical_stack(dem, optical, res_m: float) -> dict[str, np.ndarray]:
    elev = dem.elevation
    refl = optical.reflectance
    return {
        "slope_deg": slope_deg(elev, res_m),
        "roughness": roughness(elev),
        "glcm_contrast": glcm_contrast(refl),
        "boulder_density": boulder_density(refl),
    }
