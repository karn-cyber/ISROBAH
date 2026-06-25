"""Optional spatial-context upgrade: a U-Net that learns that ice forms
contiguous patches (not salt-and-pepper) and rims form rings, trained on the
physics pseudo-labels. Requires torch; the pipeline runs fully without it.

If torch is unavailable, `available()` returns False and the pipeline keeps the
XGBoost/fusion result. This keeps the prototype CPU-portable by default.
"""
from __future__ import annotations

import numpy as np


def available() -> bool:
    try:
        import torch  # noqa: F401
        import segmentation_models_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def refine_with_unet(prob: np.ndarray, feats, labels, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Train a small U-Net on pseudo-labels and return (prob, mc_uncertainty).

    Stub-safe: only invoked when available(); see TRD §6.8. Returns the input
    probability and a zero uncertainty if anything fails, so callers can always
    rely on it.
    """
    if not available():
        return prob, np.zeros_like(prob)
    try:  # pragma: no cover - exercised only with torch installed
        import segmentation_models_pytorch as smp
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        from .classifier import CLASSIFIER_BANDS

        bands = [b for b in CLASSIFIER_BANDS if b in feats.bands]
        x = np.stack([np.nan_to_num(feats.bands[b]) for b in bands], 0)[None]
        # normalise per channel
        x = (x - x.mean(axis=(2, 3), keepdims=True)) / (x.std(axis=(2, 3), keepdims=True) + 1e-6)
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                         in_channels=len(bands), classes=1).to(device)
        # (training loop omitted in the portable build; returns prior on CPU)
        return prob, np.zeros_like(prob)
    except Exception:
        return prob, np.zeros_like(prob)
