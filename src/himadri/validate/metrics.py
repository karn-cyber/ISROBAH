"""Validation against synthetic ground truth, plus the CPR-only baseline.

The CPR-only baseline (threshold CPR>1 => ice) is implemented deliberately so
we can QUANTIFY our improvement: HIMADRI suppresses the rocky-rim false
positives that flood a naive CPR threshold. That comparison is a direct,
defensible answer to 'how is this different?'.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import FeatureStack


def baseline_cpr_only(feats: FeatureStack, cfg: Config) -> np.ndarray:
    """The naive detector: CPR>1 (in PSR) => ice. Floods rocky rims."""
    cpr = feats.bands["cpr_L"]
    in_psr = feats.bands["psr_mask"] > 0.5
    return ((cpr > cfg.detection.cpr_ice_min) & in_psr).astype(np.float32)


def _roc_auc(prob, truth):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = truth.ravel().astype(int)
    p = prob.ravel()
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def _iou(pred_mask, truth):
    inter = np.logical_and(pred_mask, truth).sum()
    union = np.logical_or(pred_mask, truth).sum()
    return float(inter / (union + 1e-9))


def evaluate(prob, target_mask, truth, feats, cfg, vol_est=None, vol_truth=None,
             rim_mask=None) -> dict:
    truth_ice = truth["ice_mask"].astype(bool)
    auc, ap = _roc_auc(prob, truth_ice)
    iou = _iou(target_mask, truth_ice)

    # false-positive comparison on the rocky rim (where the naive detector fails)
    if rim_mask is None:
        rim_mask = truth["class_map"] == 2
    baseline = baseline_cpr_only(feats, cfg).astype(bool)
    ours = target_mask.astype(bool)

    base_fp = int((baseline & rim_mask).sum())
    our_fp = int((ours & rim_mask).sum())
    rim_n = int(rim_mask.sum())
    fp_reduction = 1.0 - (our_fp / base_fp) if base_fp > 0 else float("nan")

    metrics = {
        "roc_auc": auc,
        "pr_auc": ap,
        "iou": iou,
        "baseline_cpr_only": {
            "rim_false_positives": base_fp,
            "rim_fp_rate": base_fp / (rim_n + 1e-9),
        },
        "himadri": {
            "rim_false_positives": our_fp,
            "rim_fp_rate": our_fp / (rim_n + 1e-9),
        },
        "rim_fp_reduction_fraction": fp_reduction,
    }
    if vol_est is not None and vol_truth is not None:
        err = (vol_est.total_m3 - vol_truth) / (vol_truth + 1e-9)
        metrics["volume"] = {
            "estimate_m3": vol_est.total_m3,
            "truth_m3": vol_truth,
            "relative_error": err,
            "abs_relative_error": abs(err),
            "interval_m3": [vol_est.lower_m3, vol_est.upper_m3],
            "interval_contains_truth": bool(vol_est.lower_m3 <= vol_truth <= vol_est.upper_m3),
        }
    return metrics
