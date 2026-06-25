"""Interpretable baseline classifier on the physical feature stack.

Uses XGBoost when available, else falls back to scikit-learn's
HistGradientBoosting / RandomForest — so the prototype always runs on CPU with
no fragile dependency. Feature importances feed the report (great for the
'how is this different?' slide).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import FeatureStack

# Feature bands fed to the classifier (physically meaningful, interpretable).
CLASSIFIER_BANDS = [
    "cpr_L", "cpr_S", "dop_L", "dop_S", "chi_L", "chi_S",
    "mchi_single_L", "mchi_double_L", "mchi_volume_L",
    "mchi_single_S", "mchi_double_S", "mchi_volume_S",
    "ls_ratio", "slope_deg", "roughness", "glcm_contrast",
    "boulder_density", "illumination_frac",
]


def _build_model(seed: int):
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
        return model, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.08, random_state=seed,
        )
        return model, "histgradientboosting"


def _feature_matrix(feats: FeatureStack, bands: list[str]) -> np.ndarray:
    cols = [np.nan_to_num(feats.bands[b], nan=0.0).ravel() for b in bands]
    return np.column_stack(cols).astype(np.float32)


def train_predict(feats: FeatureStack, labels: np.ndarray, weights: np.ndarray,
                  cfg: Config) -> tuple[np.ndarray, dict, str]:
    """Train on physics seeds, predict ice probability for every pixel.

    Returns (probability_map (H,W), feature_importance, backend_name).
    """
    bands = [b for b in CLASSIFIER_BANDS if b in feats.bands]
    X = _feature_matrix(feats, bands)
    y = labels.ravel()
    w = weights.ravel()
    seed_mask = y >= 0

    model, backend = _build_model(cfg.seed)

    if seed_mask.sum() < 20 or len(np.unique(y[seed_mask])) < 2:
        # degenerate: fall back to a smooth physics score
        prob = _physics_score(feats)
        return prob, {}, "physics-fallback"

    Xs, ys, ws = X[seed_mask], y[seed_mask], w[seed_mask]
    try:
        model.fit(Xs, ys, sample_weight=ws)
    except TypeError:
        model.fit(Xs, ys)

    prob_flat = model.predict_proba(X)[:, 1]
    prob = prob_flat.reshape(feats.grid.shape).astype(np.float32)

    importance = _importance(model, bands, backend)
    return prob, importance, backend


def _importance(model, bands, backend) -> dict:
    try:
        if backend == "xgboost":
            imp = model.feature_importances_
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            return {}
        return {b: float(v) for b, v in sorted(
            zip(bands, imp), key=lambda kv: -kv[1])}
    except Exception:
        return {}


def _physics_score(feats: FeatureStack) -> np.ndarray:
    """Smooth fallback ice score from first principles (no training needed)."""
    b = feats.bands
    cpr = np.clip((b["cpr_L"] - 1.0) / 0.5, 0, 1)
    low_dop = np.clip((0.13 - b["dop_L"]) / 0.13, 0, 1)
    total = b["mchi_volume_L"] + b["mchi_double_L"] + b["mchi_single_L"] + 1e-6
    vol_frac = b["mchi_volume_L"] / total
    ls = np.clip((b["ls_ratio"] - 1.0) / 0.5, 0, 1)
    score = 0.25 * cpr + 0.30 * low_dop + 0.25 * vol_frac + 0.20 * ls
    return score.astype(np.float32)
