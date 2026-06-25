import numpy as np

from himadri.detect import classifier, fusion, pseudolabels, uncertainty
from himadri.validate import metrics as M


def _detect(feats, cfg):
    labels, weights = pseudolabels.make_pseudolabels(feats, cfg)
    radar_prob, _, _ = classifier.train_predict(feats, labels, weights, cfg)
    rock = fusion.optical_rock_probability(feats)
    post = fusion.bayesian_fusion(radar_prob, rock) * (feats.bands["psr_mask"] > 0.5)
    unc = uncertainty.total_uncertainty(post, radar_prob, rock, feats.bands["geometry_mask"])
    target = (post > cfg.detection.target_prob) & (unc < 0.6)
    return post, target


def test_detection_auc(scene, feats, cfg):
    post, _ = _detect(feats, cfg)
    auc, _ = M._roc_auc(post, scene.truth["ice_mask"].astype(bool))
    assert auc >= 0.90


def test_baseline_beats(scene, feats, cfg):
    post, target = _detect(feats, cfg)
    rim = scene.truth["class_map"] == 2
    baseline = M.baseline_cpr_only(feats, cfg).astype(bool)
    base_fp = (baseline & rim).sum()
    our_fp = (target.astype(bool) & rim).sum()
    assert base_fp > 0
    reduction = 1 - our_fp / base_fp
    assert reduction >= 0.70


def test_no_ice_outside_psr(scene, feats, cfg):
    post, target = _detect(feats, cfg)
    psr = feats.bands["psr_mask"] > 0.5
    assert (target & ~psr).sum() == 0
