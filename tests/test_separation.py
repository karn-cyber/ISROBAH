import numpy as np


def test_ice_has_lower_dop_than_rock(scene, feats):
    cls = scene.truth["class_map"]
    ice = cls == 1
    rock = cls == 2
    dop = feats.bands["dop_L"]
    assert dop[ice].mean() < dop[rock].mean()


def test_ice_is_volume_dominated(scene, feats):
    cls = scene.truth["class_map"]
    ice = cls == 1
    rock = cls == 2
    vol = feats.bands["mchi_volume_L"]
    dbl = feats.bands["mchi_double_L"]
    sng = feats.bands["mchi_single_L"]
    total = vol + dbl + sng + 1e-6
    vol_frac = vol / total
    dbl_frac = dbl / total
    assert vol_frac[ice].mean() > 0.5            # ice -> volume dominated
    assert dbl_frac[rock].mean() > dbl_frac[ice].mean()  # rock more double-bounce


def test_ice_in_psr(scene, feats):
    ice = scene.truth["class_map"] == 1
    psr = feats.bands["psr_mask"] > 0.5
    assert (ice & psr).sum() / ice.sum() > 0.95


def test_ls_ratio_higher_for_ice(scene, feats):
    cls = scene.truth["class_map"]
    ls = feats.bands["ls_ratio"]
    assert ls[cls == 1].mean() > ls[cls == 0].mean()
