import numpy as np

from himadri.detect import classifier, fusion, pseudolabels, uncertainty
from himadri.planning import cost_surface, energy, planner
from himadri.types import IceResult
from himadri.volume.dielectric import estimate_volume


def _ice_result(feats, cfg):
    labels, weights = pseudolabels.make_pseudolabels(feats, cfg)
    radar_prob, _, _ = classifier.train_predict(feats, labels, weights, cfg)
    rock = fusion.optical_rock_probability(feats)
    post = fusion.bayesian_fusion(radar_prob, rock) * (feats.bands["psr_mask"] > 0.5)
    unc = uncertainty.total_uncertainty(post, radar_prob, rock, feats.bands["geometry_mask"])
    target = (post > cfg.detection.target_prob) & (unc < 0.6)
    return IceResult(grid=feats.grid, probability=post, uncertainty=unc, target_mask=target)


def test_volume_error_and_interval(scene, feats, cfg):
    ice = _ice_result(feats, cfg)
    vol = estimate_volume(ice, feats, cfg)
    truth = scene.truth["volume_m3"]
    rel = abs(vol.total_m3 - truth) / truth
    assert rel <= 0.25
    assert vol.lower_m3 <= truth <= vol.upper_m3


def test_planner_respects_constraints(scene, feats, cfg):
    ice = _ice_result(feats, cfg)
    cost = cost_surface.build_cost_surface(feats, cfg)
    # start: brightest low-slope pixel; goal: ice target centroid
    score = feats.bands["illumination_frac"] - feats.bands["slope_deg"] / 90.0
    start = tuple(np.unravel_index(np.argmax(score), score.shape))
    ys, xs = np.where(ice.target_mask)
    goal = (int(np.median(ys)), int(np.median(xs)))
    route = planner.plan_route(cost, start, goal, "astar")
    assert len(route) > 0
    # never traverses an impassable (slope>max) cell
    slope = feats.bands["slope_deg"]
    assert all(slope[r, c] <= cfg.planning.max_slope_deg + 1e-3 for r, c in route)


def test_planner_no_path_when_goal_walled(feats, cfg):
    cost = cost_surface.build_cost_surface(feats, cfg)
    walled = cost.copy()
    walled[:] = 1e9  # everything impassable
    route = planner.plan_route(walled, (0, 0), (10, 10), "astar")
    assert route == []
