"""Energy + thermal model for the traverse.

Walks the planned route, accumulating distance, energy draw (base load + a
slope/hazard-dependent term, with NO solar input inside shadow) and time in
darkness (thermal exposure). Verifies the dash-and-return stays within the
battery and thermal-survival budget.
"""
from __future__ import annotations

import math

import numpy as np


def energy_profile(route, cost, illumination, slope_deg, grid, cfg) -> dict:
    if not route:
        return {"feasible": False, "reason": "no route", "waypoints": []}

    res = grid.res_m
    speed = cfg.planning.rover_speed_mps
    base_W = cfg.planning.base_draw_W
    battery_Wh = cfg.planning.battery_Wh

    total_dist = 0.0
    energy_Wh = 0.0
    time_s = 0.0
    dark_time_s = 0.0
    solar_panel_W = 50.0  # nominal panel output in full sun
    waypoints = []

    for i in range(1, len(route)):
        (r0, c0), (r1, c1) = route[i - 1], route[i]
        seg = math.hypot(r1 - r0, c1 - c0) * res
        total_dist += seg
        dt = seg / speed
        time_s += dt
        slope = float(slope_deg[r1, c1])
        # locomotion power rises with slope
        drive_W = base_W + 2.0 * max(slope, 0)
        lit = float(illumination[r1, c1])
        solar_in_W = solar_panel_W * lit
        net_W = drive_W - solar_in_W
        energy_Wh += max(net_W, -solar_panel_W) * dt / 3600.0
        if lit < 0.05:
            dark_time_s += dt
        x, y = grid.rc_to_xy(r1, c1)
        waypoints.append({
            "row": int(r1), "col": int(c1), "x": x, "y": y,
            "dist_m": round(total_dist, 1),
            "energy_Wh": round(energy_Wh, 2),
            "illumination": round(lit, 3),
            "in_shadow": lit < 0.05,
        })

    peak_energy = max((w["energy_Wh"] for w in waypoints), default=0.0)
    feasible = (peak_energy <= battery_Wh) and (dark_time_s / 60.0 <= cfg.planning.thermal_limit_min)
    return {
        "feasible": bool(feasible),
        "total_distance_m": round(total_dist, 1),
        "total_time_min": round(time_s / 60.0, 1),
        "dark_time_min": round(dark_time_s / 60.0, 1),
        "peak_energy_Wh": round(peak_energy, 2),
        "battery_Wh": battery_Wh,
        "thermal_limit_min": cfg.planning.thermal_limit_min,
        "energy_margin_Wh": round(battery_Wh - peak_energy, 2),
        "waypoints": waypoints,
    }
