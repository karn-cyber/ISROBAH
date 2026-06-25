"""Synthetic doubly-shadowed crater generator — the linchpin of the prototype.

It builds a physically plausible scene with KNOWN ground truth so the entire
pipeline runs end-to-end with no external data, and detection quality is
*measurable* (ROC-AUC, IoU, volume error). The polarimetric signatures are
constructed to obey the real physics we exploit downstream:

  * ICE (inner doubly-shadowed floor): high CPR, LOW DOP, volume-dominated
    m-chi, and STRONGER in L than S  (buried -> high L/S ratio).
  * ROCKY RIM / EJECTA: high CPR, HIGH DOP, double-bounce-dominated, similar
    in L and S, with high optical roughness / boulder density.
  * FLAT REGOLITH: low CPR, surface (single-bounce) dominated.

Crucially CPR alone does NOT separate ICE from ROCK (both high) — exactly the
ambiguity HIMADRI resolves. The separation lives in DOP, the decomposition,
the L/S band contrast and optical roughness.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..grid import build_grid
from ..types import (
    DEMProduct,
    Grid,
    OpticalProduct,
    RadarProduct,
    Scene,
    SunGeometry,
)


def stokes_from_targets(
    s0: np.ndarray,
    cpr: np.ndarray,
    dop: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a (4,H,W) Stokes field with the requested per-pixel CPR & DOP.

    Convention: sigma_OC=(S0+S3)/2, sigma_SC=(S0-S3)/2, CPR=sigma_SC/sigma_OC
      => S3 = S0 * (1 - CPR) / (1 + CPR).
    The polarised power m*S0 is split between the linear terms (S1,S2) and the
    circular term S3. The residual linear power is distributed across S1,S2 with
    a random orientation so the field is not degenerate.
    """
    s0 = np.maximum(s0, 1e-6)
    s3 = s0 * (1.0 - cpr) / (1.0 + cpr)
    pol_power = dop * s0
    lin_sq = np.maximum(pol_power**2 - s3**2, 0.0)  # S1^2 + S2^2
    lin = np.sqrt(lin_sq)
    theta = rng.uniform(0, np.pi, size=s0.shape)  # orientation of linear pol
    s1 = lin * np.cos(2 * theta)
    s2 = lin * np.sin(2 * theta)
    return np.stack([s0, s1, s2, s3], axis=0).astype(np.float32)


def _apply_speckle(stokes: np.ndarray, looks: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative gamma speckle (n-look) applied coherently to the Stokes
    vector, so CPR/DOP stay statistically meaningful but noisy — motivating the
    polarimetric speckle filter downstream."""
    shape = stokes.shape[1:]
    gain = rng.gamma(shape=looks, scale=1.0 / looks, size=shape).astype(np.float32)
    out = stokes * gain[None]
    # small independent perturbation on the polarised terms
    for i in (1, 2, 3):
        out[i] += rng.normal(0, 0.02, size=shape).astype(np.float32) * stokes[0]
    out[0] = np.maximum(out[0], 1e-6)
    return out


def _radial(grid: Grid, cx: float, cy: float) -> np.ndarray:
    rr, cc = np.mgrid[0 : grid.height, 0 : grid.width]
    return np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2) * grid.res_m


def _theta(grid: Grid, cx: float, cy: float) -> np.ndarray:
    rr, cc = np.mgrid[0 : grid.height, 0 : grid.width]
    return np.arctan2(rr - cy, cc - cx)


def add_crater(elev, r, theta, R, depth, floor_frac, rim_h, rim_w,
               breach_theta=None, breach_sigma=0.5, breach_extend=400.0):
    """Add a flat-floored crater to `elev`.

    Flat floor (r < floor_frac*R) at -depth, a smooth wall rising to the
    surface at R, and a raised rim. An optional azimuthal `breach` models a
    slumped wall: the wall is EXTENDED outward (R -> R_eff) over the breach
    sector so the same depth drop happens over a much longer run — a gentle,
    passable access ramp that still descends all the way to the floor (no
    cliff), with the rim suppressed. This is common and realistic for old
    polar craters.
    """
    floor_R = floor_frac * R
    if breach_theta is not None:
        dtheta = np.abs(np.angle(np.exp(1j * (theta - breach_theta))))
        bw = np.exp(-(dtheta ** 2) / (2 * breach_sigma ** 2))   # 0..1 in sector
    else:
        bw = np.zeros_like(r)
    R_eff = R + breach_extend * bw
    t = np.clip((r - floor_R) / (R_eff - floor_R + 1e-6), 0, 1)
    wall = 0.5 * (1 - np.cos(np.pi * t))            # 0 at floor edge, 1 at R_eff
    depress = np.where(r <= floor_R, -depth, -depth * (1 - wall))
    depress = np.where(r >= R_eff, 0.0, depress)
    rim = rim_h * np.exp(-((r - R) ** 2) / (2 * rim_w ** 2)) * (1 - bw)
    return elev + depress + rim


def generate_scene(cfg: Config) -> Scene:
    rng = np.random.default_rng(cfg.seed)
    grid = build_grid(cfg)
    H, W = grid.height, grid.width
    res = grid.res_m

    # ---- geometry (flat-floored craters, realistic slopes) --------------
    cx, cy = W * 0.42, H * 0.5
    r_main = _radial(grid, cx, cy)
    th_main = _theta(grid, cx, cy)
    R_main = min(H, W) * res * 0.36           # ~920 m radius
    # inner doubly-shadowed crater, offset east toward the access side
    icx, icy = cx + 0.40 * R_main / res, cy
    r_inner = _radial(grid, icx, icy)
    th_inner = _theta(grid, icx, icy)
    R_inner = R_main * 0.26                    # ~240 m radius

    # ---- DEM ------------------------------------------------------------
    elev = np.zeros((H, W), dtype=np.float32)
    elev += (rng.standard_normal((H, W)).astype(np.float32)) * 0.6  # background roughness
    yy = (np.arange(H)[:, None] - cy) * res
    elev += (yy / 1000.0) * 3.0                # gentle regional tilt
    # main crater: breach on the EAST side (theta ~ 0) -> the landing approach
    elev = add_crater(elev, r_main, th_main, R=R_main, depth=220.0,
                      floor_frac=0.55, rim_h=55.0, rim_w=R_main * 0.06,
                      breach_theta=0.0, breach_sigma=0.50, breach_extend=1300.0)
    # inner crater: breach on the EAST side too, facing the incoming rover
    elev = add_crater(elev, r_inner, th_inner, R=R_inner, depth=90.0,
                      floor_frac=0.45, rim_h=32.0, rim_w=R_inner * 0.12,
                      breach_theta=0.0, breach_sigma=0.65, breach_extend=650.0)

    # ---- class map ------------------------------------------------------
    # 0 = flat regolith, 1 = ice (inner floor), 2 = rocky rim/ejecta
    cls = np.zeros((H, W), dtype=np.int8)
    inner_floor = r_inner < (R_inner * 0.45)
    cls[inner_floor] = 1
    rim_zone = (
        ((r_inner > R_inner * 0.45) & (r_inner < R_inner * 1.15))
        | (np.abs(r_main - R_main) < R_main * 0.12)
    )
    cls[rim_zone & (cls == 0)] = 2
    # scattered boulders -> small rocky patches mostly near rims
    n_boulders = 240
    bcx = rng.integers(0, W, n_boulders)
    bcy = rng.integers(0, H, n_boulders)
    boulder_field = np.zeros((H, W), dtype=np.float32)
    for x, y in zip(bcx, bcy):
        near_rim = (abs(r_inner[y, x] - R_inner) < 80) or (abs(r_main[y, x] - R_main) < 90)
        if near_rim or rng.random() < 0.15:
            elev[y, x] += rng.uniform(1.5, 4.0)
            boulder_field[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] += 1.0

    # ---- illumination geometry (low polar sun) -------------------------
    n_sun = 24
    sun = SunGeometry(
        azimuth_deg=np.linspace(0, 360, n_sun, endpoint=False),
        elevation_deg=np.full(n_sun, 1.5),  # grazing polar sun
    )

    # ---- ground-truth ice fraction (needed before backscatter) ---------
    # ice fraction is highest toward the coldest centre of the inner floor and
    # is capped at a plausible pore-filling level.
    ice = cls == 1
    ice_fraction = np.zeros((H, W), np.float32)
    frac = 0.40 * np.clip(1 - (r_inner / (R_inner * 0.45)) ** 2, 0, 1)
    ice_fraction[ice] = np.clip(frac[ice] + rng.normal(0, 0.02, ice.sum()), 0.04, 0.55)

    # ---- DFSAR Stokes (L & S) ------------------------------------------
    # base backscatter level
    s0_base = 0.25 + 0.05 * rng.standard_normal((H, W)).astype(np.float32)
    s0_base = np.clip(s0_base, 0.05, None)

    cpr = np.empty((H, W), np.float32)
    dop = np.empty((H, W), np.float32)
    s0_L = s0_base.copy()
    s0_S = s0_base.copy()

    # flat regolith: low CPR, surface(single)-dominated, moderate-high DOP
    flat = cls == 0
    cpr[flat] = rng.normal(0.45, 0.07, flat.sum())
    dop[flat] = rng.normal(0.55, 0.06, flat.sum())

    # ice: high CPR, LOW DOP, volume-dominated; STRONGER in L (buried).
    # Backscatter ENCODES abundance: more buried ice -> stronger L volume
    # return and weaker S -> the L/S contrast scales monotonically with ice
    # fraction (this is the signal the volume inversion later recovers).
    f_ice = ice_fraction[ice]
    cpr[ice] = rng.normal(1.18, 0.06, ice.sum())
    dop[ice] = rng.normal(0.10, 0.015, ice.sum())
    s0_L[ice] *= (1.0 + 4.0 * f_ice) * rng.normal(1.0, 0.06, ice.sum())  # grows with ice
    s0_S[ice] *= (1.0 + 0.6 * f_ice) * rng.normal(1.0, 0.06, ice.sum())  # weak in S

    # rocky rim: high CPR, HIGH DOP, double-bounce dominated; similar L & S
    rock = cls == 2
    cpr[rock] = rng.normal(1.30, 0.10, rock.sum())
    dop[rock] = rng.normal(0.68, 0.07, rock.sum())
    s0_L[rock] *= rng.normal(1.5, 0.2, rock.sum())
    s0_S[rock] *= rng.normal(1.45, 0.2, rock.sum())

    cpr = np.clip(cpr, 0.1, 2.5)
    dop = np.clip(dop, 0.02, 0.98)

    stokes_L = stokes_from_targets(s0_L, cpr, dop, rng)
    # S-band: same CPR/DOP character but its own speckle realisation
    stokes_S = stokes_from_targets(s0_S, cpr, dop, rng)
    stokes_L = _apply_speckle(stokes_L, looks=8.0, rng=rng)
    stokes_S = _apply_speckle(stokes_S, looks=8.0, rng=rng)

    radar_l = RadarProduct(grid=grid, stokes=stokes_L, band="L",
                           meta={"wavelength_cm": 24.0})
    radar_s = RadarProduct(grid=grid, stokes=stokes_S, band="S",
                           meta={"wavelength_cm": 12.0})

    # ---- OHRC optical ---------------------------------------------------
    # crude illumination: pixels in the deep crater floor are in shadow (NaN).
    # (the proper horizon model lives in illumination/psr.py and is used for
    #  features; here we just synthesise a plausible optical image.)
    shadow = (r_main < R_main * 0.9) & (elev < (np.percentile(elev, 35)))
    albedo = 0.12 + 0.02 * rng.standard_normal((H, W)).astype(np.float32)
    refl = np.clip(albedo + 0.20 * boulder_field, 0, 1).astype(np.float32)
    refl[shadow] = np.nan  # optical is blind inside permanent shadow
    optical = OpticalProduct(grid=grid, reflectance=refl,
                             meta={"resolution_m": 0.3})

    # ---- ground truth ---------------------------------------------------
    truth_ice_mask = ice.astype(np.uint8)
    pixel_area = grid.pixel_area_m2()
    truth_volume = float(ice_fraction.sum() * pixel_area * cfg.volume.depth_m)

    dem = DEMProduct(grid=grid, elevation=elev)
    return Scene(
        grid=grid,
        radar_l=radar_l,
        radar_s=radar_s,
        optical=optical,
        dem=dem,
        sun=sun,
        truth={
            "class_map": cls,
            "ice_mask": truth_ice_mask,
            "ice_fraction": ice_fraction,
            "volume_m3": truth_volume,
            "boulder_field": boulder_field,
        },
    )
