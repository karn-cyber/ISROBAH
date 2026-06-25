"""Configuration loader. All physical constants and weights live here as
priors/ranges, not magic numbers — uncertainty is a feature, not an
afterthought."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridCfg:
    res_m: float = 10.0
    height: int = 256
    width: int = 256


@dataclass
class PathsCfg:
    dfsar_l: str | None = None
    dfsar_s: str | None = None
    ohrc: str | None = None
    dem: str | None = None
    sun_geometry: str | None = None
    outdir: str = "data/outputs"


@dataclass
class DetectionCfg:
    cpr_ice_min: float = 1.0
    dop_ice_max: float = 0.13
    use_unet: bool = False  # optional torch upgrade; off by default (CPU/portable)
    unet_epochs: int = 30
    mc_dropout_passes: int = 20
    target_prob: float = 0.6  # threshold for the high-confidence target mask


@dataclass
class VolumeCfg:
    depth_m: float = 5.0
    n_mc: int = 2000
    eps_ice: float = 3.15
    eps_regolith: float = 2.7
    regolith_density_prior: tuple[float, float] = (1400.0, 1900.0)  # kg/m^3
    porosity_prior: tuple[float, float] = (0.40, 0.55)
    loss_tangent_prior: tuple[float, float] = (0.002, 0.008)
    n_depth_bins: int = 5


@dataclass
class LandingCfg:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "slope": 0.22,
            "roughness": 0.18,
            "illumination": 0.20,
            "proximity": 0.30,
            "comms": 0.10,
        }
    )
    max_slope_deg: float = 12.0
    n_sites: int = 5
    min_separation_m: float = 200.0


@dataclass
class PlanningCfg:
    method: str = "dstar_lite"  # astar | dstar_lite
    battery_Wh: float = 2000.0       # VIPER-class battery (Wh)
    thermal_limit_min: float = 720.0  # battery-warmed survival in shadow (min)
    rover_speed_mps: float = 0.10
    base_draw_W: float = 40.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "slope": 0.30,
            "hazard": 0.30,
            "shadow_power": 0.25,
            "thermal": 0.10,
            "distance": 0.05,
        }
    )
    max_slope_deg: float = 25.0


@dataclass
class Config:
    mode: str = "synthetic"  # synthetic | real
    seed: int = 42
    grid: GridCfg = field(default_factory=GridCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    detection: DetectionCfg = field(default_factory=DetectionCfg)
    volume: VolumeCfg = field(default_factory=VolumeCfg)
    landing: LandingCfg = field(default_factory=LandingCfg)
    planning: PlanningCfg = field(default_factory=PlanningCfg)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        cfg.mode = raw.get("mode", cfg.mode)
        cfg.seed = raw.get("seed", cfg.seed)
        if "grid" in raw:
            cfg.grid = GridCfg(**{**cfg.grid.__dict__, **raw["grid"]})
        if "paths" in raw:
            cfg.paths = PathsCfg(**{**cfg.paths.__dict__, **raw["paths"]})
        if "detection" in raw:
            cfg.detection = DetectionCfg(**{**cfg.detection.__dict__, **raw["detection"]})
        if "volume" in raw:
            v = {**cfg.volume.__dict__, **raw["volume"]}
            for k in ("regolith_density_prior", "porosity_prior", "loss_tangent_prior"):
                if isinstance(v.get(k), list):
                    v[k] = tuple(v[k])
            cfg.volume = VolumeCfg(**v)
        if "landing" in raw:
            cfg.landing = LandingCfg(**{**cfg.landing.__dict__, **raw["landing"]})
        if "planning" in raw:
            cfg.planning = PlanningCfg(**{**cfg.planning.__dict__, **raw["planning"]})
        return cfg
