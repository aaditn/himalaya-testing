"""Reusable configuration and MuJoCo heightfield generation for slope tracks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ANGLE_RANGE_DEG = (5.0, 45.0)
FRICTION_RANGE = (0.2, 1.5)
ROUGHNESS_RANGE_M = (0.0, 0.05)


@dataclass(frozen=True)
class TrackConfig:
    """All user-tunable track parameters, independent of policy settings."""

    slope_degrees: float = 30.0
    friction: float = 0.9
    roughness_m: float = 0.02
    flat_start_enabled: bool = True
    flat_start_length_m: float = 2.0
    ramp_surface_length_m: float = 6.0
    summit_length_m: float = 1.5
    width_m: float = 1.2
    seed: int = 1

    def __post_init__(self) -> None:
        _bounded("slope_degrees", self.slope_degrees, *ANGLE_RANGE_DEG)
        _bounded("friction", self.friction, *FRICTION_RANGE)
        _bounded("roughness_m", self.roughness_m, *ROUGHNESS_RANGE_M)
        _bounded("flat_start_length_m", self.flat_start_length_m, 0.5, 4.0)
        _bounded("ramp_surface_length_m", self.ramp_surface_length_m, 2.0, 10.0)
        _bounded("summit_length_m", self.summit_length_m, 0.5, 4.0)
        _bounded("width_m", self.width_m, 0.8, 2.0)
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        if not 0 <= int(self.seed) <= 2**31 - 1:
            raise ValueError("seed must be between 0 and 2147483647")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrackConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown track option(s): {', '.join(unknown)}")
        return cls(**dict(values))

    @property
    def flat_start_m(self) -> float:
        return self.flat_start_length_m if self.flat_start_enabled else 0.0

    @property
    def slope_radians(self) -> float:
        return math.radians(self.slope_degrees)

    @property
    def ramp_run_m(self) -> float:
        return self.ramp_surface_length_m * math.cos(self.slope_radians)

    @property
    def rise_m(self) -> float:
        return self.ramp_surface_length_m * math.sin(self.slope_radians)

    @property
    def total_length_m(self) -> float:
        return self.flat_start_m + self.ramp_run_m + self.summit_length_m

    @property
    def minimum_static_friction(self) -> float:
        return math.tan(self.slope_radians)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "flat_start_m": self.flat_start_m,
            "ramp_run_m": self.ramp_run_m,
            "rise_m": self.rise_m,
            "total_length_m": self.total_length_m,
            "minimum_static_friction": self.minimum_static_friction,
            "static_friction_margin": self.friction - self.minimum_static_friction,
        }


@dataclass(frozen=True)
class HeightfieldSpec:
    """Generated heightfield plus its physical MuJoCo dimensions."""

    data: np.ndarray
    half_x_m: float
    half_y_m: float
    z_top_m: float
    base_depth_m: float
    track_start_x_m: float
    ramp_start_x_m: float
    ramp_end_x_m: float
    summit_end_x_m: float


def _bounded(name: str, value: float, lower: float, upper: float) -> None:
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")


def load_track_config(path: str | Path) -> TrackConfig:
    config_path = Path(path)
    if not config_path.exists():
        return TrackConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("track config must contain a JSON object")
    return TrackConfig.from_mapping(payload)


def save_track_config(config: TrackConfig, path: str | Path) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config_path)
    return config_path


def ideal_track_height(config: TrackConfig, x_m: np.ndarray | float) -> np.ndarray:
    """Return the piecewise-flat/ramp/summit height without roughness."""

    x = np.asarray(x_m, dtype=float)
    start = -0.5 * config.total_length_m
    ramp_start = start + config.flat_start_m
    ramp_end = ramp_start + config.ramp_run_m
    uphill = np.clip(x - ramp_start, 0.0, config.ramp_run_m)
    return uphill * math.tan(config.slope_radians)


def generate_heightfield(
    config: TrackConfig,
    *,
    nrow: int = 129,
    ncol: int = 257,
) -> HeightfieldSpec:
    """Build a deterministic, bounded slope-track heightfield.

    The flat reset area remains smooth. Roughness fades in over the first
    30 cm of the ramp so a spawned robot never begins on a discontinuity.
    """

    if nrow < 3 or ncol < 3:
        raise ValueError("heightfield needs at least 3 rows and 3 columns")
    padding_x = 0.5
    half_x = 0.5 * config.total_length_m + padding_x
    half_y = max(0.5 * config.width_m, 0.4) + 0.15
    x = np.linspace(-half_x, half_x, ncol)
    y = np.linspace(-half_y, half_y, nrow)
    xx, yy = np.meshgrid(x, y)
    base = ideal_track_height(config, xx)

    start = -0.5 * config.total_length_m
    ramp_start = start + config.flat_start_m
    ramp_end = ramp_start + config.ramp_run_m
    summit_end = ramp_end + config.summit_length_m

    roughness = np.zeros_like(base)
    if config.roughness_m > 0.0:
        rng = np.random.default_rng(config.seed)
        phase = rng.uniform(0.0, 2.0 * math.pi, size=4)
        texture = (
            0.44 * np.sin(8.0 * xx + 5.0 * yy + phase[0])
            + 0.27 * np.sin(15.0 * xx - 9.0 * yy + phase[1])
            + 0.18 * np.sin(24.0 * xx + 13.0 * yy + phase[2])
            + 0.11 * np.sin(37.0 * xx - 19.0 * yy + phase[3])
        )
        fade = np.clip((xx - ramp_start) / 0.30, 0.0, 1.0)
        within_course = (xx >= ramp_start) & (xx <= summit_end)
        cross_edge = np.clip((0.5 * config.width_m - np.abs(yy)) / 0.08, 0.0, 1.0)
        roughness = config.roughness_m * texture * fade * within_course * cross_edge

    heights = base + roughness
    heights -= float(np.min(heights))
    z_top = max(float(np.max(heights)) + 0.05, 0.10)
    normalized = np.clip(heights / z_top, 0.0, 1.0)
    return HeightfieldSpec(
        data=normalized.astype(np.float64),
        half_x_m=half_x,
        half_y_m=half_y,
        z_top_m=z_top,
        base_depth_m=0.10,
        track_start_x_m=start,
        ramp_start_x_m=ramp_start,
        ramp_end_x_m=ramp_end,
        summit_end_x_m=summit_end,
    )


def apply_track_to_model(model: Any, config: TrackConfig) -> HeightfieldSpec:
    """Apply track geometry and friction to an already loaded MuJoCo model."""

    if model.nhfield != 1:
        raise ValueError("custom track scene must contain exactly one heightfield")
    rows = int(model.hfield_nrow[0])
    cols = int(model.hfield_ncol[0])
    spec = generate_heightfield(config, nrow=rows, ncol=cols)
    address = int(model.hfield_adr[0])
    model.hfield_data[address : address + rows * cols] = spec.data.ravel()
    model.hfield_size[0] = (
        spec.half_x_m,
        spec.half_y_m,
        spec.z_top_m,
        spec.base_depth_m,
    )
    floor_id = model.geom("floor").id
    model.geom_friction[floor_id, :2] = config.friction
    for side in ("left", "right"):
        try:
            pair_id = model.pair(f"{side}_foot_floor").id
        except KeyError:
            continue
        model.pair_friction[pair_id, :2] = config.friction
    return spec

