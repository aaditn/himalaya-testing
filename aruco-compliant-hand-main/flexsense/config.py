from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_AXES = {"x", "y"}


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class MarkerConfig:
    dictionary: str = "DICT_4X4_50"
    size_mm: float = 14.0
    reference_id: int = 0


@dataclass(frozen=True)
class TrackingConfig:
    zero_samples: int = 40
    ema_alpha: float = 0.25
    max_missing_frames: int = 8


@dataclass(frozen=True)
class SensorConfig:
    name: str
    marker_ids: tuple[int, ...]
    normal_axis: str
    normal_sign: float
    shear_axis: str
    shear_sign: float


@dataclass(frozen=True)
class OutputConfig:
    csv_path: Path
    calibration_file: Path
    json_stdout: bool = False
    stdout_hz: float = 10.0
    udp_host: str | None = None
    udp_port: int | None = None


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    markers: MarkerConfig
    tracking: TrackingConfig
    sensors: tuple[SensorConfig, ...]
    output: OutputConfig

    @property
    def required_marker_ids(self) -> set[int]:
        ids = {self.markers.reference_id}
        for sensor in self.sensors:
            ids.update(sensor.marker_ids)
        return ids


def _camera_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return str(value)


def load_config(path: str | Path) -> AppConfig:
    from .paths import resolve

    config_path = resolve(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    camera_raw = raw.get("camera", {})
    marker_raw = raw.get("markers", {})
    tracking_raw = raw.get("tracking", {})
    output_raw = raw.get("output", {})

    sensors: list[SensorConfig] = []
    names: set[str] = set()
    marker_ids: set[int] = set()
    for item in raw.get("sensors", []):
        name = str(item["name"])
        ids = tuple(int(marker_id) for marker_id in item["marker_ids"])
        normal_axis = str(item.get("normal_axis", "x")).lower()
        shear_axis = str(item.get("shear_axis", "y")).lower()
        if not ids:
            raise ValueError(f"Sensor {name!r} needs at least one marker ID")
        if name in names:
            raise ValueError(f"Duplicate sensor name: {name}")
        if marker_ids.intersection(ids):
            raise ValueError(f"Marker IDs cannot be shared between sensors: {ids}")
        if normal_axis not in VALID_AXES or shear_axis not in VALID_AXES:
            raise ValueError("Planar mode supports only x and y measurement axes")
        if normal_axis == shear_axis:
            raise ValueError(f"Sensor {name!r} normal and shear axes must differ")

        names.add(name)
        marker_ids.update(ids)
        sensors.append(
            SensorConfig(
                name=name,
                marker_ids=ids,
                normal_axis=normal_axis,
                normal_sign=float(item.get("normal_sign", 1.0)),
                shear_axis=shear_axis,
                shear_sign=float(item.get("shear_sign", 1.0)),
            )
        )

    if not sensors:
        raise ValueError("At least one sensor must be configured")

    reference_id = int(marker_raw.get("reference_id", 0))
    if reference_id in marker_ids:
        raise ValueError("The rigid reference marker ID must not be a finger marker ID")

    zero_samples = int(tracking_raw.get("zero_samples", 40))
    ema_alpha = float(tracking_raw.get("ema_alpha", 0.25))
    if zero_samples < 3:
        raise ValueError("tracking.zero_samples must be at least 3")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("tracking.ema_alpha must be in (0, 1]")

    return AppConfig(
        camera=CameraConfig(
            source=_camera_source(camera_raw.get("source", 0)),
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
        ),
        markers=MarkerConfig(
            dictionary=str(marker_raw.get("dictionary", "DICT_4X4_50")),
            size_mm=float(marker_raw.get("size_mm", 14.0)),
            reference_id=reference_id,
        ),
        tracking=TrackingConfig(
            zero_samples=zero_samples,
            ema_alpha=ema_alpha,
            max_missing_frames=int(tracking_raw.get("max_missing_frames", 8)),
        ),
        sensors=tuple(sensors),
        output=OutputConfig(
            csv_path=Path(output_raw.get("csv_path", "data/flexsense.csv")),
            calibration_file=Path(
                output_raw.get("calibration_file", "calibration/force_models.json")
            ),
            json_stdout=bool(output_raw.get("json_stdout", False)),
            stdout_hz=float(output_raw.get("stdout_hz", 10.0)),
            udp_host=output_raw.get("udp_host"),
            udp_port=(
                int(output_raw["udp_port"])
                if output_raw.get("udp_port") is not None
                else None
            ),
        ),
    )

