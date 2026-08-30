from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import AppConfig, SensorConfig
from .geometry import (
    axis_component,
    marker_center_and_angle,
    reference_homography,
    signed_tip_curvature,
    wrapped_angle_delta_deg,
)
from .models import ForceModelSet


@dataclass
class MarkerDelta:
    marker_id: int
    dx_mm: float
    dy_mm: float


@dataclass
class SensorMeasurement:
    valid: bool
    stale: bool = False
    missing_frames: int = 0
    markers_detected: int = 0
    marker_count: int = 0
    tip_dx_mm: float | None = None
    tip_dy_mm: float | None = None
    normal_deflection_mm: float | None = None
    shear_deflection_mm: float | None = None
    tip_rotation_deg: float | None = None
    curvature_per_mm: float | None = None
    force_normal_n: float | None = None
    force_shear_n: float | None = None
    marker_deltas: list[MarkerDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameMeasurement:
    timestamp_ns: int
    monotonic_ns: int
    status: str
    zero_progress: int
    zero_target: int
    reference_visible: bool
    sensors: dict[str, SensorMeasurement]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "monotonic_ns": self.monotonic_ns,
            "status": self.status,
            "zero_progress": self.zero_progress,
            "zero_target": self.zero_target,
            "reference_visible": self.reference_visible,
            "sensors": {name: value.to_dict() for name, value in self.sensors.items()},
        }


@dataclass
class _MarkerPose2D:
    center_mm: np.ndarray
    angle_deg: float


def _circular_mean_deg(values: list[float]) -> float:
    radians = np.radians(values)
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


class PlanarDeformationEstimator:
    """Estimate finger-marker motion in a rigid marker coordinate frame."""

    def __init__(self, config: AppConfig, force_models: ForceModelSet | None = None):
        self.config = config
        self.force_models = force_models or ForceModelSet()
        self._zero_frames: list[dict[int, _MarkerPose2D]] = []
        self._baseline: dict[int, _MarkerPose2D] | None = None
        self._ema: dict[tuple[str, str], float] = {}
        self._last: dict[str, SensorMeasurement] = {}
        self._missing: defaultdict[str, int] = defaultdict(int)

    @property
    def is_zeroed(self) -> bool:
        return self._baseline is not None

    @property
    def zero_progress(self) -> int:
        return min(len(self._zero_frames), self.config.tracking.zero_samples)

    def reset_zero(self) -> None:
        self._zero_frames.clear()
        self._baseline = None
        self._ema.clear()
        self._last.clear()
        self._missing.clear()

    def _map_markers(self, corners_by_id: dict[int, np.ndarray]) -> dict[int, _MarkerPose2D]:
        reference = corners_by_id[self.config.markers.reference_id]
        homography = reference_homography(reference, self.config.markers.size_mm)
        mapped: dict[int, _MarkerPose2D] = {}
        for marker_id, corners in corners_by_id.items():
            if marker_id == self.config.markers.reference_id:
                continue
            center, angle = marker_center_and_angle(corners, homography)
            mapped[marker_id] = _MarkerPose2D(center_mm=center, angle_deg=angle)
        return mapped

    def _finish_zero(self) -> None:
        baseline: dict[int, _MarkerPose2D] = {}
        finger_ids = self.config.required_marker_ids - {self.config.markers.reference_id}
        for marker_id in finger_ids:
            centers = np.asarray(
                [frame[marker_id].center_mm for frame in self._zero_frames], dtype=float
            )
            angles = [frame[marker_id].angle_deg for frame in self._zero_frames]
            baseline[marker_id] = _MarkerPose2D(
                center_mm=np.median(centers, axis=0),
                angle_deg=_circular_mean_deg(angles),
            )
        self._baseline = baseline

    def _smooth(self, sensor_name: str, field_name: str, value: float | None) -> float | None:
        if value is None:
            return None
        key = (sensor_name, field_name)
        previous = self._ema.get(key)
        filtered = value if previous is None else (
            self.config.tracking.ema_alpha * value
            + (1.0 - self.config.tracking.ema_alpha) * previous
        )
        self._ema[key] = float(filtered)
        return float(filtered)

    def _missing_measurement(self, sensor: SensorConfig, visible_count: int) -> SensorMeasurement:
        self._missing[sensor.name] += 1
        if (
            sensor.name in self._last
            and self._missing[sensor.name] <= self.config.tracking.max_missing_frames
        ):
            held = copy.deepcopy(self._last[sensor.name])
            held.valid = False
            held.stale = True
            held.missing_frames = self._missing[sensor.name]
            held.markers_detected = visible_count
            return held
        return SensorMeasurement(
            valid=False,
            stale=False,
            missing_frames=self._missing[sensor.name],
            markers_detected=visible_count,
            marker_count=len(sensor.marker_ids),
        )

    def _measure_sensor(
        self, sensor: SensorConfig, mapped: dict[int, _MarkerPose2D]
    ) -> SensorMeasurement:
        assert self._baseline is not None
        visible_ids = [marker_id for marker_id in sensor.marker_ids if marker_id in mapped]
        tip_id = sensor.marker_ids[-1]
        if tip_id not in mapped:
            return self._missing_measurement(sensor, len(visible_ids))

        self._missing[sensor.name] = 0
        tip_delta = mapped[tip_id].center_mm - self._baseline[tip_id].center_mm
        tip_dx = self._smooth(sensor.name, "tip_dx_mm", float(tip_delta[0]))
        tip_dy = self._smooth(sensor.name, "tip_dy_mm", float(tip_delta[1]))
        normal = self._smooth(
            sensor.name,
            "normal_deflection_mm",
            axis_component(tip_delta, sensor.normal_axis, sensor.normal_sign),
        )
        shear = self._smooth(
            sensor.name,
            "shear_deflection_mm",
            axis_component(tip_delta, sensor.shear_axis, sensor.shear_sign),
        )
        rotation = self._smooth(
            sensor.name,
            "tip_rotation_deg",
            wrapped_angle_delta_deg(
                mapped[tip_id].angle_deg, self._baseline[tip_id].angle_deg
            ),
        )

        curvature = None
        if len(visible_ids) == len(sensor.marker_ids) and len(sensor.marker_ids) >= 3:
            current_points = np.asarray(
                [mapped[marker_id].center_mm for marker_id in sensor.marker_ids]
            )
            baseline_points = np.asarray(
                [self._baseline[marker_id].center_mm for marker_id in sensor.marker_ids]
            )
            current_curvature = signed_tip_curvature(current_points)
            baseline_curvature = signed_tip_curvature(baseline_points)
            if current_curvature is not None and baseline_curvature is not None:
                curvature = self._smooth(
                    sensor.name,
                    "curvature_per_mm",
                    current_curvature - baseline_curvature,
                )

        marker_deltas = []
        for marker_id in visible_ids:
            delta = mapped[marker_id].center_mm - self._baseline[marker_id].center_mm
            marker_deltas.append(
                MarkerDelta(marker_id=marker_id, dx_mm=float(delta[0]), dy_mm=float(delta[1]))
            )

        result = SensorMeasurement(
            valid=True,
            stale=False,
            missing_frames=0,
            markers_detected=len(visible_ids),
            marker_count=len(sensor.marker_ids),
            tip_dx_mm=tip_dx,
            tip_dy_mm=tip_dy,
            normal_deflection_mm=normal,
            shear_deflection_mm=shear,
            tip_rotation_deg=rotation,
            curvature_per_mm=curvature,
            force_normal_n=self.force_models.evaluate(sensor.name, "normal", normal),
            force_shear_n=self.force_models.evaluate(sensor.name, "shear", shear),
            marker_deltas=marker_deltas,
        )
        self._last[sensor.name] = copy.deepcopy(result)
        return result

    def observe(self, corners_by_id: dict[int, np.ndarray]) -> FrameMeasurement:
        timestamp_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        reference_id = self.config.markers.reference_id
        if reference_id not in corners_by_id:
            return FrameMeasurement(
                timestamp_ns=timestamp_ns,
                monotonic_ns=monotonic_ns,
                status="reference_missing",
                zero_progress=self.zero_progress,
                zero_target=self.config.tracking.zero_samples,
                reference_visible=False,
                sensors={
                    sensor.name: self._missing_measurement(sensor, 0)
                    for sensor in self.config.sensors
                },
            )

        mapped = self._map_markers(corners_by_id)
        if self._baseline is None:
            needed = self.config.required_marker_ids - {reference_id}
            if needed.issubset(mapped):
                self._zero_frames.append({marker_id: mapped[marker_id] for marker_id in needed})
                if len(self._zero_frames) >= self.config.tracking.zero_samples:
                    self._finish_zero()
            status = "tracking" if self._baseline is not None else "zeroing"
            return FrameMeasurement(
                timestamp_ns=timestamp_ns,
                monotonic_ns=monotonic_ns,
                status=status,
                zero_progress=self.zero_progress,
                zero_target=self.config.tracking.zero_samples,
                reference_visible=True,
                sensors={
                    sensor.name: SensorMeasurement(
                        valid=False,
                        markers_detected=sum(
                            marker_id in mapped for marker_id in sensor.marker_ids
                        ),
                        marker_count=len(sensor.marker_ids),
                    )
                    for sensor in self.config.sensors
                },
            )

        return FrameMeasurement(
            timestamp_ns=timestamp_ns,
            monotonic_ns=monotonic_ns,
            status="tracking",
            zero_progress=self.zero_progress,
            zero_target=self.config.tracking.zero_samples,
            reference_visible=True,
            sensors={
                sensor.name: self._measure_sensor(sensor, mapped)
                for sensor in self.config.sensors
            },
        )
