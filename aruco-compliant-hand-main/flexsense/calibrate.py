from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AppConfig
from .estimator import PlanarDeformationEstimator
from .models import ForceModelSet
from .vision import DEFAULT_INTRINSICS, build_detector, open_capture, require_cv2


@dataclass(frozen=True)
class CalibrationPoint:
    load_g: float
    force_n: float
    deflection_mm: float
    sample_std_mm: float


def _show_calibration_frame(frame, detector, corners, ids, line: str, display: bool) -> bool:
    if not display:
        return True
    cv2 = require_cv2()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.putText(frame, line, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 240, 240), 2)
    cv2.imshow("FlexSense calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def collect_calibration(
    config: AppConfig,
    source: int | str,
    sensor_name: str,
    axis_name: str,
    loads_g: list[float],
    samples_per_load: int,
    degree: int,
    display: bool,
    intrinsics: str | None = DEFAULT_INTRINSICS,
    prompt: Callable[[str], str] = input,
) -> tuple[list[CalibrationPoint], list[float], float]:
    if sensor_name not in {sensor.name for sensor in config.sensors}:
        raise ValueError(f"Unknown sensor {sensor_name!r}")
    if axis_name not in {"normal", "shear"}:
        raise ValueError("axis must be 'normal' or 'shear'")
    if len(loads_g) < degree + 2:
        raise ValueError("Use at least degree + 2 calibration loads")

    detector, note = build_detector(
        config.markers.dictionary, intrinsics,
        (config.camera.width, config.camera.height),
    )
    print(note, file=sys.stderr)
    estimator = PlanarDeformationEstimator(config)
    capture = open_capture(source, config.camera.width, config.camera.height)
    cv2 = require_cv2()
    try:
        prompt(
            "Remove all load, hold the arm still, and press Enter to capture the zero pose. "
        )
        estimator.reset_zero()
        while not estimator.is_zeroed:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped while collecting the zero pose")
            by_id, corners, ids = detector.detect(frame)
            measurement = estimator.observe(by_id)
            line = f"ZERO {measurement.zero_progress}/{measurement.zero_target}"
            if not _show_calibration_frame(frame, detector, corners, ids, line, display):
                raise KeyboardInterrupt

        points: list[CalibrationPoint] = []
        for load_g in loads_g:
            prompt(
                f"Apply {load_g:g} g along the POSITIVE {axis_name} direction, "
                "wait for creep to settle, then press Enter. "
            )
            for _ in range(12):
                capture.grab()

            samples: list[float] = []
            attempts = 0
            max_attempts = max(samples_per_load * 20, 200)
            while len(samples) < samples_per_load and attempts < max_attempts:
                attempts += 1
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped during calibration")
                by_id, corners, ids = detector.detect(frame)
                measurement = estimator.observe(by_id)
                sensor = measurement.sensors[sensor_name]
                value = (
                    sensor.normal_deflection_mm
                    if axis_name == "normal"
                    else sensor.shear_deflection_mm
                )
                if sensor.valid and not sensor.stale and value is not None:
                    samples.append(float(value))
                line = f"{load_g:g} g  samples {len(samples)}/{samples_per_load}"
                if not _show_calibration_frame(frame, detector, corners, ids, line, display):
                    raise KeyboardInterrupt
            if len(samples) < samples_per_load:
                raise RuntimeError(
                    "Could not see the reference and fingertip markers consistently enough"
                )
            points.append(
                CalibrationPoint(
                    load_g=float(load_g),
                    force_n=float(load_g) * 9.80665 / 1000.0,
                    deflection_mm=float(np.median(samples)),
                    sample_std_mm=float(np.std(samples)),
                )
            )

        x = np.asarray([point.deflection_mm for point in points], dtype=float)
        y = np.asarray([point.force_n for point in points], dtype=float)
        if np.ptp(x) < 0.05:
            raise RuntimeError(
                "Measured deflection span is below 0.05 mm. Enlarge the visible flexure "
                "motion, improve camera resolution, or use a softer flexure."
            )
        coefficients = np.polyfit(x, y, degree).tolist()
        predicted = np.polyval(coefficients, x)
        residual = float(np.sum((y - predicted) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - residual / total if total > 1e-12 else 1.0

        metadata = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "degree": degree,
            "r_squared": r_squared,
            "samples_per_load": samples_per_load,
            "points": [point.__dict__ for point in points],
            "warning": "Valid only for this print, mounting, camera geometry, and load direction.",
        }
        ForceModelSet.update_file(
            config.output.calibration_file,
            sensor_name=sensor_name,
            axis_name=axis_name,
            coefficients=coefficients,
            metadata=metadata,
            clamp_min_n=0.0 if axis_name == "normal" else None,
        )
        return points, coefficients, r_squared
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()

