from __future__ import annotations

import csv
import json
import socket
import sys
import time
from pathlib import Path
from typing import TextIO

from .estimator import FrameMeasurement


CSV_FIELDS = [
    "timestamp_ns",
    "monotonic_ns",
    "status",
    "sensor",
    "valid",
    "stale",
    "missing_frames",
    "markers_detected",
    "marker_count",
    "tip_dx_mm",
    "tip_dy_mm",
    "normal_deflection_mm",
    "shear_deflection_mm",
    "tip_rotation_deg",
    "curvature_per_mm",
    "force_normal_n",
    "force_shear_n",
]


class CsvMeasurementWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._rows_since_flush = 0

    def write(self, frame: FrameMeasurement) -> None:
        for sensor_name, measurement in frame.sensors.items():
            row = {
                "timestamp_ns": frame.timestamp_ns,
                "monotonic_ns": frame.monotonic_ns,
                "status": frame.status,
                "sensor": sensor_name,
            }
            values = measurement.to_dict()
            values.pop("marker_deltas", None)
            row.update(values)
            self._writer.writerow(row)
            self._rows_since_flush += 1
        if self._rows_since_flush >= 30:
            self._handle.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self) -> "CsvMeasurementWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class MeasurementPublisher:
    def __init__(
        self,
        json_stdout: bool,
        stdout_hz: float,
        udp_host: str | None,
        udp_port: int | None,
    ):
        self.json_stdout = json_stdout
        self.period_s = 1.0 / max(stdout_hz, 0.1)
        self.last_stdout_s = 0.0
        self.udp_address = (udp_host, udp_port) if udp_host and udp_port else None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_address else None

    def publish(self, frame: FrameMeasurement) -> None:
        payload = json.dumps(frame.to_dict(), separators=(",", ":"), allow_nan=False)
        now = time.monotonic()
        if self.json_stdout and now - self.last_stdout_s >= self.period_s:
            print(payload, file=sys.stdout, flush=True)
            self.last_stdout_s = now
        if self.socket is not None and self.udp_address is not None:
            self.socket.sendto(payload.encode("utf-8"), self.udp_address)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()

