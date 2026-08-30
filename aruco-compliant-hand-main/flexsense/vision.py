from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build has no aruco module. Install opencv-contrib-python."
        )
    return cv2


class MarkerDetector:
    """Detects markers and, when calibrated, returns metric-ready corners.

    Lens distortion breaks the pinhole assumption the reference homography
    relies on, biasing every millimetre it reports. The correction is applied to
    the handful of detected corners rather than by remapping the whole frame:
    it is far cheaper, and it leaves detection running on the original sharp
    pixels instead of an interpolated copy.

    Only the `by_id` mapping is corrected. The raw `corners` are returned
    untouched so overlays still line up with the unmodified image.
    """

    def __init__(self, dictionary_name: str, camera_matrix=None, dist_coeffs=None,
                 image_size: tuple[int, int] | None = None):
        cv2 = require_cv2()
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._cv2 = cv2
        self._detector = (
            cv2.aruco.ArucoDetector(self.dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self._parameters = parameters
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix, float)
        self.dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, float)
        self.image_size = image_size

    @property
    def undistorting(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def _undistort(self, points: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        fixed = cv2.undistortPoints(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
            self.camera_matrix, self.dist_coeffs, P=self.camera_matrix,
        )
        return fixed.reshape(-1, 2).astype(float)

    def detect(self, frame: np.ndarray) -> tuple[dict[int, np.ndarray], list[np.ndarray], np.ndarray | None]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, _rejected = self._cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self._parameters
            )
        by_id: dict[int, np.ndarray] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                quad = np.asarray(marker_corners, dtype=float).reshape(4, 2)
                if self.undistorting:
                    quad = self._undistort(quad)
                by_id[int(marker_id)] = quad
        return by_id, corners, ids


def open_capture(source: int | str, width: int, height: int):
    cv2 = require_cv2()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def parse_source(value: str | None, configured: int | str) -> int | str:
    if value is None:
        return configured
    if value.isdigit():
        return int(value)
    return str(Path(value))



DEFAULT_INTRINSICS = "calibration/camera_intrinsics.json"


def build_detector(dictionary_name: str, intrinsics_path: str | Path | None = None,
                   frame_size: tuple[int, int] | None = None):
    """Build a detector, wiring in lens correction when a calibration exists.

    Returns (detector, note). Every measurement path goes through here so the
    tracker and the force calibration cannot end up disagreeing about whether
    distortion was removed - a force model fitted on uncorrected deflections and
    applied to corrected ones would be worse than either alone.
    """
    from .camera_calib import load_intrinsics

    if intrinsics_path is None:
        return MarkerDetector(dictionary_name), "lens correction OFF (not requested)"
    loaded = load_intrinsics(intrinsics_path)
    if loaded is None:
        return (MarkerDetector(dictionary_name),
                f"lens correction OFF - no calibration at {intrinsics_path}; "
                "millimetre readings will be biased by several percent")
    camera_matrix, dist_coeffs, size = loaded
    if frame_size is not None and tuple(frame_size) != tuple(size):
        return (MarkerDetector(dictionary_name),
                f"lens correction OFF - calibration is for {size[0]}x{size[1]}, "
                f"camera is {frame_size[0]}x{frame_size[1]}")
    return (MarkerDetector(dictionary_name, camera_matrix, dist_coeffs, size),
            f"lens correction ON (fx={camera_matrix[0, 0]:.1f}, from {intrinsics_path})")
