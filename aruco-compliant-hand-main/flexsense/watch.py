"""Live marker viewer. No tracking, no logging - just show what is detected."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .camera_calib import BOARD_DICTIONARY, BoardSpec, load_intrinsics
from .config import AppConfig
from .hud import append_control_bar
from .vision import MarkerDetector, open_capture, require_cv2

WINDOW = "FlexSense marker view"


def _marker_pixel_size(corners: np.ndarray) -> float:
    """Mean edge length in pixels. Below ~50 px, decoding gets unreliable."""
    pts = np.asarray(corners, dtype=float).reshape(4, 2)
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(edges))


def _square_object_points(size_mm: float) -> np.ndarray:
    """Corners of a square marker in its own frame, in OpenCV's corner order.

    Order is top-left, top-right, bottom-right, bottom-left, matching what
    detectMarkers returns, with +x right, +y up and the marker on z=0.
    """
    half = size_mm / 2.0
    return np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)


def _marker_pose(corners: np.ndarray, size_mm: float, camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray):
    """Full 6-DoF pose of one square marker.

    IPPE_SQUARE is the planar-specific solver; it is both faster and better
    conditioned than the generic iterative one for a four-point square. A single
    small square still carries the planar two-fold ambiguity, so the returned
    rotation can flip when the marker is close to face-on.
    """
    cv2 = require_cv2()
    ok, rvec, tvec = cv2.solvePnP(
        _square_object_points(size_mm),
        np.asarray(corners, dtype=np.float32).reshape(4, 1, 2),
        camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return (rvec, tvec) if ok else None


def _draw_pose_box(frame: np.ndarray, rvec, tvec, size_mm: float,
                   camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                   colour: tuple[int, int, int]) -> None:
    """Project a cube standing on the marker plane, plus its axes."""
    cv2 = require_cv2()
    half = size_mm / 2.0
    top = size_mm  # cube as tall as the marker is wide
    box = np.array([
        [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
        [-half, half, top], [half, half, top], [half, -half, top], [-half, -half, top],
    ], dtype=np.float32)
    projected, _ = cv2.projectPoints(box, rvec, tvec, camera_matrix, dist_coeffs)
    pts = projected.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return
    pts = pts.astype(int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
    for a, b in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, 1, cv2.LINE_AA)
    for i in range(4):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 4]), colour, 1, cv2.LINE_AA)
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, size_mm * 0.6, 2)


def _pose_readout(rvec, tvec) -> tuple[float, float]:
    """Distance to the marker in mm, and how far its normal is off the view ray."""
    cv2 = require_cv2()
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    view = tvec.ravel()
    distance = float(np.linalg.norm(view))
    if distance < 1e-6:
        return 0.0, 0.0
    cosine = abs(float(normal @ (view / distance)))
    return distance, float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


# Each extra family costs one more detector pass, so keep the list short.
FAMILY_COLORS = ((60, 220, 60), (240, 170, 40), (200, 120, 220))


def run_watch(config: AppConfig, source: int | str,
              intrinsics_path: str | Path = "calibration/camera_intrinsics.json",
              dictionaries: list[str] | None = None,
              marker_mm: float | None = None,
              board_square_mm: float = 30.0) -> int:
    cv2 = require_cv2()
    # The finger dictionary first, then the calibration board's, so the printed
    # board is a usable test target before any finger markers exist.
    names = dictionaries or [config.markers.dictionary, BOARD_DICTIONARY]
    seen_names: list[str] = []
    for name in names:
        if name not in seen_names:
            seen_names.append(name)
    detectors = [(name, MarkerDetector(name)) for name in seen_names]
    primary = seen_names[0]
    wanted = sorted(config.required_marker_ids)
    reference_id = config.markers.reference_id
    # Physical marker size per family; only affects reported distance, not the
    # shape of the projected box.
    sizes = {name: (marker_mm or config.markers.size_mm) for name in seen_names}
    if BOARD_DICTIONARY in sizes:
        sizes[BOARD_DICTIONARY] = BoardSpec().rescaled(board_square_mm).marker_mm

    loaded = load_intrinsics(intrinsics_path)
    undistort = loaded is not None
    show_pose = loaded is not None
    maps = None
    new_camera_matrix = None
    zero_dist = np.zeros((1, 5))

    capture = open_capture(source, config.camera.width, config.camera.height)
    times: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if undistort and loaded is not None:
                camera_matrix, dist, size = loaded
                if maps is None or maps[0] != frame.shape[:2]:
                    h, w = frame.shape[:2]
                    if (w, h) != size:
                        # Intrinsics belong to the resolution they were measured at.
                        undistort = False
                    else:
                        new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 0)
                        mx, my = cv2.initUndistortRectifyMap(
                            camera_matrix, dist, None, new_K, (w, h), cv2.CV_16SC2)
                        maps = (frame.shape[:2], mx, my)
                        new_camera_matrix = new_K
                if undistort and maps is not None:
                    frame = cv2.remap(frame, maps[1], maps[2], cv2.INTER_LINEAR)

            found: dict[str, dict[int, np.ndarray]] = {}
            for index, (name, det) in enumerate(detectors):
                by_id, corners, ids = det.detect(frame)
                found[name] = by_id
                if ids is not None and len(ids):
                    colour = FAMILY_COLORS[index % len(FAMILY_COLORS)]
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids, colour)
            by_id = found[primary]
            total = sum(len(v) for v in found.values())

            poses: dict[int, tuple[float, float]] = {}
            if show_pose and loaded is not None:
                camera_matrix, dist, _size = loaded
                # An undistorted frame is described by the new matrix with no
                # remaining distortion; passing the original pair here would
                # correct the image twice and bias every pose.
                if undistort and new_camera_matrix is not None:
                    pose_K, pose_d = new_camera_matrix, zero_dist
                else:
                    pose_K, pose_d = camera_matrix, dist
                for index, (name, _det) in enumerate(detectors):
                    colour = FAMILY_COLORS[index % len(FAMILY_COLORS)]
                    for marker_id, quad in found[name].items():
                        pose = _marker_pose(quad, sizes[name], pose_K, pose_d)
                        if pose is None:
                            continue
                        rvec, tvec = pose
                        _draw_pose_box(frame, rvec, tvec, sizes[name], pose_K, pose_d, colour)
                        if name == primary:
                            poses[marker_id] = _pose_readout(rvec, tvec)

            times.append(time.monotonic())
            del times[:-30]
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            head = f"{total} markers   {fps:4.1f} fps"
            if loaded is not None:
                head += (f"   undistort {'ON' if undistort else 'OFF'} (u)"
                         f"   pose {'ON' if show_pose else 'OFF'} (p)")
            else:
                head += "   no calibration found"
            cv2.putText(frame, head, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (40, 220, 40) if total else (40, 80, 240), 2)

            y = 62
            for marker_id in wanted:
                seen = marker_id in by_id
                role = "REF" if marker_id == reference_id else "tip"
                if seen:
                    px = _marker_pixel_size(by_id[marker_id])
                    small = px < 50
                    text = f"ID {marker_id:2d} {role}  OK   {px:5.1f} px"
                    if marker_id in poses:
                        distance, tilt = poses[marker_id]
                        text += f"   {distance:6.1f} mm   tilt {tilt:4.1f} deg"
                    if small:
                        text += "  TOO SMALL"
                    color = (40, 180, 240) if small else (60, 220, 60)
                else:
                    text = f"ID {marker_id:2d} {role}  MISSING"
                    color = (40, 80, 240)
                cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 26

            extra = sorted(set(by_id) - set(wanted))
            if extra:
                cv2.putText(frame, f"unconfigured {primary} IDs: {extra}", (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                y += 24
            for index, (name, det) in enumerate(detectors[1:], start=1):
                hits = found[name]
                colour = FAMILY_COLORS[index % len(FAMILY_COLORS)]
                if hits:
                    pixel_sizes = [_marker_pixel_size(c) for c in hits.values()]
                    label = (f"{name}: {len(hits)} seen, ids {sorted(hits)[:8]}"
                             f"{'...' if len(hits) > 8 else ''}  ~{np.mean(pixel_sizes):.0f} px")
                else:
                    label = f"{name}: none"
                cv2.putText(frame, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            colour if hits else (150, 150, 150), 2 if hits else 1)
                y += 24

            frame = append_control_bar(
                frame,
                (("Q", "quit"), ("U", "undistort"), ("P", "pose boxes")),
                "calibration loaded" if loaded is not None else "no calibration",
            )
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("u") and loaded is not None:
                undistort = not undistort
            if key == ord("p") and loaded is not None:
                show_pose = not show_pose
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()
