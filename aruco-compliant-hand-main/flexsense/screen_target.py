"""Full-screen ChArUco target that animates around the display.

The screen is a perfectly flat, correctly-proportioned calibration target, which
removes the two things that most often ruin a printed board: paper curl and
printer rescaling. It cannot remove the need to move the camera. Every corner
shown on the screen lies on one physical plane, so a stationary camera sees a
single homography no matter where the board is drawn, and focal length is not
recoverable from one plane orientation. The animation sweeps the image frame;
the operator supplies the tilt.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .camera_calib import (
    BoardSpec,
    CoverageGrid,
    RadialReach,
    TILT_BINS,
    _sharpness,
    _tilt_bin,
    _tilt_degrees,
    _view_signature,
    build_board,
    horizontal_fov_deg,
    plausible_intrinsics,
    robust_calibrate,
    per_view_errors,
    sanity_report,
    save_intrinsics,
)
from .vision import open_capture, require_cv2

WINDOW = "FlexSense calibration target"

# Screen position (as a fraction of the display) and relative size for each step
# of the sweep. Corners come first so the image edges are reached early.
LAYOUT = [
    (0.50, 0.50, 0.92), (0.22, 0.24, 0.46), (0.78, 0.24, 0.46),
    (0.22, 0.76, 0.46), (0.78, 0.76, 0.46), (0.50, 0.20, 0.55),
    (0.50, 0.80, 0.55), (0.18, 0.50, 0.55), (0.82, 0.50, 0.55),
    (0.50, 0.50, 0.64), (0.32, 0.38, 0.40), (0.68, 0.62, 0.40),
]

POSE_PROMPTS = [
    "hold the camera square to the screen",
    "tilt the camera LEFT about 30 degrees",
    "tilt the camera RIGHT about 30 degrees",
    "tilt the camera UP about 30 degrees",
    "tilt the camera DOWN about 30 degrees",
    "move CLOSER, keep it angled",
    "move BACK, keep it angled",
    "roll the camera about 45 degrees",
]


def _render_target(canvas: np.ndarray, board_image: np.ndarray,
                   cx: float, cy: float, rel: float) -> None:
    cv2 = require_cv2()
    canvas[:] = 255
    height, width = canvas.shape[:2]
    board_h, board_w = board_image.shape[:2]
    target_h = int(rel * height)
    target_w = int(target_h * board_w / board_h)
    if target_w > rel * width:
        target_w = int(rel * width)
        target_h = int(target_w * board_h / board_w)
    # INTER_AREA when shrinking keeps the squares antialiased, which suppresses
    # moire between the display grid and the camera sensor.
    scaled = cv2.resize(board_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x0 = int(np.clip(cx * width - target_w / 2, 0, width - target_w))
    y0 = int(np.clip(cy * height - target_h / 2, 0, height - target_h))
    canvas[y0:y0 + target_h, x0:x0 + target_w] = scaled[:, :, None]


def _draw_coverage_map(canvas: np.ndarray, coverage, reach, strip_top: int) -> None:
    """Miniature of the CAMERA frame, so the operator can see where to aim.

    Without this the operator is blind: the screen shows the target, but nothing
    tells them which part of the camera's own view is still empty.
    """
    cv2 = require_cv2()
    height, width = canvas.shape[:2]
    map_h = int((height - strip_top) * 0.62)
    map_w = int(map_h * 16 / 9)
    x0 = width - map_w - int(width * 0.03)
    y0 = strip_top + int((height - strip_top - map_h) / 2)
    cells = coverage.cells
    cell_w, cell_h = map_w / cells, map_h / cells
    for row in range(cells):
        for col in range(cells):
            a = (x0 + int(col * cell_w), y0 + int(row * cell_h))
            b = (x0 + int((col + 1) * cell_w) - 1, y0 + int((row + 1) * cell_h) - 1)
            filled = coverage.grid[row, col]
            cv2.rectangle(canvas, a, b, (70, 190, 70) if filled else (55, 55, 150), -1)
            cv2.rectangle(canvas, a, b, (30, 30, 30), 1)
    cv2.rectangle(canvas, (x0, y0), (x0 + map_w, y0 + map_h), (230, 230, 230), 2)
    cv2.putText(canvas, f"camera view  edges {reach.fraction * 100:.0f}%",
                (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)


def _hud(canvas: np.ndarray, lines: list[tuple[str, tuple[int, int, int], float]],
         coverage=None, reach=None) -> None:
    cv2 = require_cv2()
    height, width = canvas.shape[:2]
    strip = max(int(height * 0.15), 120)
    cv2.rectangle(canvas, (0, height - strip), (width, height), (28, 28, 28), -1)
    y = height - strip + int(strip * 0.26)
    for text, color, scale in lines:
        cv2.putText(canvas, text, (int(width * 0.03), y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)
        y += int(strip * 0.25)
    if coverage is not None and reach is not None:
        _draw_coverage_map(canvas, coverage, reach, height - strip)


def run_screen_calibration(
    source: int | str,
    width: int,
    height: int,
    output: str | Path,
    spec: BoardSpec | None = None,
    target_views: int = 24,
    min_corners: int = 12,
    diversity: float = 0.12,
    blur_ratio: float = 0.35,
    min_coverage: float = 0.55,
    min_radial_reach: float = 0.75,
    max_view_error_px: float = 1.5,
    fix_principal_point: bool = True,
    frames_dir: str | Path | None = None,
) -> dict[str, Any]:
    cv2 = require_cv2()
    spec = spec or BoardSpec()
    board = build_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    # Rendered once at high resolution, then shrunk per step.
    board_image = board.generateImage((1200, int(1200 * spec.height_mm / spec.width_mm)),
                                      marginSize=40, borderBits=1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(200)
    rect = cv2.getWindowImageRect(WINDOW)
    screen_w, screen_h = (rect[2], rect[3]) if rect and rect[2] > 0 else (1512, 982)
    canvas = np.full((screen_h, screen_w, 3), 255, dtype=np.uint8)

    live = isinstance(source, int)
    capture = open_capture(source, width, height)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    tilts: list[float] = []
    kept_frames: list[np.ndarray] = []
    recent_sharpness: list[float] = []
    coverage: CoverageGrid | None = None
    reach: RadialReach | None = None
    image_size: tuple[int, int] | None = None
    step = 0
    last_step_change = time.monotonic()
    last_capture = 0.0

    def outstanding() -> list[str]:
        items = []
        if len(object_points) < target_views:
            items.append(f"{target_views - len(object_points)} views")
        if coverage is not None and coverage.fraction < min_coverage:
            items.append(f"coverage {coverage.fraction * 100:.0f}/{min_coverage * 100:.0f}%")
        if reach is not None and reach.fraction < min_radial_reach:
            items.append(f"MOVE CLOSER - edges {reach.fraction * 100:.0f}/{min_radial_reach * 100:.0f}%")
        if sum(1 for t in tilts if t >= 25.0) < 4:
            items.append("more tilt >25deg")
        return items

    try:
        while True:
            if not outstanding() and len(object_points) >= target_views:
                break
            if len(object_points) >= target_views * 3:
                break

            ok, frame = capture.read()
            if not ok:
                break
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
                coverage = CoverageGrid(*image_size)
                reach = RadialReach(*image_size)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, _mc, _mi = detector.detectBoard(gray)
            detected = 0 if charuco_ids is None else len(charuco_ids)
            accepted = False
            note = f"searching... {detected} corners"
            tilt = None

            if detected >= min_corners:
                sharp = _sharpness(gray, charuco_corners)
                recent_sharpness.append(sharp)
                del recent_sharpness[:-120]
                reference = float(np.percentile(recent_sharpness, 90))
                obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                tilt = _tilt_degrees(obj_pts, img_pts, *image_size)
                signature = np.append(_view_signature(charuco_corners, *image_size),
                                      (tilt or 0.0) / 45.0)
                novel = all(float(np.linalg.norm(signature - prev)) > diversity
                            for prev in signatures)
                if sharp < blur_ratio * reference:
                    note = "too blurry - hold the camera still"
                elif not novel:
                    note = "already have this view - change angle or distance"
                elif (not live) or time.monotonic() - last_capture > 0.35:
                    object_points.append(obj_pts)
                    image_points.append(img_pts)
                    signatures.append(signature)
                    tilts.append(tilt or 0.0)
                    coverage.add(charuco_corners)
                    reach.add(charuco_corners)
                    if frames_dir is not None:
                        kept_frames.append(frame.copy())
                    last_capture = time.monotonic()
                    accepted = True
                    note = f"captured  (tilt {tilt:.0f} deg)" if tilt else "captured"

            # Advance the sweep on a capture, or if this spot is not working out.
            if accepted or (live and time.monotonic() - last_step_change > 2.5):
                step = (step + 1) % len(LAYOUT)
                last_step_change = time.monotonic()

            cx, cy, rel = LAYOUT[step]
            _render_target(canvas, board_image, cx, cy, rel)
            bins = np.bincount([_tilt_bin(t) for t in tilts], minlength=len(TILT_BINS) - 1)
            prompt = POSE_PROMPTS[(len(object_points) // 3) % len(POSE_PROMPTS)]
            green, amber, grey = (60, 200, 60), (40, 190, 250), (210, 210, 210)
            _hud(canvas, [
                (f"views {len(object_points)}/{target_views}    "
                 f"coverage {coverage.fraction * 100:.0f}%    "
                 f"edges {reach.fraction * 100:.0f}%    "
                 f"tilt {list(map(int, bins))}",
                 green if accepted else grey, 0.9),
                (f"NOW: {prompt}", amber, 0.85),
                (f"{note}    |    q to finish", grey, 0.7),
            ], coverage, reach)
            if accepted:
                cv2.rectangle(canvas, (0, 0), (screen_w - 1, screen_h - 1), (60, 200, 60), 14)
            cv2.imshow(WINDOW, canvas)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if len(object_points) < 6:
        raise RuntimeError(
            f"Only {len(object_points)} usable views; need at least 6. Move the "
            "camera closer to the screen, raise the display brightness, and make "
            "sure no reflection is sitting on the board."
        )

    def fit(objs, imgs):
        rms_, K_, d_, rv_, tv_, sane_ = robust_calibrate(
            objs, imgs, image_size, fix_principal_point
        )
        free = robust_calibrate(objs, imgs, image_size, fix_principal_point=False)
        return rms_, K_, d_, rv_, tv_, free[1], sane_

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, free_K, sane = fit(object_points, image_points)
    errors = per_view_errors(object_points, image_points, rvecs, tvecs,
                             camera_matrix, dist_coeffs)
    threshold = max(max_view_error_px, 2.5 * float(np.median(errors)))
    keep = errors <= threshold
    dropped = int((~keep).sum())
    if dropped and int(keep.sum()) >= 8:
        object_points = [o for o, k in zip(object_points, keep) if k]
        image_points = [i for i, k in zip(image_points, keep) if k]
        kept_frames = [f for f, k in zip(kept_frames, keep) if k]
        tilts = [t for t, k in zip(tilts, keep) if k]
        rms, camera_matrix, dist_coeffs, rvecs, tvecs, free_K, sane = fit(object_points, image_points)
        errors = per_view_errors(object_points, image_points, rvecs, tvecs,
                                 camera_matrix, dist_coeffs)
    else:
        dropped = 0

    if frames_dir is not None and kept_frames:
        directory = Path(frames_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for index, saved in enumerate(kept_frames):
            cv2.imwrite(str(directory / f"screen_{index:03d}.png"), saved)

    saved_path = save_intrinsics(output, camera_matrix, dist_coeffs, image_size, rms,
                                 len(object_points), spec, fix_principal_point)
    report = sanity_report(camera_matrix, image_size, fix_principal_point)
    if not sane:
        ok, why = plausible_intrinsics(camera_matrix, dist_coeffs, image_size)
        report["warnings"].append(
            f"no physically plausible solution found ({why}); do not use this "
            "calibration - capture again with the board nearer the frame edges"
        )
    tilt_bins = np.bincount([_tilt_bin(t) for t in tilts], minlength=len(TILT_BINS) - 1)
    if coverage is not None and coverage.fraction < min_coverage:
        report["warnings"].append(
            f"corners covered only {coverage.fraction * 100:.0f}% of the frame"
        )
    if reach is not None and reach.fraction < min_radial_reach:
        report["warnings"].append(
            f"corners reached only {reach.fraction * 100:.0f}% of the way to the image "
            "corner; radial distortion and focal length are degenerate out there - "
            "move the camera closer so the screen fills the frame"
        )
    if tilt_bins[2] + tilt_bins[3] < 4:
        report["warnings"].append(
            f"only {int(tilt_bins[2] + tilt_bins[3])} views tilted beyond 25 degrees; "
            "a screen cannot supply tilt, you have to move the camera"
        )
    return {
        "saved": str(saved_path),
        "rms_reprojection_error_px": float(rms),
        "views_used": len(object_points),
        "views_dropped_as_outliers": dropped,
        "worst_view_error_px": float(errors.max()),
        "coverage_fraction": coverage.fraction if coverage else 0.0,
        "radial_reach_fraction": reach.fraction if reach else 0.0,
        "tilt_histogram": {
            f"{int(TILT_BINS[i])}-{int(TILT_BINS[i + 1])}deg": int(tilt_bins[i])
            for i in range(len(tilt_bins))
        },
        "principal_point_fixed": bool(fix_principal_point),
        "horizontal_fov_deg": horizontal_fov_deg(float(camera_matrix[0, 0]), image_size[0]),
        "physically_plausible": bool(sane),
        "unconstrained_principal_point": [float(free_K[0, 2]), float(free_K[1, 2])],
        "camera_matrix": np.asarray(camera_matrix).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs).ravel().tolist(),
        **report,
    }
