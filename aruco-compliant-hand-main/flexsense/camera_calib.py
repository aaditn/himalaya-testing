"""Intrinsic camera calibration from a printed ChArUco board.

Written against the OpenCV >= 4.7 / 5.x API: CharucoDetector +
CharucoBoard.matchImagePoints + cv2.calibrateCamera. The older
cv2.aruco.calibrateCameraCharuco and interpolateCornersCharuco helpers were
removed in OpenCV 5 and are deliberately not used here.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .vision import open_capture, require_cv2

MM_PER_INCH = 25.4

# Deliberately a different family from the finger markers (DICT_4X4_50) so a
# calibration board left in view can never be mistaken for a finger tag.
BOARD_DICTIONARY = "DICT_5X5_100"
BOARD_COLS = 5
BOARD_ROWS = 7
SQUARE_MM = 30.0
MARKER_MM = 22.0


@dataclass(frozen=True)
class BoardSpec:
    cols: int = BOARD_COLS
    rows: int = BOARD_ROWS
    square_mm: float = SQUARE_MM
    marker_mm: float = MARKER_MM
    dictionary: str = BOARD_DICTIONARY

    @property
    def width_mm(self) -> float:
        return self.cols * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.rows * self.square_mm

    def rescaled(self, measured_square_mm: float) -> "BoardSpec":
        """Return this spec with the actually-measured printed square size.

        Printers rescale. The marker size is derived from the measured square so
        the marker-to-square ratio matches what was really printed; only the
        overall scale changed, never the board's proportions.
        """
        if measured_square_mm <= 0:
            raise ValueError("measured square size must be positive")
        ratio = self.marker_mm / self.square_mm
        return BoardSpec(
            cols=self.cols,
            rows=self.rows,
            square_mm=measured_square_mm,
            marker_mm=measured_square_mm * ratio,
            dictionary=self.dictionary,
        )

    @property
    def total_corners(self) -> int:
        """Number of interior chessboard corners the detector can report."""
        return (self.cols - 1) * (self.rows - 1)


def build_board(spec: BoardSpec):
    cv2 = require_cv2()
    if not hasattr(cv2.aruco, spec.dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {spec.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    # Board sizes are (cols, rows). Lengths share units; millimetres here, so
    # every distance downstream is also in millimetres.
    return cv2.aruco.CharucoBoard(
        (spec.cols, spec.rows), spec.square_mm, spec.marker_mm, dictionary
    )


def generate_board_image(spec: BoardSpec, output: str | Path, dpi: int = 300,
                         margin_mm: float = 10.0) -> Path:
    cv2 = require_cv2()
    board = build_board(spec)
    px_per_mm = dpi / MM_PER_INCH
    margin_px = int(round(margin_mm * px_per_mm))
    size_px = (
        int(round(spec.width_mm * px_per_mm)) + 2 * margin_px,
        int(round(spec.height_mm * px_per_mm)) + 2 * margin_px,
    )
    image = board.generateImage(size_px, marginSize=margin_px, borderBits=1)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write board image to {path}")
    return path


def save_intrinsics(path: str | Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                    image_size: tuple[int, int], rms: float, view_count: int,
                    spec: BoardSpec, principal_point_fixed: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "principal_point_fixed": bool(principal_point_fixed),
        "camera_matrix": np.asarray(camera_matrix).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs).ravel().tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms_reprojection_error_px": float(rms),
        "views_used": int(view_count),
        "board": {
            "cols": spec.cols,
            "rows": spec.rows,
            "square_mm": spec.square_mm,
            "marker_mm": spec.marker_mm,
            "dictionary": spec.dictionary,
        },
        "created_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def intrinsics_summary(path: str | Path) -> str | None:
    """One line describing a calibration: where it came from and how good it is.

    Worth printing every run. A calibration belongs to the camera it was made
    with, and silently reusing someone else's is the failure that looks like a
    working system reporting wrong millimetres.
    """
    from .paths import describe, resolve

    found = resolve(path)
    if not found.exists():
        return None
    data = json.loads(found.read_text())
    made = data.get("created_unix")
    when = ("" if made is None else
            " made " + dt.datetime.fromtimestamp(float(made)).strftime("%Y-%m-%d"))
    return (f"{describe(found)}: {data['image_width']}x{data['image_height']}, "
            f"fx {float(data['camera_matrix'][0][0]):.0f}, "
            f"rms {float(data.get('rms_reprojection_error_px', float('nan'))):.2f} px, "
            f"{int(data.get('views_used', 0))} views{when}")


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    from .paths import resolve

    path = resolve(path)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    camera_matrix = np.asarray(data["camera_matrix"], dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=float).reshape(1, -1)
    size = (int(data["image_width"]), int(data["image_height"]))
    return camera_matrix, dist_coeffs, size


def _view_signature(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Compact descriptor of where and how the board sits in the frame.

    Used to reject near-duplicate views; calibration needs pose variety, not
    many photos of the same pose.
    """
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    centroid = points.mean(axis=0)
    spread = points.std(axis=0)
    scale = float(np.sqrt(max(spread[0] * spread[1], 1e-9)))
    return np.array(
        [centroid[0] / width, centroid[1] / height, scale / width,
         float(spread[0] / (spread[1] + 1e-9))]
    )


class RadialReach:
    """Tracks how far from the image centre corners have actually reached.

    Grid coverage can look healthy while every corner still sits in the middle
    of the frame. Radial distortion is only observable where corners are, so
    without corners out near the image corners, k1..k3 are extrapolated and the
    focal length becomes degenerate with them.
    """

    def __init__(self, width: int, height: int):
        self.centre = np.array([width / 2.0, height / 2.0])
        self.max_radius = float(np.hypot(width / 2.0, height / 2.0))
        self.reached = 0.0

    def add(self, corners: np.ndarray) -> None:
        points = np.asarray(corners, dtype=float).reshape(-1, 2)
        radii = np.linalg.norm(points - self.centre, axis=1)
        self.reached = max(self.reached, float(radii.max()) / self.max_radius)

    @property
    def fraction(self) -> float:
        return self.reached


class CoverageGrid:
    """Tracks which parts of the image have ever contained a board corner.

    Distortion coefficients are only constrained where you actually put
    corners, so edge and corner coverage matters more than view count.
    """

    def __init__(self, width: int, height: int, cells: int = 6):
        self.width = width
        self.height = height
        self.cells = cells
        self.grid = np.zeros((cells, cells), dtype=bool)

    def add(self, corners: np.ndarray) -> None:
        points = np.asarray(corners, dtype=float).reshape(-1, 2)
        cols = np.clip((points[:, 0] / self.width * self.cells).astype(int), 0, self.cells - 1)
        rows = np.clip((points[:, 1] / self.height * self.cells).astype(int), 0, self.cells - 1)
        self.grid[rows, cols] = True

    @property
    def fraction(self) -> float:
        return float(self.grid.mean())

    def draw(self, frame) -> None:
        cv2 = require_cv2()
        cell_w = self.width / self.cells
        cell_h = self.height / self.cells
        for row in range(self.cells):
            for col in range(self.cells):
                top_left = (int(col * cell_w) + 1, int(row * cell_h) + 1)
                bottom_right = (int((col + 1) * cell_w) - 1, int((row + 1) * cell_h) - 1)
                if self.grid[row, col]:
                    cv2.rectangle(frame, top_left, bottom_right, (60, 140, 60), 1)
                else:
                    # Dim red marks a region that has never held a corner; the
                    # distortion model is unconstrained there.
                    cv2.rectangle(frame, top_left, bottom_right, (40, 40, 130), 1)


def _sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    """Variance of the Laplacian over the board region only.

    Motion blur is the most common cause of a high reprojection error: blurred
    corners still localise, just wrongly, so they poison the fit silently.
    Measured on the board crop so background clutter cannot mask it.
    """
    cv2 = require_cv2()
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    x0 = max(int(points[:, 0].min()), 0)
    x1 = min(int(points[:, 0].max()) + 1, gray.shape[1])
    y0 = max(int(points[:, 1].min()), 0)
    y1 = min(int(points[:, 1].max()) + 1, gray.shape[0])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    return float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())


def _tilt_degrees(object_points: np.ndarray, image_points: np.ndarray,
                  width: int, height: int) -> float | None:
    """Obliquity of the board, using a provisional pinhole guess.

    Only needs to be good enough to tell a flat-on view from a steeply angled
    one, so a rough fx ~ image width is fine. Fronto-parallel returns ~0.
    """
    cv2 = require_cv2()
    guess = np.array([[float(width), 0.0, width / 2.0],
                      [0.0, float(width), height / 2.0],
                      [0.0, 0.0, 1.0]])
    try:
        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(object_points, dtype=np.float32).reshape(-1, 1, 3),
            np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
            guess, None, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    view = tvec.ravel()
    norm = np.linalg.norm(view)
    if norm < 1e-6:
        return None
    cosine = abs(float(normal @ (view / norm)))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


TILT_BINS = (0.0, 12.0, 25.0, 40.0, 90.0)


def _tilt_bin(tilt: float | None) -> int:
    if tilt is None:
        return 0
    for index in range(len(TILT_BINS) - 1):
        if TILT_BINS[index] <= tilt < TILT_BINS[index + 1]:
            return index
    return len(TILT_BINS) - 2


def per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix,
                    dist_coeffs) -> np.ndarray:
    cv2 = require_cv2()
    errors = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        delta = projected.reshape(-1, 2) - np.asarray(imgp, dtype=float).reshape(-1, 2)
        errors.append(float(np.sqrt((delta ** 2).sum(axis=1).mean())))
    return np.asarray(errors)


def sanity_report(camera_matrix: np.ndarray, image_size: tuple[int, int],
                  principal_point_fixed: bool = False) -> dict[str, Any]:
    """Flag physically implausible results a low RMS alone will not catch."""
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    width, height = image_size
    aspect = abs(fx - fy) / max(fx, fy)
    offset_x = abs(cx - width / 2.0) / (width / 2.0)
    offset_y = abs(cy - height / 2.0) / (height / 2.0)
    warnings: list[str] = []
    if aspect > 0.02:
        warnings.append(
            f"fx and fy differ by {aspect * 100:.1f}% (expect <2% for square pixels); "
            "usually means too little tilt variety or blurred views"
        )
    if not principal_point_fixed and max(offset_x, offset_y) > 0.10:
        warnings.append(
            f"principal point is {offset_x * 100:.0f}%/{offset_y * 100:.0f}% off centre "
            "in x/y (expect <10%); the fit is poorly conditioned"
        )
    return {
        "fx_fy_mismatch_pct": aspect * 100.0,
        "principal_point_offset_pct": [offset_x * 100.0, offset_y * 100.0],
        "warnings": warnings,
    }


SEED_FIELDS_OF_VIEW = (40.0, 55.0, 70.0, 85.0, 100.0, 120.0)


def horizontal_fov_deg(fx: float, width: int) -> float:
    return float(2.0 * np.degrees(np.arctan(width / 2.0 / fx)))


def plausible_intrinsics(camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                         image_size: tuple[int, int]) -> tuple[bool, str]:
    """Reject solutions no real webcam could produce.

    The distortion polynomial can run away and still reproject well, giving a
    low error on a physically impossible lens, so reprojection error alone is
    not a sufficient acceptance test.
    """
    width, _ = image_size
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    if fx <= 0 or fy <= 0:
        return False, "non-positive focal length"
    fov = horizontal_fov_deg(fx, width)
    if not 25.0 <= fov <= 150.0:
        return False, f"implied field of view {fov:.0f} deg is not a real webcam"
    if abs(fx - fy) / max(fx, fy) > 0.05:
        return False, f"fx/fy differ by {abs(fx - fy) / max(fx, fy) * 100:.0f}%"
    d = np.abs(np.asarray(dist_coeffs).ravel())
    if d[0] > 2.0 or (len(d) > 1 and d[1] > 10.0) or (len(d) > 4 and d[4] > 50.0):
        return False, "distortion coefficients have run away"
    return True, "ok"


def robust_calibrate(object_points, image_points, image_size: tuple[int, int],
                     fix_principal_point: bool = True):
    """Calibrate from several explicit starting guesses and keep the best sane one.

    OpenCV's own initial estimate can land in a basin where the focal length is
    several times too long and the distortion terms absorb the difference. Every
    explicit seed converges to the correct optimum, so seed deliberately across
    the plausible range and choose by reprojection error among physical results.
    """
    cv2 = require_cv2()
    width, height = image_size
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    best = None
    fallback = None
    for fov in SEED_FIELDS_OF_VIEW:
        seed_fx = (width / 2.0) / np.tan(np.radians(fov / 2.0))
        guess = np.array([[seed_fx, 0.0, width / 2.0],
                          [0.0, seed_fx, height / 2.0],
                          [0.0, 0.0, 1.0]])
        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size, guess, None, flags=flags
            )
        except cv2.error:
            continue
        candidate = (rms, K, dist, rvecs, tvecs)
        if fallback is None or rms < fallback[0]:
            fallback = candidate
        ok, _why = plausible_intrinsics(K, dist, image_size)
        if ok and (best is None or rms < best[0]):
            best = candidate

    if best is not None:
        return best + (True,)
    if fallback is not None:
        return fallback + (False,)
    raise RuntimeError("calibration failed from every starting guess")


def run_camera_calibration(
    source: int | str,
    width: int,
    height: int,
    output: str | Path,
    spec: BoardSpec | None = None,
    measured_square_mm: float | None = None,
    fix_principal_point: bool = True,
    target_views: int = 20,
    min_corners: int = 12,
    diversity: float = 0.12,
    blur_ratio: float = 0.35,
    min_coverage: float = 0.55,
    max_view_error_px: float = 1.5,
    frames_dir: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    cv2 = require_cv2()
    spec = spec or BoardSpec()
    if measured_square_mm is not None:
        spec = spec.rescaled(measured_square_mm)
    board = build_board(spec)
    detector = cv2.aruco.CharucoDetector(board)

    live = isinstance(source, int)
    capture = open_capture(source, width, height)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    tilts: list[float] = []
    kept_frames: list[np.ndarray] = []
    recent_sharpness: deque[float] = deque(maxlen=120)
    coverage: CoverageGrid | None = None
    image_size: tuple[int, int] | None = None
    last_capture_time = 0.0

    try:
        max_views = max(target_views * 3, target_views + 20)

        def still_needed() -> bool:
            """Enough views, enough of the frame, and enough oblique views."""
            if len(object_points) >= max_views:
                return False
            if len(object_points) < target_views:
                return True
            if coverage is not None and coverage.fraction < min_coverage:
                return True
            return sum(1 for t in tilts if t >= 25.0) < 4

        while still_needed():
            ok, frame = capture.read()
            if not ok:
                break
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
                coverage = CoverageGrid(*image_size)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)

            detected = 0 if charuco_ids is None else len(charuco_ids)
            accepted = False
            reason = ""
            tilt = None
            sharp = 0.0

            if detected >= min_corners:
                sharp = _sharpness(gray, charuco_corners)
                recent_sharpness.append(sharp)
                # Adaptive: absolute Laplacian variance depends on lighting and
                # texture, so judge each frame against the sharpest recently seen.
                reference = float(np.percentile(recent_sharpness, 90)) if recent_sharpness else 0.0
                sharp_enough = sharp >= blur_ratio * reference

                obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                tilt = _tilt_degrees(obj_pts, img_pts, *image_size)
                signature = _view_signature(charuco_corners, *image_size)
                # Tilt joins the signature so a steeply angled view counts as new
                # even from a position already sampled. Flat-on views alone leave
                # focal length and distance nearly degenerate.
                signature = np.append(signature, (tilt or 0.0) / 45.0)
                novel = all(
                    float(np.linalg.norm(signature - previous)) > diversity
                    for previous in signatures
                )
                fresh = (not live) or (time.monotonic() - last_capture_time) > 0.4

                if not sharp_enough:
                    reason = "too blurry - hold still"
                elif not novel:
                    reason = "move or tilt the board - too similar to a captured view"
                elif fresh and obj_pts is not None and len(obj_pts) >= min_corners:
                    object_points.append(obj_pts)
                    image_points.append(img_pts)
                    signatures.append(signature)
                    tilts.append(tilt if tilt is not None else 0.0)
                    coverage.add(charuco_corners)
                    if frames_dir is not None:
                        kept_frames.append(frame.copy())
                    last_capture_time = time.monotonic()
                    accepted = True
            else:
                reason = f"need {min_corners}+ corners, see {detected}"

            if display:
                if charuco_ids is not None and detected:
                    cv2.aruco.drawDetectedCornersCharuco(
                        frame, charuco_corners.reshape(-1, 1, 2), charuco_ids
                    )
                coverage.draw(frame)
                bins = np.bincount([_tilt_bin(t) for t in tilts], minlength=len(TILT_BINS) - 1)
                cv2.putText(
                    frame,
                    f"views {len(object_points)}/{target_views}   "
                    f"corners {detected}/{spec.total_corners}   "
                    f"coverage {coverage.fraction * 100:.0f}%",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                    (40, 220, 40) if accepted else (240, 210, 30), 2,
                )
                tilt_text = "  ".join(
                    f"{int(TILT_BINS[i])}-{int(TILT_BINS[i + 1])}deg:{bins[i]}"
                    for i in range(len(bins))
                )
                need_tilt = bins[2] + bins[3] < 4
                need_cover = coverage.fraction < min_coverage
                cv2.putText(frame, f"tilt {tilt_text}", (18, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (40, 120, 240) if need_tilt else (60, 220, 60), 2)
                if tilt is not None:
                    cv2.putText(frame, f"this view: tilt {tilt:.0f}deg  sharp {sharp:.0f}",
                                (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                outstanding = []
                if len(object_points) < target_views:
                    outstanding.append(f"{target_views - len(object_points)} more views")
                if need_cover:
                    outstanding.append(
                        f"coverage {coverage.fraction * 100:.0f}/{min_coverage * 100:.0f}%"
                    )
                if need_tilt:
                    outstanding.append("more tilt >25deg")
                if outstanding and len(object_points) > 4:
                    cv2.putText(frame, "NEED: " + ",  ".join(outstanding),
                                (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 120, 240), 2)
                if need_cover and len(object_points) > 4:
                    cv2.putText(frame, "fill the RED cells - move the board to the frame edges",
                                (18, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 120, 240), 2)
                if reason:
                    cv2.putText(frame, reason, (18, image_size[1] - 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 80, 240), 2)
                if accepted:
                    cv2.rectangle(frame, (0, 0), (image_size[0] - 1, image_size[1] - 1),
                                  (40, 220, 40), 8)
                cv2.putText(frame, "q: finish early   r: reset",
                            (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1)
                cv2.imshow("FlexSense camera calibration", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    object_points.clear(); image_points.clear(); signatures.clear()
                    tilts.clear(); kept_frames.clear()
                    coverage = CoverageGrid(*image_size)
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()

    if len(object_points) < 6:
        raise RuntimeError(
            f"Only {len(object_points)} usable views captured; need at least 6. "
            "Improve lighting, hold each pose still for about a second, and move "
            "the board closer so its squares are larger in frame. Partial views "
            "are fine - ChArUco only needs enough identified corners."
        )

    def fit(objs, imgs):
        rms_, K_, d_, rv_, tv_, sane_ = robust_calibrate(
            objs, imgs, image_size, fix_principal_point
        )
        free = robust_calibrate(objs, imgs, image_size, fix_principal_point=False)
        return rms_, K_, d_, rv_, tv_, free[1], sane_

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, unconstrained_K, sane = fit(
        object_points, image_points
    )
    errors = per_view_errors(object_points, image_points, rvecs, tvecs,
                             camera_matrix, dist_coeffs)
    # A few bad views (blur, a corner mis-snapped) drag the whole fit. Drop them
    # and refit, provided enough views survive to stay well constrained.
    threshold = max(max_view_error_px, 2.5 * float(np.median(errors)))
    keep = errors <= threshold
    dropped = int((~keep).sum())
    if dropped and int(keep.sum()) >= 8:
        object_points = [o for o, k in zip(object_points, keep) if k]
        image_points = [i for i, k in zip(image_points, keep) if k]
        kept_frames = [f for f, k in zip(kept_frames, keep) if k]
        tilts = [t for t, k in zip(tilts, keep) if k]
        rms, camera_matrix, dist_coeffs, rvecs, tvecs, unconstrained_K, sane = fit(
            object_points, image_points
        )
        errors = per_view_errors(object_points, image_points, rvecs, tvecs,
                                 camera_matrix, dist_coeffs)
    else:
        dropped = 0

    if frames_dir is not None and kept_frames:
        directory = Path(frames_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for index, saved in enumerate(kept_frames):
            cv2.imwrite(str(directory / f"view_{index:03d}.png"), saved)

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
            f"corners covered only {coverage.fraction * 100:.0f}% of the frame; the "
            "principal point and distortion are extrapolated outside that region"
        )
    if tilt_bins[2] + tilt_bins[3] < 4:
        report["warnings"].append(
            f"only {int(tilt_bins[2] + tilt_bins[3])} views tilted beyond 25 degrees; "
            "flat-on views leave focal length and distance nearly degenerate"
        )
    return {
        "saved": str(saved_path),
        "rms_reprojection_error_px": float(rms),
        "views_used": len(object_points),
        "principal_point_fixed": bool(fix_principal_point),
        "horizontal_fov_deg": horizontal_fov_deg(float(camera_matrix[0, 0]), image_size[0]),
        "physically_plausible": bool(sane),
        "unconstrained_principal_point": [float(unconstrained_K[0, 2]),
                                          float(unconstrained_K[1, 2])],
        "views_dropped_as_outliers": dropped,
        "worst_view_error_px": float(errors.max()),
        "coverage_fraction": coverage.fraction if coverage else 0.0,
        "tilt_histogram": {
            f"{int(TILT_BINS[i])}-{int(TILT_BINS[i + 1])}deg": int(tilt_bins[i])
            for i in range(len(tilt_bins))
        },
        "camera_matrix": np.asarray(camera_matrix).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs).ravel().tolist(),
        **report,
    }


def generate_board_svg(spec: BoardSpec, output: str | Path, page_width_mm: float = 210.0,
                       page_height_mm: float = 297.0, dpi: int = 600) -> Path:
    """Emit the board as SVG sized in millimetres.

    PNG carries no reliable physical size (OpenCV writes no pHYs chunk), so a
    print dialog has to guess the DPI and will scale the board. SVG states its
    size in mm, which every print dialog honours at 100% scale. This mirrors
    the approach already used for the finger marker sheet.
    """
    import base64

    cv2 = require_cv2()
    board = build_board(spec)
    px_per_mm = dpi / MM_PER_INCH
    # No margin here: the embedded raster maps exactly onto width_mm x height_mm
    # so a printed square measures exactly square_mm. The page supplies margin.
    size_px = (int(round(spec.width_mm * px_per_mm)), int(round(spec.height_mm * px_per_mm)))
    raster = board.generateImage(size_px, marginSize=0, borderBits=1)
    ok, encoded = cv2.imencode(".png", raster)
    if not ok:
        raise RuntimeError("Could not encode board raster")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")

    board_x = (page_width_mm - spec.width_mm) / 2.0
    board_y = 16.0
    ruler_y = board_y + spec.height_mm + 18.0
    ruler_x = (page_width_mm - 100.0) / 2.0

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{page_width_mm}mm" '
            f'height="{page_height_mm}mm" viewBox="0 0 {page_width_mm} {page_height_mm}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{page_width_mm / 2}" y="9" text-anchor="middle" '
            f'font-family="sans-serif" font-size="4">FlexSense ChArUco '
            f'{spec.cols}x{spec.rows}, {spec.dictionary}, print at 100% actual size</text>'
        ),
        (
            f'<image x="{board_x}" y="{board_y}" width="{spec.width_mm}" '
            f'height="{spec.height_mm}" style="image-rendering:pixelated" '
            f'xlink:href="data:image/png;base64,{data}"/>'
        ),
        # Physical check bar: if this is not exactly 100 mm the printer rescaled.
        (
            f'<line x1="{ruler_x}" y1="{ruler_y}" x2="{ruler_x + 100.0}" y2="{ruler_y}" '
            'stroke="black" stroke-width="0.4"/>'
        ),
        (
            f'<line x1="{ruler_x}" y1="{ruler_y - 2.5}" x2="{ruler_x}" y2="{ruler_y + 2.5}" '
            'stroke="black" stroke-width="0.4"/>'
        ),
        (
            f'<line x1="{ruler_x + 100.0}" y1="{ruler_y - 2.5}" x2="{ruler_x + 100.0}" '
            f'y2="{ruler_y + 2.5}" stroke="black" stroke-width="0.4"/>'
        ),
        (
            f'<text x="{page_width_mm / 2}" y="{ruler_y + 8}" text-anchor="middle" '
            'font-family="sans-serif" font-size="3.4">'
            'This bar must measure exactly 100 mm. Each black square must be '
            f'{spec.square_mm:.0f} mm.</text>'
        ),
        "</svg>",
    ]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


def generate_board(spec: BoardSpec, output: str | Path, dpi: int = 300) -> Path:
    """Write the board, choosing SVG or raster from the output suffix."""
    if Path(output).suffix.lower() == ".svg":
        return generate_board_svg(spec, output)
    return generate_board_image(spec, output, dpi=dpi)

def refit_from_frames(frames_dir: str | Path, output: str | Path,
                      spec: BoardSpec | None = None,
                      measured_square_mm: float | None = None,
                      fix_principal_point: bool = True,
                      max_view_error_px: float = 1.5) -> dict[str, Any]:
    """Recalibrate from frames saved by an earlier capture.

    Capturing is the slow, physical part; refitting is free. Keeping the frames
    means a fitting problem can be corrected without asking the operator to wave
    a board around again.
    """
    cv2 = require_cv2()
    spec = spec or BoardSpec()
    if measured_square_mm is not None:
        spec = spec.rescaled(measured_square_mm)
    board = build_board(spec)
    detector = cv2.aruco.CharucoDetector(board)

    paths = sorted(Path(frames_dir).glob("*.png"))
    if not paths:
        raise RuntimeError(f"No .png frames found in {frames_dir}")

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    tilts: list[float] = []
    coverage: CoverageGrid | None = None
    reach: RadialReach | None = None
    image_size: tuple[int, int] | None = None
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
            coverage = CoverageGrid(*image_size)
            reach = RadialReach(*image_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _mc, _mi = detector.detectBoard(gray)
        if ids is None or len(ids) < 12:
            continue
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        object_points.append(obj_pts)
        image_points.append(img_pts)
        tilts.append(_tilt_degrees(obj_pts, img_pts, *image_size) or 0.0)
        coverage.add(corners)
        reach.add(corners)

    if len(object_points) < 6:
        raise RuntimeError(f"Only {len(object_points)} usable frames in {frames_dir}")

    rms, K, dist, rvecs, tvecs, sane = robust_calibrate(
        object_points, image_points, image_size, fix_principal_point
    )
    errors = per_view_errors(object_points, image_points, rvecs, tvecs, K, dist)
    threshold = max(max_view_error_px, 2.5 * float(np.median(errors)))
    keep = errors <= threshold
    dropped = int((~keep).sum())
    if dropped and int(keep.sum()) >= 8:
        object_points = [o for o, k in zip(object_points, keep) if k]
        image_points = [i for i, k in zip(image_points, keep) if k]
        tilts = [t for t, k in zip(tilts, keep) if k]
        rms, K, dist, rvecs, tvecs, sane = robust_calibrate(
            object_points, image_points, image_size, fix_principal_point
        )
        errors = per_view_errors(object_points, image_points, rvecs, tvecs, K, dist)
    else:
        dropped = 0

    saved = save_intrinsics(output, K, dist, image_size, rms, len(object_points),
                            spec, fix_principal_point)
    report = sanity_report(K, image_size, fix_principal_point)
    if not sane:
        _ok, why = plausible_intrinsics(K, dist, image_size)
        report["warnings"].append(f"no physically plausible solution found ({why})")
    tilt_bins = np.bincount([_tilt_bin(t) for t in tilts], minlength=len(TILT_BINS) - 1)
    return {
        "saved": str(saved),
        "frames_read": len(paths),
        "views_used": len(object_points),
        "views_dropped_as_outliers": dropped,
        "rms_reprojection_error_px": float(rms),
        "worst_view_error_px": float(errors.max()),
        "coverage_fraction": coverage.fraction if coverage else 0.0,
        "radial_reach_fraction": reach.fraction if reach else 0.0,
        "tilt_histogram": {
            f"{int(TILT_BINS[i])}-{int(TILT_BINS[i + 1])}deg": int(tilt_bins[i])
            for i in range(len(tilt_bins))
        },
        "principal_point_fixed": bool(fix_principal_point),
        "horizontal_fov_deg": horizontal_fov_deg(float(K[0, 0]), image_size[0]),
        "physically_plausible": bool(sane),
        "camera_matrix": np.asarray(K).tolist(),
        "dist_coeffs": np.asarray(dist).ravel().tolist(),
        **report,
    }

