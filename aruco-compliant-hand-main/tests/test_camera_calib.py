"""Exercise the camera-calibration path end to end.

The CLI-argument test catches parser drift, but only running a real calibration
catches a broken helper signature, which is how a NameError inside
save_intrinsics reached the user.
"""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from flexsense.camera_calib import (
    BoardSpec,
    build_board,
    generate_board_svg,
    load_intrinsics,
    run_camera_calibration,
    sanity_report,
    save_intrinsics,
)

TRUE_K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.30, 0.12, 0.001, -0.0008, -0.02])
FRAME = (1280, 720)


def synthetic_video(path: Path, spec: BoardSpec) -> None:
    """Render a board sweeping position, depth and tilt in front of a known camera."""
    board = build_board(spec)
    px_per_mm = 600 / 25.4
    raster = board.generateImage(
        (int(spec.width_mm * px_per_mm), int(spec.height_mm * px_per_mm)),
        marginSize=0, borderBits=1,
    )
    raster = cv2.resize(raster, (raster.shape[1] // 4, raster.shape[0] // 4),
                        interpolation=cv2.INTER_AREA)
    plane = np.array([[0, 0, 0], [spec.width_mm, 0, 0],
                      [spec.width_mm, spec.height_mm, 0], [0, spec.height_mm, 0]], float)
    src = np.array([[0, 0], [raster.shape[1], 0],
                    [raster.shape[1], raster.shape[0]], [0, raster.shape[0]]], np.float32)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, FRAME)
    index = 0
    for (gx, gy), depth, tilt in itertools.product(
        [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
        [290.0, 420.0], [0.0, 0.8],
    ):
        rvec = np.array([[tilt * np.cos(index), tilt * np.sin(index),
                          0.35 * np.sin(index * 0.7)]]).reshape(3, 1)
        tvec = np.array([[gx * 84.0 - spec.width_mm / 2],
                         [gy * 52.0 - spec.height_mm / 2], [depth]])
        index += 1
        projected, _ = cv2.projectPoints(plane, rvec, tvec, TRUE_K, TRUE_DIST)
        warp = cv2.getPerspectiveTransform(src, projected.reshape(4, 2).astype(np.float32))
        frame = cv2.warpPerspective(raster, warp, FRAME, flags=cv2.INTER_LINEAR,
                                    borderValue=255)
        for _ in range(3):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


class TestIntrinsicsIO(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intr.json"
            save_intrinsics(path, TRUE_K, TRUE_DIST, FRAME, 0.3, 12,
                            BoardSpec(), principal_point_fixed=True)
            payload = json.loads(path.read_text())
            self.assertTrue(payload["principal_point_fixed"])
            camera_matrix, dist, size = load_intrinsics(path)
            self.assertTrue(np.allclose(camera_matrix, TRUE_K))
            self.assertTrue(np.allclose(dist.ravel(), TRUE_DIST))
            self.assertEqual(size, FRAME)

    def test_load_missing_file_returns_none(self) -> None:
        self.assertIsNone(load_intrinsics("/nonexistent/intr.json"))


class TestBoardSpec(unittest.TestCase):
    def test_rescale_preserves_marker_ratio(self) -> None:
        nominal = BoardSpec()
        rescaled = nominal.rescaled(28.0)
        self.assertAlmostEqual(rescaled.square_mm, 28.0)
        self.assertAlmostEqual(rescaled.marker_mm / rescaled.square_mm,
                               nominal.marker_mm / nominal.square_mm)

    def test_rescale_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            BoardSpec().rescaled(0.0)

    def test_svg_states_physical_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_board_svg(BoardSpec(), Path(tmp) / "b.svg")
            text = path.read_text()
            self.assertIn('width="210.0mm"', text)
            self.assertIn('height="297.0mm"', text)


class TestSanityReport(unittest.TestCase):
    def test_offset_principal_point_warns_when_free(self) -> None:
        skewed = TRUE_K.copy()
        skewed[1, 2] = 480.0
        self.assertTrue(sanity_report(skewed, FRAME, principal_point_fixed=False)["warnings"])

    def test_no_principal_point_warning_when_fixed(self) -> None:
        skewed = TRUE_K.copy()
        skewed[1, 2] = 480.0
        report = sanity_report(skewed, FRAME, principal_point_fixed=True)
        self.assertFalse([w for w in report["warnings"] if "principal point" in w])


class TestPlausibility(unittest.TestCase):
    """The runaway solution a real capture produced must be rejected."""

    def test_runaway_focal_length_is_rejected(self) -> None:
        from flexsense.camera_calib import plausible_intrinsics
        bad = np.array([[3837.7, 0.0, 640.0], [0.0, 5695.2, 360.0], [0.0, 0.0, 1.0]])
        bad_dist = np.array([-11.69, 290.18, 0.045, -0.081, -4282.27])
        ok, why = plausible_intrinsics(bad, bad_dist, FRAME)
        self.assertFalse(ok)
        self.assertTrue(why)

    def test_realistic_webcam_is_accepted(self) -> None:
        from flexsense.camera_calib import plausible_intrinsics
        good = np.array([[729.9, 0.0, 640.0], [0.0, 725.7, 360.0], [0.0, 0.0, 1.0]])
        good_dist = np.array([-0.265, 0.161, -0.0024, -0.0008, -0.075])
        ok, why = plausible_intrinsics(good, good_dist, FRAME)
        self.assertTrue(ok, why)

    def test_fov_matches_focal_length(self) -> None:
        from flexsense.camera_calib import horizontal_fov_deg
        self.assertAlmostEqual(horizontal_fov_deg(729.9, 1280), 82.5, places=0)

    def test_multi_start_agrees_across_seeds(self) -> None:
        """Every seed must reach the same optimum, or the fit is not robust."""
        from flexsense.camera_calib import robust_calibrate
        spec = BoardSpec().rescaled(28.0)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "board.mp4"
            synthetic_video(video, spec)
            board = build_board(spec)
            detector = cv2.aruco.CharucoDetector(board)
            capture = cv2.VideoCapture(str(video))
            objs, imgs = [], []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                corners, ids, _a, _b = detector.detectBoard(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if ids is None or len(ids) < 12:
                    continue
                o, p = board.matchImagePoints(corners, ids)
                objs.append(o)
                imgs.append(p)
            capture.release()
            rms, K, dist, _r, _t, sane = robust_calibrate(objs, imgs, FRAME, True)
            self.assertTrue(sane)
            self.assertAlmostEqual(K[0, 2], FRAME[0] / 2, places=3)
            self.assertLess(abs(K[0, 0] - K[1, 1]) / max(K[0, 0], K[1, 1]), 0.05)


class TestFullCalibration(unittest.TestCase):
    """Runs the real function, so helper signatures cannot silently break."""

    def test_calibration_runs_and_writes_file(self) -> None:
        spec = BoardSpec().rescaled(28.0)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "board.mp4"
            synthetic_video(video, spec)
            out = Path(tmp) / "intr.json"
            result = run_camera_calibration(
                source=str(video), width=FRAME[0], height=FRAME[1], output=out,
                measured_square_mm=28.0, target_views=12, display=False, frames_dir=None,
            )
            self.assertTrue(out.exists())
            self.assertGreaterEqual(result["views_used"], 6)
            self.assertIn("principal_point_fixed", result)
            self.assertIn("unconstrained_principal_point", result)
            self.assertTrue(result["principal_point_fixed"])
            # principal point must actually be pinned to the image centre
            matrix = np.asarray(result["camera_matrix"])
            self.assertAlmostEqual(matrix[0, 2], FRAME[0] / 2, places=3)
            self.assertAlmostEqual(matrix[1, 2], FRAME[1] / 2, places=3)
            self.assertLess(result["rms_reprojection_error_px"], 3.0)

    def test_free_principal_point_is_not_pinned(self) -> None:
        spec = BoardSpec().rescaled(28.0)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "board.mp4"
            synthetic_video(video, spec)
            result = run_camera_calibration(
                source=str(video), width=FRAME[0], height=FRAME[1],
                output=Path(tmp) / "intr.json", measured_square_mm=28.0,
                target_views=12, display=False, frames_dir=None,
                fix_principal_point=False,
            )
            self.assertFalse(result["principal_point_fixed"])


if __name__ == "__main__":
    unittest.main()
