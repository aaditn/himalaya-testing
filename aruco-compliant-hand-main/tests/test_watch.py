"""Run the viewer loop over many frames with both marker families present.

A single-frame smoke test misses state that only breaks on the second pass,
which is how a rebound local (`sizes`) shipped: frame one populated it, frame
two indexed it as the wrong type.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from flexsense.camera_calib import (
    BOARD_DICTIONARY,
    BoardSpec,
    build_board,
    save_intrinsics,
)
from flexsense.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "so101.yaml"
FRAME = (1280, 720)
K = np.array([[729.9, 0.0, 640.0], [0.0, 725.7, 360.0], [0.0, 0.0, 1.0]])


def mixed_scene(config) -> np.ndarray:
    """A frame holding both the ChArUco board and several finger markers."""
    frame = np.full((FRAME[1], FRAME[0], 3), 242, np.uint8)
    board_img = build_board(BoardSpec()).generateImage((430, 602), marginSize=14,
                                                       borderBits=1)
    frame[40:642, 40:470] = board_img[:, :, None]
    finger = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.markers.dictionary))
    for marker_id, (x, y, size) in {0: (620, 90, 150), 1: (860, 330, 110),
                                    2: (620, 430, 120)}.items():
        img = cv2.aruco.generateImageMarker(finger, marker_id, size)
        frame[y:y + size, x:x + size] = img[:, :, None]
    return frame


def write_clip(path: Path, frame: np.ndarray, frames: int = 12) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, FRAME)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


class TestWatchLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)
        self.tmp = Path(tempfile.mkdtemp())
        self.clip = self.tmp / "scene.mp4"
        write_clip(self.clip, mixed_scene(self.config))
        self._imshow = cv2.imshow
        self._waitkey = cv2.waitKey
        self._destroy = cv2.destroyAllWindows
        self.shown = []
        cv2.imshow = lambda name, img: self.shown.append(img)
        cv2.waitKey = lambda *a, **k: 255          # never quit early
        cv2.destroyAllWindows = lambda *a, **k: None

    def tearDown(self) -> None:
        cv2.imshow = self._imshow
        cv2.waitKey = self._waitkey
        cv2.destroyAllWindows = self._destroy

    def test_runs_many_frames_without_calibration(self) -> None:
        from flexsense.watch import run_watch
        code = run_watch(self.config, str(self.clip),
                         intrinsics_path=self.tmp / "missing.json")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(self.shown), 10)

    def test_runs_many_frames_with_pose_enabled(self) -> None:
        from flexsense.watch import run_watch
        path = self.tmp / "intr.json"
        save_intrinsics(path, K, np.zeros(5), FRAME, 0.3, 20, BoardSpec(), True)
        code = run_watch(self.config, str(self.clip), intrinsics_path=path,
                         marker_mm=14.0, board_square_mm=28.0)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(self.shown), 10)

    def test_detects_both_families(self) -> None:
        from flexsense.vision import MarkerDetector
        frame = mixed_scene(self.config)
        finger = MarkerDetector(self.config.markers.dictionary).detect(frame)[0]
        board = MarkerDetector(BOARD_DICTIONARY).detect(frame)[0]
        self.assertGreaterEqual(len(finger), 3)
        self.assertGreaterEqual(len(board), 5)

    def test_explicit_single_dictionary(self) -> None:
        from flexsense.watch import run_watch
        code = run_watch(self.config, str(self.clip),
                         intrinsics_path=self.tmp / "missing.json",
                         dictionaries=[BOARD_DICTIONARY])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()


class TestLensCorrection(unittest.TestCase):
    """Distortion biases every millimetre the tracker reports; correction must remove it."""

    K = np.array([[729.9, 0.0, 640.0], [0.0, 725.7, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.265, 0.1608, -0.0024, -0.0008, -0.0752])
    SIZE = 14.0

    def _intrinsics(self, tmp: Path, size=(1280, 720)) -> Path:
        path = tmp / "intr.json"
        save_intrinsics(path, self.K, self.D, size, 0.431, 64, BoardSpec(), True)
        return path

    def _deflection(self, detector, offset, applied):
        from flexsense.geometry import reference_homography, marker_center_and_angle
        half = self.SIZE / 2

        def quad(cx):
            return np.array([[cx - half, half, 0], [cx + half, half, 0],
                             [cx + half, -half, 0], [cx - half, -half, 0]], float)

        tvec = np.array([[float(offset[0])], [float(offset[1])], [150.0]])

        def measure(tip):
            ref, _ = cv2.projectPoints(np.asarray(quad(0), np.float32),
                                       np.zeros((3, 1)), tvec, self.K, self.D)
            tipp, _ = cv2.projectPoints(np.asarray(quad(30 + tip), np.float32),
                                        np.zeros((3, 1)), tvec, self.K, self.D)
            ref = ref.reshape(-1, 2)
            tipp = tipp.reshape(-1, 2)
            if detector.undistorting:
                ref = detector._undistort(ref)
                tipp = detector._undistort(tipp)
            return marker_center_and_angle(tipp, reference_homography(ref, self.SIZE))[0][0]

        return measure(applied) - measure(0.0)

    def test_correction_removes_the_bias(self) -> None:
        from flexsense.vision import build_detector
        with tempfile.TemporaryDirectory() as tmp:
            path = self._intrinsics(Path(tmp))
            plain, _ = build_detector("DICT_4X4_50", None)
            fixed, _ = build_detector("DICT_4X4_50", path, (1280, 720))
            for offset in [(0, 0), (40, 25), (80, 45)]:
                raw = self._deflection(plain, offset, 5.0)
                corrected = self._deflection(fixed, offset, 5.0)
                self.assertGreater(abs(raw - 5.0), 0.15, "expected a real uncorrected bias")
                self.assertLess(abs(corrected - 5.0), 0.02, f"correction failed at {offset}")

    def test_resolution_mismatch_disables_correction(self) -> None:
        from flexsense.vision import build_detector
        with tempfile.TemporaryDirectory() as tmp:
            path = self._intrinsics(Path(tmp))
            detector, note = build_detector("DICT_4X4_50", path, (640, 480))
            self.assertFalse(detector.undistorting)
            self.assertIn("640x480", note)

    def test_missing_calibration_warns_and_continues(self) -> None:
        from flexsense.vision import build_detector
        with tempfile.TemporaryDirectory() as tmp:
            detector, note = build_detector("DICT_4X4_50", Path(tmp) / "none.json")
            self.assertFalse(detector.undistorting)
            self.assertIn("biased", note)

    def test_raw_corners_stay_uncorrected_for_overlays(self) -> None:
        from flexsense.vision import build_detector
        with tempfile.TemporaryDirectory() as tmp:
            path = self._intrinsics(Path(tmp))
            detector, _ = build_detector("DICT_4X4_50", path, (1280, 720))
            config = load_config(CONFIG)
            by_id, corners, ids = detector.detect(mixed_scene(config))
            self.assertTrue(len(by_id) >= 3)
            raw = np.asarray(corners[0]).reshape(4, 2)
            metric = by_id[int(ids.flatten()[0])]
            self.assertFalse(np.allclose(raw, metric),
                             "by_id must be corrected while corners stay raw")
