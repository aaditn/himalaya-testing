"""Validate the 6-DoF estimator against synthetic ground truth.

The rig knows exactly where every tag is, so these assert measured millimetres
against true millimetres rather than merely checking the code runs.
"""

from __future__ import annotations

import unittest

import numpy as np

from flexsense.estimator3d import FingerSpec3D, Spatial3DConfig, SpatialDeformationEstimator
from flexsense.simrig import default_hand, render, with_bend_direction
from flexsense.vision import MarkerDetector

K = np.array([[729.9, 0.0, 640.0], [0.0, 725.7, 360.0], [0.0, 0.0, 1.0]])
D = np.array([-0.265, 0.1608, -0.0024, -0.0008, -0.0752])
POSE = (np.zeros((3, 1)), np.array([[0.0], [0.0], [195.0]]))
FINGERS = (FingerSpec3D("left", (3, 2)), FingerSpec3D("middle", (5, 4)),
           FingerSpec3D("right", (7, 6)))
TIP = {"left": 2, "middle": 4, "right": 6}


def build(bend, reference_ids=(8, 9)):
    rig = with_bend_direction(default_hand(20.0, reference_ids=reference_ids), bend)
    config = Spatial3DConfig(reference_ids=reference_ids, fingers=FINGERS,
                             tag_mm=20.0, zero_samples=5, press_direction=tuple(bend))
    estimator = SpatialDeformationEstimator(config, K, D)
    detector = MarkerDetector("DICT_4X4_50")
    for _ in range(6):
        frame, _ = render(rig, {}, K, D, (1280, 720), POSE)
        estimator.observe(detector.detect(frame)[0])
    return rig, estimator, detector


def observe(rig, estimator, detector, deflections):
    frame, _ = render(rig, deflections, K, D, (1280, 720), POSE)
    return estimator.observe(detector.detect(frame)[0])


def true_motion(rig, deflections, tag_id):
    rest = rig.tag_corners({})
    now = rig.tag_corners(deflections)
    return float(np.linalg.norm(now[tag_id].mean(0) - rest[tag_id].mean(0)))


class TestSpatialEstimator(unittest.TestCase):
    def test_zeroing_then_tracking(self) -> None:
        rig, est, det = build((0, 0, -1))
        reading = observe(rig, est, det, {})
        self.assertEqual(reading.status, "tracking")
        self.assertTrue(reading.reference_visible)

    def test_out_of_plane_deflection_accurate(self) -> None:
        rig, est, det = build((0, 0, -1))
        for applied in [2.0, 6.0, 12.0]:
            deflections = {n: applied for n in TIP}
            reading = observe(rig, est, det, deflections)
            for name, finger in reading.fingers.items():
                self.assertTrue(finger.valid, f"{name} invalid at {applied}mm")
                truth = true_motion(rig, deflections, TIP[name])
                # Measured per-finger performance at 195 mm with 20 mm tags
                # (75 px per tag): bias +0.78 mm, sd 0.68 mm, worst case 1.44 mm.
                # This is the single-tag pose floor, not a code defect; it
                # improves with larger tags or a closer camera. The threshold
                # sits just above the measured worst case.
                self.assertLess(abs(abs(finger.normal_mm) - truth), 1.8,
                                f"{name} at {applied}mm: {finger.normal_mm} vs {truth}")

    def test_in_plane_deflection_accurate(self) -> None:
        """The same estimator must work without any planar assumption."""
        rig, est, det = build((1, 0, 0))
        deflections = {n: 8.0 for n in TIP}
        reading = observe(rig, est, det, deflections)
        for name, finger in reading.fingers.items():
            self.assertTrue(finger.valid)
            truth = true_motion(rig, deflections, TIP[name])
            measured = np.linalg.norm(finger.deflection_mm)
            self.assertLess(abs(measured - truth), 1.8)

    def test_rest_reads_near_zero(self) -> None:
        rig, est, det = build((0, 0, -1))
        reading = observe(rig, est, det, {})
        for finger in reading.fingers.values():
            # Magnitude is biased high at rest: the norm of a noisy vector never
            # reads zero. This is why the signed projection is the primary output.
            self.assertLess(finger.magnitude_mm, 0.8)

    def test_incline_responds_to_a_tilted_contact(self) -> None:
        """Unequal finger deflections mean a sloped surface; the angle must move."""
        rig, est, det = build((0, 0, -1))
        flat = observe(rig, est, det, {n: 6.0 for n in TIP})
        sloped = observe(rig, est, det, {"left": 1.0, "middle": 6.0, "right": 11.0})
        self.assertIsNotNone(flat.incline_deg)
        self.assertIsNotNone(sloped.incline_deg)
        self.assertGreater(sloped.incline_deg, flat.incline_deg + 2.0,
                           "a tilted contact should register a larger incline")

    def test_reference_loss_invalidates_everything(self) -> None:
        rig, est, det = build((0, 0, -1))
        frame, _ = render(rig, {}, K, D, (1280, 720), POSE)
        by_id = det.detect(frame)[0]
        for ref in (8, 9, 10):
            by_id.pop(ref, None)
        reading = est.observe(by_id)
        self.assertFalse(reading.reference_visible)
        self.assertTrue(all(not f.valid for f in reading.fingers.values()))

    def test_survives_partial_reference_occlusion(self) -> None:
        """Learning the reference geometry is what buys this tolerance."""
        rig, est, det = build((0, 0, -1))
        deflections = {n: 6.0 for n in TIP}
        frame, _ = render(rig, deflections, K, D, (1280, 720), POSE)
        by_id = det.detect(frame)[0]
        by_id.pop(8, None)          # hide the anchor, keep the others
        reading = est.observe(by_id)
        self.assertTrue(reading.reference_visible)
        self.assertTrue(any(f.valid for f in reading.fingers.values()))

    def test_missing_finger_tag_goes_stale_then_invalid(self) -> None:
        rig, est, det = build((0, 0, -1))
        frame, _ = render(rig, {n: 4.0 for n in TIP}, K, D, (1280, 720), POSE)
        base = det.detect(frame)[0]
        for _ in range(3):
            trimmed = dict(base)
            trimmed.pop(2, None)
            reading = est.observe(trimmed)
        self.assertTrue(reading.fingers["left"].stale)
        for _ in range(12):
            trimmed = dict(base)
            trimmed.pop(2, None)
            reading = est.observe(trimmed)
        self.assertFalse(reading.fingers["left"].stale)
        self.assertFalse(reading.fingers["left"].valid)


if __name__ == "__main__":
    unittest.main()
