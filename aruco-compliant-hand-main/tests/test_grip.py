"""Grip classification must separate wrapping from back-bending.

The unsigned angle between two normals is symmetric, so it reports the same
value for both. These tests pin the signed behaviour that distinguishes them.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from flexsense.grip import (
    GripClassifier,
    GripState,
    GripThresholds,
    signed_bend_deg,
    unsigned_angle_deg,
)
from flexsense.pose3d import Pose


def finger(bend_deg: float, length_mm: float = 30.0):
    base = Pose(np.eye(3), np.array([0.0, 0.0, 0.0]))
    rotation, _ = cv2.Rodrigues(np.array([[np.radians(bend_deg)], [0.0], [0.0]]))
    return base, Pose(rotation, np.array([0.0, length_mm, 0.0]))


def bend_axis(base, tip):
    along = tip.translation - base.translation
    along = along / np.linalg.norm(along)
    axis = np.cross(along, base.normal)
    return axis / np.linalg.norm(axis)


class TestSignedBend(unittest.TestCase):
    def test_unsigned_cannot_separate_the_two_cases(self) -> None:
        up_base, up_tip = finger(18.0)
        down_base, down_tip = finger(-18.0)
        self.assertAlmostEqual(unsigned_angle_deg(up_base.normal, up_tip.normal),
                               unsigned_angle_deg(down_base.normal, down_tip.normal),
                               places=6)

    def test_signed_does_separate_them(self) -> None:
        up_base, up_tip = finger(18.0)
        down_base, down_tip = finger(-18.0)
        up = signed_bend_deg(up_base.normal, up_tip.normal, bend_axis(up_base, up_tip))
        down = signed_bend_deg(down_base.normal, down_tip.normal,
                               bend_axis(down_base, down_tip))
        self.assertAlmostEqual(up, -down, places=6)
        self.assertGreater(up * down, -1e9)
        self.assertLess(up * down, 0.0)

    def test_signed_magnitude_matches_applied_bend(self) -> None:
        for applied in [-25.0, -10.0, 0.0, 10.0, 25.0]:
            base, tip = finger(applied)
            measured = signed_bend_deg(base.normal, tip.normal, bend_axis(base, tip))
            self.assertAlmostEqual(abs(measured), abs(applied), places=4)


class TestClassification(unittest.TestCase):
    def _classifier(self, wrap_sign: float = 1.0) -> GripClassifier:
        clf = GripClassifier(GripThresholds(wrap_deg=6.0, backbend_deg=4.0,
                                            wrap_sign=wrap_sign))
        for name in ("left", "middle", "right"):
            clf.learn_rest(name, *finger(0.0))
        return clf

    def test_all_wrapping_is_good(self) -> None:
        clf = self._classifier()
        result = clf.assess({n: finger(v) for n, v in
                             [("left", 14), ("middle", 11), ("right", 16)]})
        self.assertEqual(result.verdict, "good")
        self.assertEqual(result.wrapping, 3)

    def test_any_backbending_is_bad(self) -> None:
        """One finger bending the wrong way condemns the hold, not a majority vote."""
        clf = self._classifier()
        result = clf.assess({n: finger(v) for n, v in
                             [("left", 14), ("middle", 11), ("right", -11)]})
        self.assertEqual(result.verdict, "bad")
        self.assertEqual(result.backbending, 1)

    def test_light_load_is_no_contact(self) -> None:
        clf = self._classifier()
        result = clf.assess({n: finger(v) for n, v in
                             [("left", 1), ("middle", -1), ("right", 2)]})
        self.assertEqual(result.verdict, "no-contact")

    def test_single_finger_is_marginal(self) -> None:
        clf = self._classifier()
        result = clf.assess({n: finger(v) for n, v in
                             [("left", 13), ("middle", 1), ("right", 0)]})
        self.assertEqual(result.verdict, "marginal")

    def test_wrap_sign_flips_the_interpretation(self) -> None:
        """Which curvature means 'wrapping' depends on which face the tags are on."""
        bends = {n: finger(v) for n, v in [("left", 14), ("middle", 11), ("right", 16)]}
        self.assertEqual(self._classifier(wrap_sign=1.0).assess(bends).verdict, "good")
        self.assertEqual(self._classifier(wrap_sign=-1.0).assess(bends).verdict, "bad")

    def test_rest_curvature_is_subtracted(self) -> None:
        """A flexure that is not straight at rest must still read zero when unloaded."""
        clf = GripClassifier(GripThresholds(wrap_deg=6.0, backbend_deg=4.0))
        clf.learn_rest("left", *finger(9.0))       # already curved when unloaded
        result = clf.assess_finger("left", *finger(9.0))
        self.assertEqual(result.state, GripState.NEUTRAL.value)
        self.assertLess(abs(result.signed_bend_deg), 0.5)

    def test_missing_tag_reports_unknown(self) -> None:
        clf = self._classifier()
        base, _tip = finger(10.0)
        result = clf.assess_finger("left", base, None)
        self.assertEqual(result.state, GripState.UNKNOWN.value)


if __name__ == "__main__":
    unittest.main()
