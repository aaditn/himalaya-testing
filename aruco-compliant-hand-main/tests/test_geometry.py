import unittest

import numpy as np

from flexsense.geometry import (
    apply_homography,
    homography_from_four_points,
    signed_tip_curvature,
    wrapped_angle_delta_deg,
)


class GeometryTests(unittest.TestCase):
    def test_homography_maps_four_points(self):
        src = np.asarray([[10, 20], [110, 15], [120, 90], [5, 100]], dtype=float)
        dst = np.asarray([[-7, 7], [7, 7], [7, -7], [-7, -7]], dtype=float)
        homography = homography_from_four_points(src, dst)
        np.testing.assert_allclose(apply_homography(src, homography), dst, atol=1e-8)

    def test_angle_delta_wraps(self):
        self.assertAlmostEqual(wrapped_angle_delta_deg(-179.0, 179.0), 2.0)

    def test_straight_points_have_zero_curvature(self):
        value = signed_tip_curvature(np.asarray([[0, 0], [5, 0], [10, 0], [15, 0]]))
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()

