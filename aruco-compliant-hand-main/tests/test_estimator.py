import unittest
from pathlib import Path

import numpy as np

from flexsense.config import (
    AppConfig,
    CameraConfig,
    MarkerConfig,
    OutputConfig,
    SensorConfig,
    TrackingConfig,
)
from flexsense.estimator import PlanarDeformationEstimator
from flexsense.models import ForceModelSet, PolynomialForceModel


def marker(center_x, center_y, half=5.0):
    return np.asarray(
        [
            [center_x - half, center_y - half],
            [center_x + half, center_y - half],
            [center_x + half, center_y + half],
            [center_x - half, center_y + half],
        ],
        dtype=float,
    )


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig(
            camera=CameraConfig(),
            markers=MarkerConfig(size_mm=10.0, reference_id=0),
            tracking=TrackingConfig(zero_samples=3, ema_alpha=1.0, max_missing_frames=2),
            sensors=(
                SensorConfig(
                    name="finger",
                    marker_ids=(1,),
                    normal_axis="x",
                    normal_sign=1.0,
                    shear_axis="y",
                    shear_sign=1.0,
                ),
            ),
            output=OutputConfig(Path("unused.csv"), Path("unused.json")),
        )
        self.reference = np.asarray([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)

    def test_relative_displacement_cancels_reference_frame(self):
        models = ForceModelSet(
            {"finger": {"normal": PolynomialForceModel((2.0, 0.0), clamp_min_n=0.0)}}
        )
        estimator = PlanarDeformationEstimator(self.config, models)
        baseline = {0: self.reference, 1: marker(150, 50)}
        for _ in range(3):
            estimator.observe(baseline)
        self.assertTrue(estimator.is_zeroed)

        camera_shift = np.asarray([25.0, 13.0])
        moved = {
            0: self.reference + camera_shift,
            1: marker(160 + camera_shift[0], 30 + camera_shift[1]),
        }
        result = estimator.observe(moved).sensors["finger"]
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.normal_deflection_mm, 1.0, places=6)
        self.assertAlmostEqual(result.shear_deflection_mm, 2.0, places=6)
        self.assertAlmostEqual(result.force_normal_n, 2.0, places=6)

    def test_missing_tip_is_invalid_and_temporarily_held(self):
        estimator = PlanarDeformationEstimator(self.config)
        baseline = {0: self.reference, 1: marker(150, 50)}
        for _ in range(3):
            estimator.observe(baseline)
        estimator.observe(baseline)
        result = estimator.observe({0: self.reference}).sensors["finger"]
        self.assertFalse(result.valid)
        self.assertTrue(result.stale)
        self.assertEqual(result.missing_frames, 1)


if __name__ == "__main__":
    unittest.main()
