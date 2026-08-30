"""Keep marker poses steady enough to render without lying about them.

Two problems, both invisible while the output was a printed angle and both
glaring once a mesh is drawn on top of the finger:

*Ambiguity.* A single square gives two poses that reproject almost identically,
and the pair separates least when the marker is near face-on - which is most of
the time here. Left alone the solver picks whichever won by a hair that frame,
so the finger flips inside out at random. `solvePnPGeneric` hands back both, and
the one closer to last frame's answer is almost always the true one.

*Jitter.* Corner noise of a few tenths of a pixel is a degree or so of tilt.
A one-euro filter is the right tool: it smooths hard when the pose is holding
still and barely at all when it is moving, so the display settles without
adding the lag a fixed low-pass would.
"""

from __future__ import annotations

import numpy as np

from .pose3d import Pose, square_object_points
from .vision import require_cv2


class OneEuroFilter:
    """Adaptive low-pass: cutoff rises with speed, so lag falls when it matters."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03,
                 derivative_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._value = None
        self._derivative = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._value = None
        self._derivative = None

    def __call__(self, sample, dt: float):
        sample = np.asarray(sample, float)
        if self._value is None:
            self._value = sample
            self._derivative = np.zeros_like(sample)
            return sample
        rate = (sample - self._value) / max(dt, 1e-6)
        alpha_d = self._alpha(self.derivative_cutoff, dt)
        self._derivative = alpha_d * rate + (1 - alpha_d) * self._derivative
        speed = float(np.linalg.norm(self._derivative))
        alpha = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self._value = alpha * sample + (1 - alpha) * self._value
        return self._value


def _rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    cv2 = require_cv2()
    rvec, _ = cv2.Rodrigues(a.T @ b)
    return float(np.linalg.norm(rvec))


def _slerp_towards(current: np.ndarray, target: np.ndarray, weight: float) -> np.ndarray:
    """Move `current` a fraction `weight` of the way to `target`."""
    cv2 = require_cv2()
    rvec, _ = cv2.Rodrigues(current.T @ target)
    step, _ = cv2.Rodrigues(rvec * float(weight))
    return current @ step


class MarkerPoseTracker:
    """Per-tag pose solving with ambiguity resolution and temporal smoothing."""

    def __init__(self, size_mm: float, camera_matrix, dist_coeffs,
                 min_cutoff: float = 1.5, beta: float = 0.04,
                 rotation_gain: float = 0.55, flip_tolerance: float = 0.35):
        self.size_mm = float(size_mm)
        self.camera_matrix = np.asarray(camera_matrix, float)
        self.dist_coeffs = np.asarray(dist_coeffs, float)
        self.rotation_gain = float(rotation_gain)
        self.flip_tolerance = float(flip_tolerance)
        self._filters: dict[int, OneEuroFilter] = {}
        self._rotations: dict[int, np.ndarray] = {}
        self._settings = (min_cutoff, beta)
        self.flips_rejected = 0

    def reset(self) -> None:
        self._filters.clear()
        self._rotations.clear()
        self.flips_rejected = 0

    def solve(self, marker_id: int, corners: np.ndarray, dt: float) -> Pose | None:
        cv2 = require_cv2()
        try:
            count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                square_object_points(self.size_mm),
                np.asarray(corners, np.float32).reshape(4, 1, 2),
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if not count:
            return None

        candidates = []
        for rvec, tvec, error in zip(rvecs, tvecs, np.asarray(errors).ravel()):
            matrix, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
            candidates.append((matrix, np.asarray(tvec, float).reshape(3), float(error)))

        previous = self._rotations.get(marker_id)
        if previous is None or len(candidates) == 1:
            rotation, translation, _ = candidates[0]
        else:
            best = min(candidates, key=lambda c: _rotation_distance(previous, c[0]))
            chosen = min(candidates, key=lambda c: c[2])
            if best is not chosen:
                self.flips_rejected += 1
            # Only override the solver when its own preference is a genuine
            # flip away from continuity, not when both agree closely.
            rotation, translation, _ = (
                best if _rotation_distance(previous, chosen[0]) > self.flip_tolerance
                else chosen)

        smoothed_rotation = (rotation if previous is None else
                             _slerp_towards(previous, rotation, self.rotation_gain))
        self._rotations[marker_id] = smoothed_rotation

        if marker_id not in self._filters:
            self._filters[marker_id] = OneEuroFilter(*self._settings)
        smoothed_translation = self._filters[marker_id](translation, dt)
        return Pose(smoothed_rotation, np.asarray(smoothed_translation, float))
