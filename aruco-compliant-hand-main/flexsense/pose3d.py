"""Rigid-body pose primitives for marker-based 3D measurement.

Everything here works in millimetres and returns poses as object-to-camera
transforms: X_camera = R @ X_object + t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vision import require_cv2


def square_object_points(size_mm: float) -> np.ndarray:
    """Marker corners in its own frame, in OpenCV's detected-corner order.

    Top-left, top-right, bottom-right, bottom-left, with +x right, +y up and
    the marker lying on z=0. This is the convention aruco pose estimation uses.
    """
    half = size_mm / 2.0
    return np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)


@dataclass(frozen=True)
class Pose:
    """Object-to-camera rigid transform."""

    rotation: np.ndarray      # 3x3
    translation: np.ndarray   # (3,)

    @classmethod
    def from_rvec(cls, rvec, tvec) -> "Pose":
        cv2 = require_cv2()
        matrix, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
        return cls(matrix, np.asarray(tvec, float).reshape(3))

    def inverse(self) -> "Pose":
        transposed = self.rotation.T
        return Pose(transposed, -transposed @ self.translation)

    def compose(self, other: "Pose") -> "Pose":
        """self applied after other."""
        return Pose(self.rotation @ other.rotation,
                    self.rotation @ other.translation + self.translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 3)
        return (self.rotation @ pts.T).T + self.translation

    @property
    def normal(self) -> np.ndarray:
        """The object's +z axis expressed in the target frame."""
        return self.rotation[:, 2]

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.translation))


def solve_marker_pose(corners: np.ndarray, size_mm: float, camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray) -> Pose | None:
    """Pose of one square marker from its four detected corners.

    IPPE_SQUARE is the planar-specific solver. A lone square carries a two-fold
    ambiguity that can flip when the marker is near face-on; solving several
    markers as one rigid body (see `solve_rigid_group`) removes it.
    """
    cv2 = require_cv2()
    ok, rvec, tvec = cv2.solvePnP(
        square_object_points(size_mm),
        np.asarray(corners, dtype=np.float32).reshape(4, 1, 2),
        camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return Pose.from_rvec(rvec, tvec) if ok else None


def solve_rigid_group(object_points: np.ndarray, image_points: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Pose | None:
    """Pose of a rigid constellation of markers from all their corners at once.

    More corners spread over a wider baseline make the solution far better
    conditioned than any single marker, and it keeps working when part of the
    constellation is hidden.
    """
    cv2 = require_cv2()
    object_points = np.asarray(object_points, np.float32).reshape(-1, 1, 3)
    image_points = np.asarray(image_points, np.float32).reshape(-1, 1, 2)
    if len(object_points) < 4:
        return None
    flags = cv2.SOLVEPNP_ITERATIVE if len(object_points) > 4 else cv2.SOLVEPNP_IPPE_SQUARE
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                  dist_coeffs, flags=flags)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix,
                                      dist_coeffs, rvec, tvec)
    return Pose.from_rvec(rvec, tvec)


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Magnitude of the rotation taking `first` to `second`."""
    relative = np.asarray(second, float) @ np.asarray(first, float).T
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def fit_plane(points: np.ndarray, toward=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares plane through 3 or more points; returns (centroid, unit normal).

    This is what turns individual fingertip positions into a statement about the
    surface the hand is resting on. The SVD normal has arbitrary sign, so pass
    `toward` (any vector on the side the normal should face, typically the
    direction the hand presses) to get a consistent orientation frame to frame.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    if toward is not None and float(normal @ np.asarray(toward, float)) < 0.0:
        normal = -normal
    return centroid, normal


def angle_between_deg(first, second) -> float:
    a = np.asarray(first, float)
    b = np.asarray(second, float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / denominator, -1.0, 1.0))))
