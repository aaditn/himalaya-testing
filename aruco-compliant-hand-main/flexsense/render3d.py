"""Draw 3D geometry over a camera frame, and from invented viewpoints.

Two projectors share one interface. The pinhole one uses the measured
intrinsics so a rendered finger lands exactly on the real one in the video. The
orthographic one is what makes the inset side view possible: once poses are
solved in the reference tag's frame, re-rendering them from any angle costs
nothing, and a viewpoint square to the bend is the one where the bend is
actually visible. From the wrist camera the fingers point nearly at the lens,
so their curvature is foreshortened into almost nothing.

Triangles are drawn back to front. A depth buffer would be more correct, but
painter's ordering on a few hundred convex triangles is a few milliseconds in
numpy where a per-pixel buffer in Python is not.
"""

from __future__ import annotations

import numpy as np

from .vision import require_cv2

DEFAULT_LIGHT = np.array([0.35, -0.55, -0.75])


class PinholeProjector:
    """Projects reference-frame points into the real camera image."""

    def __init__(self, camera_matrix, dist_coeffs, reference_pose):
        self.camera_matrix = np.asarray(camera_matrix, float)
        self.dist_coeffs = np.asarray(dist_coeffs, float)
        self.rotation = np.asarray(reference_pose.rotation, float)
        self.translation = np.asarray(reference_pose.translation, float)

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) @ self.rotation.T + self.translation

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cv2 = require_cv2()
        camera_points = self.to_camera(points)
        depth = camera_points[:, 2]
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(points, dtype=np.float64),
            np.asarray(self._rvec(), float), self.translation,
            self.camera_matrix, self.dist_coeffs)
        return projected.reshape(-1, 2), depth

    def _rvec(self):
        cv2 = require_cv2()
        rvec, _ = cv2.Rodrigues(self.rotation)
        return rvec

    @property
    def eye_direction(self) -> np.ndarray:
        # Viewing direction expressed back in the reference frame.
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])


class OrthoProjector:
    """A virtual orthographic camera looking at the hand from any angle."""

    def __init__(self, forward, up, centre, scale, offset):
        forward = _unit(np.asarray(forward, float))
        up = np.asarray(up, float)
        right = _unit(np.cross(up, forward))
        true_up = np.cross(forward, right)
        self.basis = np.vstack([right, true_up, forward])
        self.centre = np.asarray(centre, float)
        self.scale = float(scale)
        self.offset = np.asarray(offset, float)

    @classmethod
    def framing(cls, points: np.ndarray, forward, up, size: tuple[int, int],
                margin: int = 14) -> "OrthoProjector":
        """Fit a view of `points` into a `size` (w, h) panel."""
        points = np.asarray(points, float).reshape(-1, 3)
        centre = points.mean(axis=0)
        probe = cls(forward, up, centre, 1.0, np.zeros(2))
        flat = probe._to_plane(points)
        span = np.ptp(flat, axis=0)
        span[span < 1e-6] = 1.0
        width, height = size
        scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
        mid = (flat.max(axis=0) + flat.min(axis=0)) / 2.0
        offset = np.array([width / 2.0, height / 2.0]) - mid * scale * np.array([1, -1])
        return cls(forward, up, centre, scale, offset)

    def _to_plane(self, points: np.ndarray) -> np.ndarray:
        local = (np.asarray(points, float).reshape(-1, 3) - self.centre) @ self.basis.T
        return local[:, :2]

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = (np.asarray(points, float).reshape(-1, 3) - self.centre) @ self.basis.T
        pixels = local[:, :2] * self.scale * np.array([1, -1]) + self.offset
        return pixels, local[:, 2]

    @property
    def eye_direction(self) -> np.ndarray:
        return self.basis[2]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


def shade(colour, amount: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * amount, 0, 255)) for channel in colour)


def draw_mesh(canvas: np.ndarray, triangles: np.ndarray, projector,
              colour: tuple[int, int, int], alpha: float = 1.0,
              edge_colour: tuple[int, int, int] | None = None,
              light=DEFAULT_LIGHT, ambient: float = 0.34) -> None:
    """Paint a triangle mesh, shaded, with optional constant transparency."""
    cv2 = require_cv2()
    triangles = np.asarray(triangles, float)
    if not len(triangles):
        return

    pixels, depth = projector.project(triangles.reshape(-1, 3))
    pixels = pixels.reshape(-1, 3, 2)
    depth = depth.reshape(-1, 3).mean(axis=1)

    finite = np.isfinite(pixels).all(axis=(1, 2)) & np.isfinite(depth)
    # Geometry behind the camera projects to a mirrored ghost in front of it.
    finite &= depth > 1.0
    if not finite.any():
        return
    pixels, depth = pixels[finite], depth[finite]
    faces = triangles[finite]

    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals = normals / lengths

    # Winding tells us which side we are looking at; a closed mesh only needs
    # the near half drawn, which halves the fill cost.
    signed = ((pixels[:, 1, 0] - pixels[:, 0, 0]) * (pixels[:, 2, 1] - pixels[:, 0, 1])
              - (pixels[:, 2, 0] - pixels[:, 0, 0]) * (pixels[:, 1, 1] - pixels[:, 0, 1]))
    front = signed < 0
    if front.sum() < 3:
        front = np.ones(len(faces), bool)
    pixels, depth, normals = pixels[front], depth[front], normals[front]

    light = _unit(np.asarray(light, float))
    lit = ambient + (1.0 - ambient) * np.clip(np.abs(normals @ light), 0.0, 1.0)

    target = canvas if alpha >= 0.999 else canvas.copy()
    order = np.argsort(-depth)
    corners = pixels.astype(np.int32)
    for index in order:
        cv2.fillConvexPoly(target, corners[index], shade(colour, lit[index]),
                           cv2.LINE_AA)
    if edge_colour is not None:
        cv2.polylines(target, corners[order], True, edge_colour, 1, cv2.LINE_AA)

    if alpha < 0.999:
        cv2.addWeighted(target, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


def draw_polyline_3d(canvas: np.ndarray, points: np.ndarray, projector,
                     colour, thickness: int = 2,
                     dashed_from: float | None = None) -> None:
    """Draw a 3D polyline; past `dashed_from` (a 0-1 fraction) it goes dashed."""
    cv2 = require_cv2()
    pixels, depth = projector.project(points)
    good = np.isfinite(pixels).all(axis=1) & (depth > 1.0)
    if good.sum() < 2:
        return
    cut = len(points) if dashed_from is None else int(len(points) * dashed_from)
    solid = pixels[:cut][good[:cut]].astype(np.int32)
    if len(solid) >= 2:
        cv2.polylines(canvas, [solid], False, colour, thickness, cv2.LINE_AA)
    tail = pixels[cut:][good[cut:]].astype(np.int32)
    for i in range(0, len(tail) - 1, 2):
        cv2.line(canvas, tuple(tail[i]), tuple(tail[i + 1]), colour, thickness,
                 cv2.LINE_AA)


def draw_points_3d(canvas: np.ndarray, points: np.ndarray, projector, colour,
                   radius: int = 3) -> None:
    cv2 = require_cv2()
    pixels, depth = projector.project(points)
    for pixel, z in zip(pixels, depth):
        if np.isfinite(pixel).all() and z > 1.0:
            cv2.circle(canvas, tuple(pixel.astype(int)), radius, colour, -1, cv2.LINE_AA)
