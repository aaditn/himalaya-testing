"""The curve a finger takes between the tags that measure it.

Two tags give two poses: a position and an orientation at each. A cubic is the
unique curve fixed by position and tangent at two endpoints, and under
Euler-Bernoulli a cantilever's deflection curve *is* a cubic - so interpolating
with one is not decoration, it is the physically correct family evaluated at
the only measurements available.

The root adds a third station for free. The finger is clamped to the cuff, so
its position and tangent there are fixed relative to the reference tag; that is
learned once at zeroing and held. Only the stretch past the last tag is
genuinely unmeasured, and it is extended straight rather than by continuing the
cubic: bending moment vanishes at a free end, so curvature really does decay
there, and an extrapolated cubic would instead curl away hardest exactly where
there is no evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


@dataclass(frozen=True)
class Station:
    """A measured cross-section: how far along the finger, and its frame."""

    y_mm: float
    position: np.ndarray      # (3,)
    rotation: np.ndarray      # (3, 3), columns are the local x, y, z axes
    measured: bool = True

    @property
    def tangent(self) -> np.ndarray:
        return self.rotation[:, 1]


def frame_from_marker(marker_rotation: np.ndarray, along_hint: np.ndarray,
                      ) -> np.ndarray:
    """Turn an ArUco marker's rotation into a finger frame.

    A tag's own x and y axes depend on which way up it was stuck, so the
    along-finger axis is chosen as whichever of +-x, +-y sits closest to the
    root-to-tip direction rather than assumed. The tag normal is taken as the
    finger's face normal, and the third axis is completed from those two.
    """
    hint = _unit(np.asarray(along_hint, float))
    candidates = [marker_rotation[:, 0], -marker_rotation[:, 0],
                  marker_rotation[:, 1], -marker_rotation[:, 1]]
    y = max(candidates, key=lambda axis: float(_unit(axis) @ hint))
    z = marker_rotation[:, 2]
    y = _unit(y - z * float(y @ z))     # keep y square to the face normal
    x = np.cross(y, z)
    return np.column_stack([_unit(x), y, _unit(z)])


def _hermite(p0, m0, p1, m1, t):
    """Cubic Hermite and its derivative, for t as an (n,) array in [0, 1]."""
    t = t[:, None]
    t2, t3 = t * t, t * t * t
    value = ((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0
             + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1)
    slope = ((6 * t2 - 6 * t) * p0 + (3 * t2 - 4 * t + 1) * m0
             + (-6 * t2 + 6 * t) * p1 + (3 * t2 - 2 * t) * m1)
    return value, slope


def _rotation_minimising_frames(points: np.ndarray, tangents: np.ndarray,
                                first_normal: np.ndarray) -> np.ndarray:
    """Sweep a frame along the curve without letting it spin about the tangent.

    Double-reflection method: each step reflects the previous frame twice, which
    lands it on the next tangent while adding no rotation about it. Naively
    rebuilding a frame from a fixed world "up" instead makes the mesh visibly
    barrel-roll wherever the finger points near that up vector.
    """
    count = len(points)
    normals = np.zeros((count, 3))
    normals[0] = _unit(first_normal - tangents[0] * float(first_normal @ tangents[0]))
    for i in range(count - 1):
        step = points[i + 1] - points[i]
        squared = float(step @ step)
        if squared < 1e-12:
            normals[i + 1] = normals[i]
            continue
        reflected_n = normals[i] - (2.0 / squared) * float(step @ normals[i]) * step
        reflected_t = tangents[i] - (2.0 / squared) * float(step @ tangents[i]) * step
        delta = tangents[i + 1] - reflected_t
        squared2 = float(delta @ delta)
        if squared2 < 1e-12:
            normals[i + 1] = _unit(reflected_n)
            continue
        normals[i + 1] = _unit(
            reflected_n - (2.0 / squared2) * float(delta @ reflected_n) * delta)
    return normals


def _roll_between(reference: np.ndarray, target: np.ndarray,
                  tangent: np.ndarray) -> float:
    """Signed angle from `reference` to `target` about `tangent`."""
    a = _unit(reference - tangent * float(reference @ tangent))
    b = _unit(target - tangent * float(target @ tangent))
    return float(np.arctan2(float(np.cross(a, b) @ tangent), float(a @ b)))


def _rotate_about(vectors: np.ndarray, axes: np.ndarray,
                  angles: np.ndarray) -> np.ndarray:
    """Rodrigues rotation, vectorised over (n, 3) inputs."""
    cos = np.cos(angles)[:, None]
    sin = np.sin(angles)[:, None]
    dot = np.sum(axes * vectors, axis=1)[:, None]
    return (vectors * cos + np.cross(axes, vectors) * sin + axes * dot * (1 - cos))


class FingerSpine:
    """A sampled, frame-carrying curve through a finger's measured stations."""

    def __init__(self, points: np.ndarray, tangents: np.ndarray,
                 normals: np.ndarray, arc: np.ndarray, measured_to: float):
        self.points = points
        self.tangents = tangents
        self.normals = normals
        self.arc = arc
        self.measured_to = measured_to

    @classmethod
    def from_stations(cls, stations: list[Station], length_mm: float,
                      samples: int = 160) -> "FingerSpine":
        ordered = sorted(stations, key=lambda s: s.y_mm)
        if len(ordered) < 2:
            raise ValueError("a spine needs at least two stations")

        knots = np.array([s.y_mm for s in ordered], float)
        positions = np.array([s.position for s in ordered], float)
        tangents = np.array([_unit(s.tangent) for s in ordered], float)

        end = max(float(length_mm), float(knots[-1]))
        arc = np.linspace(0.0, end, samples)
        points = np.zeros((samples, 3))
        slopes = np.zeros((samples, 3))

        for i in range(len(ordered) - 1):
            span = float(knots[i + 1] - knots[i])
            if span <= 1e-6:
                continue
            # Knots land in two segments; both write the same value there.
            mask = (arc >= knots[i]) & (arc <= knots[i + 1])
            if not mask.any():
                continue
            t = (arc[mask] - knots[i]) / span
            # Tangent magnitudes scale with the span so the cubic's shape is
            # independent of how the stations happen to be spaced.
            value, slope = _hermite(positions[i], tangents[i] * span,
                                    positions[i + 1], tangents[i + 1] * span, t)
            points[mask] = value
            slopes[mask] = slope / span

        beyond = arc > knots[-1]
        if beyond.any():
            points[beyond] = positions[-1] + tangents[-1] * (arc[beyond] - knots[-1])[:, None]
            slopes[beyond] = tangents[-1]

        unit_tangents = np.array([_unit(s) for s in slopes])
        normals = _rotation_minimising_frames(
            points, unit_tangents, ordered[0].rotation[:, 2])

        # The transported frame keeps the right tangent but drifts in roll
        # relative to what the tags actually measured. Correct it by blending
        # the per-station roll residuals across the curve.
        residuals = []
        for station in ordered:
            index = int(np.argmin(np.abs(arc - station.y_mm)))
            residuals.append(_roll_between(
                normals[index], station.rotation[:, 2], unit_tangents[index]))
        correction = np.interp(arc, knots, np.unwrap(residuals))
        normals = np.array([_unit(v) for v in
                            _rotate_about(normals, unit_tangents, correction)])

        return cls(points, unit_tangents, normals, arc, float(knots[-1]))

    def frames_at(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Origins and (n, 3, 3) rotations at arbitrary distances along the finger."""
        y = np.clip(np.asarray(y, float), self.arc[0], self.arc[-1])
        index = np.searchsorted(self.arc, y).clip(1, len(self.arc) - 1)
        lo, hi = index - 1, index
        span = self.arc[hi] - self.arc[lo]
        weight = np.where(span > 1e-9, (y - self.arc[lo]) / np.where(span > 1e-9, span, 1), 0.0)[:, None]

        origins = self.points[lo] * (1 - weight) + self.points[hi] * weight
        # Rotations are picked from the nearer sample rather than blended: at
        # 160 samples the frames are ~0.7 mm apart and averaging two rotation
        # matrices does not produce a rotation matrix.
        nearest = np.where(weight[:, 0] < 0.5, lo, hi)
        tangent = self.tangents[nearest]
        normal = self.normals[nearest]
        cross = np.cross(tangent, normal)
        return origins, np.stack([cross, tangent, normal], axis=-1)

    def sample(self, count: int = 48) -> np.ndarray:
        """Evenly spaced points along the spine, for drawing it as a curve."""
        want = np.linspace(self.arc[0], self.arc[-1], count)
        return np.column_stack([np.interp(want, self.arc, self.points[:, i])
                                for i in range(3)])

    def measured_fraction(self) -> float:
        total = float(self.arc[-1])
        return 1.0 if total <= 1e-9 else float(self.measured_to / total)


def build_stations(root_frame, base_pose, tip_pose, base_y: float, tip_y: float,
                   ) -> list[Station]:
    """Assemble root/base/tip stations from measured marker poses.

    `root_frame` is (position, rotation) for the clamped end, learned at zeroing.
    Either tag pose may be None; the spine simply has fewer stations.
    """
    known = [p for p in (base_pose, tip_pose) if p is not None]
    if not known:
        return []
    if base_pose is not None and tip_pose is not None:
        hint = tip_pose.translation - base_pose.translation
    else:
        hint = known[0].translation - np.asarray(root_frame[0], float)

    stations = [Station(0.0, np.asarray(root_frame[0], float),
                        np.asarray(root_frame[1], float), measured=False)]
    if base_pose is not None:
        stations.append(Station(base_y, base_pose.translation,
                                frame_from_marker(base_pose.rotation, hint)))
    if tip_pose is not None:
        stations.append(Station(tip_y, tip_pose.translation,
                                frame_from_marker(tip_pose.rotation, hint)))
    return stations
