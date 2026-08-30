"""Full 6-DoF deformation estimator.

Measures each finger tag's pose in a rigid reference frame carried by the hand
itself, so camera motion, arm motion and mount flex cancel out. Unlike the
planar estimator this makes no assumption that bending happens in the image
plane, which is what lets it read deflection into or out of the camera axis -
the case that matters when the hand presses on a slope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .pose3d import (
    Pose,
    angle_between_deg,
    fit_plane,
    rotation_angle_deg,
    solve_marker_pose,
    solve_rigid_group,
    square_object_points,
)


@dataclass(frozen=True)
class FingerSpec3D:
    """One finger. `tag_ids` run root to tip; the last is the tip tag.

    Axes are left as None by default and derived from the finger's own rest
    geometry during zeroing. Hand-specifying them means writing vectors in the
    reference *marker's* frame, which is arbitrary and easy to get backwards;
    the derived frame is physically meaningful instead:

      along  - root tag to tip tag, down the finger
      normal - out of the finger's face, the direction it presses
      shear  - across the finger, in the plane of its face
    """

    name: str
    tag_ids: tuple[int, ...]
    normal_axis: tuple[float, float, float] | None = None
    shear_axis: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Spatial3DConfig:
    reference_ids: tuple[int, ...]
    fingers: tuple[FingerSpec3D, ...]
    tag_mm: float = 20.0
    zero_samples: int = 30
    max_missing_frames: int = 8
    press_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass
class FingerReading:
    valid: bool
    stale: bool = False
    tags_seen: int = 0
    deflection_mm: list[float] | None = None      # 3D vector, reference frame
    magnitude_mm: float | None = None
    normal_mm: float | None = None
    shear_mm: float | None = None
    along_mm: float | None = None
    tip_rotation_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Frame3DReading:
    status: str
    reference_visible: bool
    zero_progress: int
    zero_target: int
    fingers: dict[str, FingerReading] = field(default_factory=dict)
    contact_plane_normal: list[float] | None = None
    incline_deg: float | None = None
    tip_positions: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "status": self.status,
            "reference_visible": self.reference_visible,
            "zero_progress": self.zero_progress,
            "zero_target": self.zero_target,
            "contact_plane_normal": self.contact_plane_normal,
            "incline_deg": self.incline_deg,
            "tip_positions": self.tip_positions,
            "fingers": {k: v.to_dict() for k, v in self.fingers.items()},
        }
        return data


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class SpatialDeformationEstimator:
    """Tracks finger tag poses relative to a rigid reference constellation.

    With several reference tags the geometry between them is learned during
    zeroing rather than measured by hand, so afterwards any visible subset can
    fix the frame. That removes the single-point-of-failure a lone reference tag
    creates, where one occlusion invalidates every finger for that frame.
    """

    def __init__(self, config: Spatial3DConfig, camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray):
        self.config = config
        self.camera_matrix = np.asarray(camera_matrix, float)
        self.dist_coeffs = np.asarray(dist_coeffs, float)
        self._zero_frames: list[dict[int, Pose]] = []
        self._baseline: dict[int, np.ndarray] | None = None
        self._baseline_rotation: dict[int, np.ndarray] = {}
        # reference tag id -> its corners in the anchor tag's frame
        self._reference_model: dict[int, np.ndarray] | None = None
        self._axes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._missing: dict[str, int] = {}
        self._last: dict[str, FingerReading] = {}

    # ---- reference frame -------------------------------------------------

    @property
    def _anchor(self) -> int:
        return self.config.reference_ids[0]

    def _learn_reference_model(self, samples: list[dict[int, Pose]]) -> None:
        """Fix each reference tag's corners in the anchor tag's frame."""
        corners = square_object_points(self.config.tag_mm).astype(float)
        model: dict[int, list[np.ndarray]] = {}
        for poses in samples:
            if self._anchor not in poses:
                continue
            to_anchor = poses[self._anchor].inverse()
            for tag_id in self.config.reference_ids:
                if tag_id not in poses:
                    continue
                in_camera = poses[tag_id].apply(corners)
                model.setdefault(tag_id, []).append(to_anchor.apply(in_camera))
        if model:
            self._reference_model = {k: np.mean(v, axis=0) for k, v in model.items()}

    def _reference_pose(self, by_id: dict[int, np.ndarray]) -> Pose | None:
        visible = [t for t in self.config.reference_ids if t in by_id]
        if not visible:
            return None
        if self._reference_model is not None and len(visible) >= 1:
            object_points = np.vstack([self._reference_model[t] for t in visible
                                       if t in self._reference_model])
            image_points = np.vstack([np.asarray(by_id[t], float).reshape(4, 2)
                                      for t in visible if t in self._reference_model])
            if len(object_points) >= 4:
                return solve_rigid_group(object_points, image_points,
                                         self.camera_matrix, self.dist_coeffs)
        if self._anchor in by_id:
            return solve_marker_pose(by_id[self._anchor], self.config.tag_mm,
                                     self.camera_matrix, self.dist_coeffs)
        return None

    # ---- main entry point ------------------------------------------------

    def observe(self, by_id: dict[int, np.ndarray]) -> Frame3DReading:
        poses: dict[int, Pose] = {}
        for tag_id, corners in by_id.items():
            pose = solve_marker_pose(corners, self.config.tag_mm,
                                     self.camera_matrix, self.dist_coeffs)
            if pose is not None:
                poses[tag_id] = pose

        reference = self._reference_pose(by_id)
        if reference is None:
            return self._all_invalid("reference lost", False)
        to_reference = reference.inverse()

        # Every tag expressed in the hand's own rigid frame.
        local: dict[int, Pose] = {
            tag_id: to_reference.compose(pose) for tag_id, pose in poses.items()
        }

        if self._baseline is None:
            self._zero_frames.append(poses)
            if len(self.config.reference_ids) > 1:
                self._learn_reference_model(self._zero_frames)
            if len(self._zero_frames) >= self.config.zero_samples:
                self._finish_zero()
            return self._all_invalid("zeroing", True)

        reading = Frame3DReading("tracking", True, self.config.zero_samples,
                                 self.config.zero_samples)
        tips: dict[str, np.ndarray] = {}
        for finger in self.config.fingers:
            reading.fingers[finger.name] = self._finger(finger, local, tips)

        if len(tips) >= 3:
            fitted = fit_plane(np.array(list(tips.values())),
                               toward=self.config.press_direction)
            if fitted is not None:
                _centroid, normal = fitted
                reading.contact_plane_normal = [float(v) for v in normal]
                reading.incline_deg = angle_between_deg(
                    normal, self._baseline_plane_normal)
        reading.tip_positions = {k: [float(v) for v in tips[k]] for k in tips}
        return reading

    def _finish_zero(self) -> None:
        corners = square_object_points(self.config.tag_mm).astype(float)
        _ = corners
        accumulated: dict[int, list[np.ndarray]] = {}
        rotations: dict[int, list[np.ndarray]] = {}
        for poses in self._zero_frames:
            reference = poses.get(self._anchor)
            if reference is None:
                continue
            to_reference = reference.inverse()
            for tag_id, pose in poses.items():
                local = to_reference.compose(pose)
                accumulated.setdefault(tag_id, []).append(local.translation)
                rotations.setdefault(tag_id, []).append(local.rotation)
        self._baseline = {k: np.mean(v, axis=0) for k, v in accumulated.items()}
        self._baseline_rotation = {k: v[len(v) // 2] for k, v in rotations.items()}
        self._axes = {}
        for finger in self.config.fingers:
            tip_id = finger.tag_ids[-1]
            root_id = finger.tag_ids[0]
            if tip_id not in self._baseline:
                continue
            face_normal = self._baseline_rotation[tip_id][:, 2]
            if root_id in self._baseline and root_id != tip_id:
                along = self._baseline[tip_id] - self._baseline[root_id]
                along = along / max(np.linalg.norm(along), 1e-9)
            else:
                along = self._baseline_rotation[tip_id][:, 1]
            # Orthonormalise: normal out of the face, shear across the finger.
            normal = _unit(face_normal - along * float(face_normal @ along))
            if finger.normal_axis is not None:
                normal = _unit(finger.normal_axis)
            shear = (_unit(finger.shear_axis) if finger.shear_axis is not None
                     else _unit(np.cross(along, normal)))
            self._axes[finger.name] = (normal, shear, along)

        tips = [self._baseline[f.tag_ids[-1]] for f in self.config.fingers
                if f.tag_ids[-1] in self._baseline]
        fitted = fit_plane(np.array(tips), toward=self.config.press_direction) \
            if len(tips) >= 3 else None
        self._baseline_plane_normal = (fitted[1] if fitted is not None
                                       else np.asarray(self.config.press_direction, float))

    def _finger(self, finger: FingerSpec3D, local: dict[int, Pose],
                tips: dict[str, np.ndarray]) -> FingerReading:
        tip_id = finger.tag_ids[-1]
        seen = [t for t in finger.tag_ids if t in local]
        if tip_id not in local or self._baseline is None or tip_id not in self._baseline:
            missed = self._missing.get(finger.name, 0) + 1
            self._missing[finger.name] = missed
            if missed <= self.config.max_missing_frames and finger.name in self._last:
                stale = FingerReading(**{**self._last[finger.name].to_dict(),
                                         "valid": False, "stale": True})
                return stale
            return FingerReading(valid=False, tags_seen=len(seen))
        self._missing[finger.name] = 0

        pose = local[tip_id]
        tips[finger.name] = pose.translation
        delta = pose.translation - self._baseline[tip_id]
        normal_axis, shear_axis, along_axis = self._axes.get(
            finger.name, (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]),
                          np.array([0.0, 1.0, 0.0])))
        reading = FingerReading(
            valid=True,
            tags_seen=len(seen),
            deflection_mm=[float(v) for v in delta],
            magnitude_mm=float(np.linalg.norm(delta)),
            normal_mm=float(delta @ normal_axis),
            shear_mm=float(delta @ shear_axis),
            along_mm=float(delta @ along_axis),
            tip_rotation_deg=rotation_angle_deg(
                self._baseline_rotation.get(tip_id, pose.rotation), pose.rotation),
        )
        self._last[finger.name] = reading
        return reading

    def _all_invalid(self, status: str, reference_visible: bool) -> Frame3DReading:
        return Frame3DReading(
            status=status,
            reference_visible=reference_visible,
            zero_progress=len(self._zero_frames),
            zero_target=self.config.zero_samples,
            fingers={f.name: FingerReading(valid=False) for f in self.config.fingers},
        )

    def reset_zero(self) -> None:
        self._zero_frames.clear()
        self._baseline = None
        self._baseline_rotation.clear()
        self._reference_model = None
        self._axes.clear()
