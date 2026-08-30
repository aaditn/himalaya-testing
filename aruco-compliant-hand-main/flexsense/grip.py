"""Classify grip quality from how each compliant finger has curved.

The physical idea: tags along a finger sit on a surface that is convex when the
finger wraps around what it is pushing on, and concave when it bends the wrong
way. Surface normals diverge on a convex surface and converge on a concave one,
so the relative tilt between a finger's base and tip tags says which is
happening.

The angle *between* two normals is unsigned and therefore cannot tell those two
cases apart - both wrapping and back-bending produce a positive angle. What
separates them is the direction of rotation, so bend is measured as a signed
rotation about the finger's own bend axis.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class GripState(str, Enum):
    WRAPPING = "wrapping"        # curving around the object - a good hold
    NEUTRAL = "neutral"          # barely loaded, or not in contact
    BACKBENDING = "backbending"  # curving the wrong way - a bad hold
    UNKNOWN = "unknown"          # tags missing


@dataclass(frozen=True)
class GripThresholds:
    """Degrees of signed bend separating the three states.

    `wrap_sign` is +1 or -1 and depends on which face of the finger the tags are
    stuck to. It cannot be derived from geometry alone - calibrate it once by
    making a known good grip and reading the reported sign.
    """

    neutral_deg: float = 3.0
    wrap_deg: float = 6.0
    backbend_deg: float = 4.0
    wrap_sign: float = 1.0


@dataclass
class FingerGrip:
    name: str
    state: str = GripState.UNKNOWN.value
    signed_bend_deg: float | None = None
    unsigned_angle_deg: float | None = None
    chord_change_mm: float | None = None
    tags_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GripAssessment:
    verdict: str
    reason: str
    fingers: dict[str, FingerGrip] = field(default_factory=dict)
    wrapping: int = 0
    backbending: int = 0
    neutral: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "wrapping": self.wrapping,
            "backbending": self.backbending,
            "neutral": self.neutral,
            "fingers": {k: v.to_dict() for k, v in self.fingers.items()},
        }


def _unit(vector) -> np.ndarray:
    arr = np.asarray(vector, float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 1e-9 else arr


def signed_bend_deg(base_normal, tip_normal, bend_axis) -> float:
    """Rotation from base normal to tip normal, signed about `bend_axis`.

    Positive and negative correspond to the two curvature directions. Which one
    means "wrapping" depends on the mounting and is set by `wrap_sign`.
    """
    first = _unit(base_normal)
    second = _unit(tip_normal)
    axis = _unit(bend_axis)
    cross = np.cross(first, second)
    return float(np.degrees(np.arctan2(float(cross @ axis), float(first @ second))))


def unsigned_angle_deg(first, second) -> float:
    cosine = float(_unit(first) @ _unit(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class GripClassifier:
    """Turns per-finger tag poses into a grip verdict.

    Bend is measured relative to the finger's own unloaded shape rather than
    against zero, because a printed flexure rarely starts perfectly straight and
    the tags are never mounted perfectly coplanar.
    """

    def __init__(self, thresholds: GripThresholds | None = None):
        self.thresholds = thresholds or GripThresholds()
        self._rest_bend: dict[str, float] = {}
        self._bend_axis: dict[str, np.ndarray] = {}
        self._rest_chord: dict[str, float] = {}

    def learn_rest(self, name: str, base_pose, tip_pose) -> None:
        """Record a finger's unloaded bend and its bend axis."""
        along = _unit(tip_pose.translation - base_pose.translation)
        normal = _unit(base_pose.normal)
        axis = np.cross(along, normal)
        if np.linalg.norm(axis) < 1e-6:
            # Degenerate only if the tag faces straight down its own finger.
            axis = np.cross(along, np.array([0.0, 0.0, 1.0]))
        axis = _unit(axis)
        self._bend_axis[name] = axis
        self._rest_bend[name] = signed_bend_deg(base_pose.normal, tip_pose.normal, axis)
        self._rest_chord[name] = float(np.linalg.norm(
            tip_pose.translation - base_pose.translation))

    def assess_finger(self, name: str, base_pose, tip_pose) -> FingerGrip:
        if base_pose is None or tip_pose is None:
            return FingerGrip(name=name, state=GripState.UNKNOWN.value,
                              tags_seen=sum(p is not None for p in (base_pose, tip_pose)))
        if name not in self._bend_axis:
            self.learn_rest(name, base_pose, tip_pose)
        axis = self._bend_axis[name]
        bend = signed_bend_deg(base_pose.normal, tip_pose.normal, axis)
        relative = bend - self._rest_bend.get(name, 0.0)
        oriented = relative * self.thresholds.wrap_sign

        if oriented >= self.thresholds.wrap_deg:
            state = GripState.WRAPPING
        elif oriented <= -self.thresholds.backbend_deg:
            state = GripState.BACKBENDING
        else:
            state = GripState.NEUTRAL

        chord = float(np.linalg.norm(tip_pose.translation - base_pose.translation))
        return FingerGrip(
            name=name,
            state=state.value,
            signed_bend_deg=round(relative, 2),
            # Kept for comparison: this is what an unsigned measurement would
            # have reported, and it cannot separate wrapping from back-bending.
            unsigned_angle_deg=round(
                unsigned_angle_deg(base_pose.normal, tip_pose.normal), 2),
            chord_change_mm=round(chord - self._rest_chord.get(name, chord), 3),
            tags_seen=2,
        )

    def assess(self, finger_poses: dict[str, tuple]) -> GripAssessment:
        """`finger_poses` maps finger name to (base_pose, tip_pose); either may be None."""
        results = {name: self.assess_finger(name, *poses)
                   for name, poses in finger_poses.items()}
        wrapping = sum(1 for f in results.values() if f.state == GripState.WRAPPING.value)
        backbending = sum(1 for f in results.values()
                          if f.state == GripState.BACKBENDING.value)
        neutral = sum(1 for f in results.values() if f.state == GripState.NEUTRAL.value)
        unknown = sum(1 for f in results.values() if f.state == GripState.UNKNOWN.value)

        if unknown and unknown == len(results):
            verdict, reason = "unknown", "no finger tags visible"
        elif backbending:
            # One finger bending the wrong way means the hand is pushing against
            # the hold rather than into it; treat it as unsafe regardless of the
            # others, since that is the finger most likely to slip.
            verdict = "bad"
            reason = f"{backbending} finger(s) back-bending"
        elif wrapping >= 2:
            verdict, reason = "good", f"{wrapping} finger(s) wrapping"
        elif wrapping == 1:
            verdict, reason = "marginal", "only one finger wrapping"
        else:
            verdict, reason = "no-contact", "no finger loaded enough to classify"
        return GripAssessment(verdict=verdict, reason=reason, fingers=results,
                              wrapping=wrapping, backbending=backbending, neutral=neutral)

    def reset(self) -> None:
        self._rest_bend.clear()
        self._bend_axis.clear()
        self._rest_chord.clear()


# ---- live runner -------------------------------------------------------


def bend_deg_for_tip_travel(tip_travel_mm: float, base_mm: float, tip_mm: float,
                            length_mm: float) -> float:
    """Signed bend the tags would report for a given tip deflection.

    Pure geometry: an end-loaded cantilever takes the shape
    y(s) = d(3r^2 - r^3)/2 with r = s/length, and the tags measure the
    difference in slope angle between their two stations. Nothing here depends
    on how stiff the flexure is, so the conversion carries straight over to the
    printed part - which is what makes it usable for setting thresholds in
    millimetres of travel rather than in degrees nobody has an intuition for.

    This is small-deflection beam theory, so it is a good approximation for the
    handful of millimetres a grip actually produces and drifts once the tip
    moves an appreciable fraction of the finger's length.
    """
    def slope(station: float) -> float:
        ratio = station / length_mm
        return tip_travel_mm / length_mm * (3.0 * ratio - 1.5 * ratio ** 2)

    return float(np.degrees(np.arctan(slope(tip_mm)) - np.arctan(slope(base_mm))))


def tip_travel_for_bend_deg(bend_deg: float, base_mm: float, tip_mm: float,
                            length_mm: float) -> float:
    """Inverse of `bend_deg_for_tip_travel`, for reading a threshold in mm.

    Bounded at one finger length. The forward curve does eventually turn over -
    both slope angles saturate toward 90 degrees, so their difference peaks
    around 120% of the length and falls after - and a bisection allowed past
    that peak converges onto the wrong branch. Nothing in that region is
    physical anyway; the beam model gave out long before.
    """
    low, high = 0.0, float(length_mm)
    for _ in range(80):
        middle = (low + high) / 2.0
        if bend_deg_for_tip_travel(middle, base_mm, tip_mm, length_mm) < bend_deg:
            low = middle
        else:
            high = middle
    return float(low)


def load_finger_mesh(hand_config, strict: bool = False):
    """Load the CAD finger declared by the config, or None if none is.

    A missing or unreadable mesh degrades to the text-only display rather than
    stopping the tool: the measurement does not depend on the mesh, only the
    picture of it does.
    """
    if not hand_config.mesh.path:
        return None
    from .fingermesh import FingerMesh
    try:
        mesh = FingerMesh.from_stl(hand_config.mesh.path,
                                   units=hand_config.mesh.units,
                                   face_sign=hand_config.mesh.face_sign,
                                   roll_deg=hand_config.mesh.roll_deg)
    except (OSError, ValueError, IndexError) as exc:
        if strict:
            raise
        print(f"finger mesh not loaded ({exc}); falling back to the plain display",
              file=sys.stderr)
        return None
    if hand_config.mesh.length_mm:
        mesh = mesh.scaled_to_length(hand_config.mesh.length_mm)
    return mesh.with_face_on_spine()


def _rest_root_frame(base_pose, tip_pose, base_y: float):
    """The clamped end's frame, inferred from an unloaded finger.

    At rest the flexure is straight, so the root sits one station-length back
    along the finger from the base tag and shares its orientation. The cuff is
    rigid, so this holds for the whole session once zeroed - which is what turns
    the root into a third, free spline station.
    """
    from .spine import frame_from_marker
    along = _unit(tip_pose.translation - base_pose.translation)
    rotation = frame_from_marker(base_pose.rotation, along)
    return base_pose.translation - rotation[:, 1] * base_y, rotation


def run_grip_live(hand_config, source, intrinsics_path=None, display: bool = True,
                  thresholds: GripThresholds | None = None, zero_frames: int = 30,
                  max_frames: int | None = None, use_mesh: bool = True,
                  dataset_dir: str | None = None):
    """Watch the hand and classify the grip, live.

    Poses are taken in the rigid reference tag's frame rather than the camera's,
    so moving the arm does not look like the fingers bending.
    """
    import time

    from .camera_calib import intrinsics_summary, load_intrinsics
    from .gripview import FrameState, GripView
    from .hud import append_control_bar
    from .labeling import GripLabelDataset
    from .smoothing import MarkerPoseTracker
    from .spine import FingerSpine, build_stations
    from .vision import open_capture, require_cv2, MarkerDetector

    cv2 = require_cv2()
    loaded = load_intrinsics(intrinsics_path or hand_config.camera.intrinsics)
    if loaded is None:
        raise RuntimeError(
            "No camera calibration found. Grip angles are meaningless without it - "
            "run `flexsense screen-calibrate` first.")
    camera_matrix, dist_coeffs, size = loaded
    where = intrinsics_summary(intrinsics_path or hand_config.camera.intrinsics)
    # Printed every run on purpose. A calibration belongs to one camera; reusing
    # another machine's is the failure mode that still looks like it works.
    print(f"calibration  {where}", file=sys.stderr)
    print("             recalibrate with `flexsense screen-calibrate` if this is "
          "not the camera it was measured on", file=sys.stderr)
    thresholds = thresholds or GripThresholds()

    detector = MarkerDetector(hand_config.dictionary)
    classifier = GripClassifier(thresholds)
    tracker = MarkerPoseTracker(hand_config.tag_mm, camera_matrix, dist_coeffs)
    mesh = load_finger_mesh(hand_config) if use_mesh else None
    view = (GripView(hand_config, mesh, thresholds, camera_matrix, dist_coeffs)
            if mesh is not None else None)
    label_dataset = GripLabelDataset(dataset_dir) if dataset_dir else None

    reference_ids = list(hand_config.reference.tag_ids)
    capture = open_capture(source, hand_config.camera.width, hand_config.camera.height)
    learned = 0
    roots: dict[str, tuple] = {}
    zeroed_at: float | None = None
    show_mesh = True
    frames = 0
    calibration_label = "" if not where else where.split(":")[0]
    last = time.monotonic()
    times: list[float] = []
    notice = ""
    notice_until = 0.0

    live_controls = (("Q", "quit"), ("Z", "re-zero"),
                     ("R", "reframe"), ("M", "mesh"))
    label_controls = (("G", "save GOOD"), ("B", "save BAD"),
                      ("U", "undo"), ("Z", "re-zero"),
                      ("R", "reframe"), ("M", "mesh"), ("Q", "quit"))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            now = time.monotonic()
            dt, last = max(now - last, 1e-3), now
            times.append(now)
            del times[:-30]
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            if (frame.shape[1], frame.shape[0]) != tuple(size):
                raise RuntimeError(
                    f"camera is {frame.shape[1]}x{frame.shape[0]} but the calibration "
                    f"is for {size[0]}x{size[1]}")

            by_id, _corners, _ids = detector.detect(frame)
            poses = {tag: tracker.solve(tag, quad, dt) for tag, quad in by_id.items()}
            poses = {tag: pose for tag, pose in poses.items() if pose is not None}
            reference = next((poses[t] for t in reference_ids if poses.get(t)), None)

            state = FrameState(
                tag_pixels=by_id, expected_tags=len(hand_config.all_tag_ids),
                fps=fps, flips_rejected=tracker.flips_rejected,
                calibration=calibration_label)

            if reference is None:
                state.message = "reference tag not visible"
            else:
                to_reference = reference.inverse()
                finger_poses = {}
                for finger in hand_config.fingers:
                    base = poses.get(finger.root_id)
                    tip = poses.get(finger.tip_id)
                    finger_poses[finger.name] = (
                        to_reference.compose(base) if base else None,
                        to_reference.compose(tip) if tip else None,
                    )

                if learned < zero_frames:
                    # Relearn rest continuously through the zero window so a
                    # single noisy frame cannot fix a bad baseline.
                    classifier.reset()
                    roots.clear()
                    for name, (base, tip) in finger_poses.items():
                        if base and tip:
                            classifier.learn_rest(name, base, tip)
                            station = _station_of(hand_config, name, 0)
                            roots[name] = _rest_root_frame(base, tip, station)
                    learned += 1
                    state.zeroing = (learned, zero_frames)
                    if learned >= zero_frames:
                        zeroed_at = now
                else:
                    state.assessment = classifier.assess(finger_poses)
                    state.zeroed_seconds = None if zeroed_at is None else now - zeroed_at

                state.reference = reference
                if show_mesh and mesh is not None:
                    for finger in hand_config.fingers:
                        root = roots.get(finger.name)
                        if root is None:
                            continue
                        base, tip = finger_poses[finger.name]
                        stations = build_stations(
                            root, base, tip,
                            _station_of(hand_config, finger.name, 0),
                            _station_of(hand_config, finger.name, -1))
                        if len(stations) < 2:
                            continue
                        state.spines[finger.name] = FingerSpine.from_stations(
                            stations, mesh.length_mm)

            if view is not None:
                view.record(state.assessment)
                canvas = view.render(frame, state)
            else:
                canvas = frame
                if state.assessment:
                    _draw(cv2, canvas, state.assessment)
                elif state.zeroing:
                    _banner(cv2, canvas, f"ZEROING {state.zeroing[0]}/{state.zeroing[1]}",
                            (30, 210, 240))

            if display:
                controls = label_controls if label_dataset else live_controls
                status = notice if now < notice_until else (
                    label_dataset.summary if label_dataset else "")
                canvas = append_control_bar(canvas, controls, status)
                window = "FlexSense grip labeling" if label_dataset else "FlexSense grip"
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("z"):
                    classifier.reset()
                    tracker.reset()
                    roots.clear()
                    learned = 0
                    zeroed_at = None
                if key == ord("r") and view is not None:
                    view.reframe()
                if key == ord("m"):
                    show_mesh = not show_mesh
                if label_dataset is not None and key in (ord("g"), ord("b")):
                    if state.assessment is None:
                        notice = "not saved — wait for zeroing and visible tags"
                    else:
                        label = "GOOD" if key == ord("g") else "BAD"
                        label_dataset.save(
                            frame, label, state.assessment, calibration_label)
                        notice = f"saved {label}  ·  {label_dataset.summary}"
                    notice_until = now + 2.0
                if label_dataset is not None and key == ord("u"):
                    removed = label_dataset.undo()
                    notice = (f"undid {removed['label']}  ·  {label_dataset.summary}"
                              if removed else "nothing to undo")
                    notice_until = now + 2.0
            if max_frames is not None and frames >= max_frames:
                break
        return 0
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _station_of(hand_config, name: str, index: int) -> float:
    for finger in hand_config.fingers:
        if finger.name == name:
            return float(finger.stations_mm[index])
    raise KeyError(name)


VERDICT_COLORS = {
    "good": (60, 210, 60),
    "marginal": (40, 190, 240),
    "bad": (40, 60, 235),
    "no-contact": (170, 170, 170),
    "unknown": (120, 120, 120),
}
STATE_COLORS = {
    GripState.WRAPPING.value: (60, 210, 60),
    GripState.BACKBENDING.value: (40, 60, 235),
    GripState.NEUTRAL.value: (180, 180, 180),
    GripState.UNKNOWN.value: (110, 110, 110),
}


def _banner(cv2, frame, text, colour) -> None:
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, colour, 2)


def _draw(cv2, frame, assessment: GripAssessment) -> None:
    colour = VERDICT_COLORS.get(assessment.verdict, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), colour, 10)
    cv2.putText(frame, assessment.verdict.upper(), (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
    cv2.putText(frame, assessment.reason, (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    y = 100
    for name, finger in assessment.fingers.items():
        bend = "--" if finger.signed_bend_deg is None else f"{finger.signed_bend_deg:+6.1f}"
        cv2.putText(frame, f"{name:9} {finger.state:12} bend {bend} deg",
                    (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    STATE_COLORS.get(finger.state, (200, 200, 200)), 2)
        y += 26
