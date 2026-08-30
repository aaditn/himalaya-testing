"""Synthetic compliant hand: render tagged fingers at known deflections.

Lets the estimator be written and validated against ground truth before the
printed hand exists. The renderer produces ordinary camera frames, so everything
downstream - detection, lens correction, pose - runs exactly as it will on real
hardware; only the image source changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .vision import require_cv2


def _rotate_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of `vector` about a unit `axis`."""
    cos, sin = np.cos(angle), np.sin(angle)
    return (vector * cos + np.cross(axis, vector) * sin
            + axis * float(axis @ vector) * (1.0 - cos))


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("zero-length direction")
    return vector / norm


@dataclass(frozen=True)
class Finger:
    """One flexure, tagged at two stations along its length.

    `bend_dir` is the direction the tip travels under load, in hub coordinates.
    Point it along the camera axis to model out-of-plane bending, or across the
    image to model in-plane bending; the estimator should not care which.
    """

    name: str
    base: tuple[float, float, float]
    axis: tuple[float, float, float]
    bend_dir: tuple[float, float, float]
    length: float
    tag_ids: tuple[int, ...]
    tag_stations: tuple[float, ...]
    tag_mm: float = 20.0
    # Which way the tagged face points at rest. Fixed by the print, independent
    # of which way the finger bends.
    face: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def station_pose(self, station: float, tip_deflection: float):
        """Position and orientation of one station on the deflected beam.

        Uses the small-deflection cantilever shape for an end load,
        y(s) = d * (3(s/L)^2 - (s/L)^3) / 2, which is exact for a uniform beam
        and close enough for a tapered one to exercise the estimator.
        """
        axis = _unit(self.axis)
        bend = _unit(self.bend_dir)
        # Bending must be perpendicular to the beam. If the requested direction
        # is parallel to this finger's axis it says nothing, so fall back to a
        # well-defined perpendicular rather than dividing by zero.
        perpendicular = bend - axis * (bend @ axis)
        if np.linalg.norm(perpendicular) < 1e-6:
            fallback = np.array([0.0, 0.0, -1.0])
            if abs(fallback @ axis) > 0.99:
                fallback = np.array([1.0, 0.0, 0.0])
            perpendicular = fallback - axis * (fallback @ axis)
        bend = _unit(perpendicular)
        ratio = np.clip(station / self.length, 0.0, 1.0)
        offset = tip_deflection * (3.0 * ratio ** 2 - ratio ** 3) / 2.0
        slope = tip_deflection * (6.0 * ratio - 3.0 * ratio ** 2) / (2.0 * self.length)
        centre = np.asarray(self.base, float) + axis * station + bend * offset
        # The tag tilts with the local beam slope; this is the out-of-plane
        # rotation a planar homography cannot represent. The whole local frame
        # rotates rigidly with the beam, so the face turns by the same angle.
        tangent = _unit(axis + bend * slope)
        face = _unit(np.asarray(self.face, float))
        face = _unit(face - axis * (face @ axis))
        pivot = np.cross(axis, bend)
        pivot_norm = np.linalg.norm(pivot)
        if pivot_norm > 1e-9:
            face = _rotate_about(face, pivot / pivot_norm, np.arctan(slope))
        return centre, tangent, _unit(face - tangent * (face @ tangent))


@dataclass(frozen=True)
class ReferenceTag:
    """A tag on rigid structure. Never deflects, defines the measurement frame."""

    tag_id: int
    centre: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tag_mm: float = 20.0


@dataclass
class HandRig:
    fingers: list[Finger]
    references: list[ReferenceTag] = field(default_factory=list)
    dictionary: str = "DICT_4X4_50"

    def tag_corners(self, deflections: dict[str, float]) -> dict[int, np.ndarray]:
        """3D corners of every tag, ordered top-left, top-right, bottom-right, bottom-left."""
        out: dict[int, np.ndarray] = {}
        for ref in self.references:
            up = _unit(ref.up)
            normal = _unit(ref.normal)
            right = _unit(np.cross(up, normal))
            out[ref.tag_id] = _quad(np.asarray(ref.centre, float), right, up, ref.tag_mm)
        for finger in self.fingers:
            deflection = float(deflections.get(finger.name, 0.0))
            for tag_id, station in zip(finger.tag_ids, finger.tag_stations):
                centre, tangent, normal = finger.station_pose(station, deflection)
                right = _unit(np.cross(tangent, normal))
                out[tag_id] = _quad(centre, right, tangent, finger.tag_mm)
        return out


def _quad(centre: np.ndarray, right: np.ndarray, up: np.ndarray, size: float) -> np.ndarray:
    half = size / 2.0
    return np.array([
        centre - right * half + up * half,   # top-left
        centre + right * half + up * half,   # top-right
        centre + right * half - up * half,   # bottom-right
        centre - right * half - up * half,   # bottom-left
    ], dtype=float)


def render(rig: HandRig, deflections: dict[str, float], camera_matrix: np.ndarray,
           dist_coeffs: np.ndarray, image_size: tuple[int, int],
           camera_pose: tuple[np.ndarray, np.ndarray] | None = None,
           background: int = 245) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Render the tags and return (frame, true 3D corners in hub coordinates)."""
    cv2 = require_cv2()
    width, height = image_size
    frame = np.full((height, width, 3), background, np.uint8)
    corners_3d = rig.tag_corners(deflections)
    rvec, tvec = camera_pose if camera_pose is not None else (
        np.zeros((3, 1)), np.array([[0.0], [0.0], [180.0]]))

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, rig.dictionary))
    rotation, _ = cv2.Rodrigues(rvec)

    # Painter's algorithm: far tags first, so a nearer finger occludes correctly.
    depths = {tag: float((rotation @ pts.mean(axis=0) + tvec.ravel())[2])
              for tag, pts in corners_3d.items()}
    for tag_id in sorted(corners_3d, key=lambda t: -depths[t]):
        if depths[tag_id] <= 1.0:
            continue
        projected, _ = cv2.projectPoints(
            corners_3d[tag_id].astype(np.float32), rvec, tvec, camera_matrix, dist_coeffs)
        dst = projected.reshape(4, 2).astype(np.float32)
        if not np.all(np.isfinite(dst)):
            continue
        bitmap = cv2.aruco.generateImageMarker(dictionary, int(tag_id), 400)
        # Pad so the marker keeps the white quiet zone detection needs.
        padded = cv2.copyMakeBorder(bitmap, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        side = padded.shape[0]
        scale = side / 400.0
        span = (scale - 1.0) / 2.0
        # Expand the destination quad by the same fraction the padding added.
        centre = dst.mean(axis=0)
        outer = (centre + (dst - centre) * scale).astype(np.float32)
        src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, outer)
        warped = cv2.warpPerspective(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), matrix,
                                     (width, height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), matrix,
                                   (width, height))
        frame[mask > 128] = warped[mask > 128]
        _ = span
    return frame, corners_3d


def default_hand(tag_mm: float = 20.0, reference_ids: tuple[int, ...] = ()) -> HandRig:
    """Four fingers, two tags each, matching the CAD layout.

    Three fingers stand along +y; one lies along -x. Tag ids follow the render:
    0/1 on the horizontal member, then 2/3, 4/5, 6/7 root-to-tip on the uprights.
    Reference tags, when requested, sit on the hub and never move.
    """
    length = 110.0
    stations = (38.0, 78.0)
    out_of_plane = (0.0, 0.0, -1.0)
    # Fingers extend -y: the camera's +y runs down the image, so this puts the
    # uprights at the top of the frame exactly as they appear in the CAD view.
    fingers = [
        Finger("horizontal", (-30.0, 14.0, 0.0), (-1.0, 0.0, 0.0), out_of_plane,
               length, (0, 1), (66.0, 30.0), tag_mm),
        Finger("left", (-30.0, 0.0, 0.0), (0.0, -1.0, 0.0), out_of_plane,
               length, (2, 3), (78.0, 38.0), tag_mm),
        Finger("middle", (0.0, -6.0, 0.0), (0.0, -1.0, 0.0), out_of_plane,
               length, (4, 5), (78.0, 38.0), tag_mm),
        Finger("right", (30.0, 0.0, 0.0), (0.0, -1.0, 0.0), out_of_plane,
               length, (6, 7), (78.0, 38.0), tag_mm),
    ]
    _ = stations
    # Reference tag 8 sits on the domed hub, protruding toward the camera.
    # Extra reference ids, if given, spread across the hub either side of it.
    references = []
    for index, tag_id in enumerate(reference_ids):
        x = 0.0 if len(reference_ids) == 1 else float(
            np.linspace(-15.0, 15.0, len(reference_ids))[index] * 1.0)
        # Staggered in y so several fit across the hub without overlapping the
        # finger columns at x = -30, 0, +30.
        y = 26.0 if index % 2 == 0 else 48.0
        references.append(ReferenceTag(tag_id, (x, y, -12.0), tag_mm=tag_mm))
    return HandRig(fingers=fingers, references=references)


def with_bend_direction(rig: HandRig, direction) -> HandRig:
    """Return the rig with every finger bending along `direction`."""
    return HandRig(
        fingers=[replace(f, bend_dir=tuple(float(v) for v in direction)) for f in rig.fingers],
        references=list(rig.references),
        dictionary=rig.dictionary,
    )
