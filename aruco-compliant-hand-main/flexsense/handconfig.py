"""One declaration of the hand, shared by the simulator and the live camera.

Everything that describes the physical build - which ArUco id is printed at
which station on which finger, how long the fingers are, where the camera sits -
lives in a single YAML file. The simulator and the live tracker both consume it,
so a layout explored in simulation is the same layout that runs on hardware; only
the image source changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

STATION_ORDER = ("base", "middle", "tip")


def _vec(value, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must be three numbers, got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class FingerDef:
    """One finger and the tags printed along it.

    `stations_mm` are arc distances from the finger's root, matching the order of
    `tag_ids`. Naming them base/middle/tip is a convention, not a requirement -
    two tags or four both work; the last is always treated as the tip.
    """

    name: str
    tag_ids: tuple[int, ...]
    stations_mm: tuple[float, ...]
    base_mm: tuple[float, float, float]
    axis: tuple[float, float, float]
    length_mm: float
    face: tuple[float, float, float] = (0.0, 0.0, -1.0)
    bend_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)

    @property
    def tip_id(self) -> int:
        return self.tag_ids[-1]

    @property
    def root_id(self) -> int:
        return self.tag_ids[0]

    def station_name(self, index: int) -> str:
        if len(self.tag_ids) == len(STATION_ORDER):
            return STATION_ORDER[index]
        if index == 0:
            return "base"
        if index == len(self.tag_ids) - 1:
            return "tip"
        return f"mid{index}"


@dataclass(frozen=True)
class ReferenceDef:
    """Tags on rigid structure. These define the frame everything is measured in."""

    tag_ids: tuple[int, ...]
    positions_mm: tuple[tuple[float, float, float], ...] = ()
    face: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class CameraDef:
    """Where the camera sits, in hub coordinates.

    Expressed as eye/target/up rather than a rotation matrix so it can be
    reasoned about from CAD without touching rotation conventions.
    """

    position_mm: tuple[float, float, float] = (0.0, 0.0, -195.0)
    look_at_mm: tuple[float, float, float] = (0.0, -40.0, 0.0)
    up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    width: int = 1280
    height: int = 720
    intrinsics: str | None = "calibration/camera_intrinsics.json"

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rvec, tvec) mapping hub coordinates into camera coordinates."""
        import cv2

        eye = np.asarray(self.position_mm, float)
        forward = np.asarray(self.look_at_mm, float) - eye
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError("camera position and look_at are the same point")
        z_axis = forward / norm
        # OpenCV cameras have +y pointing down the image, so the requested "up"
        # direction becomes -y.
        down = -np.asarray(self.up, float)
        down = down - z_axis * float(down @ z_axis)
        if np.linalg.norm(down) < 1e-9:
            raise ValueError("camera `up` is parallel to the viewing direction")
        y_axis = down / np.linalg.norm(down)
        x_axis = np.cross(y_axis, z_axis)
        rotation = np.vstack([x_axis, y_axis, z_axis])
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = (-rotation @ eye).reshape(3, 1)
        return rvec, tvec

    @property
    def distance_to_target_mm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.look_at_mm, float)
                                    - np.asarray(self.position_mm, float)))


@dataclass(frozen=True)
class MeshDef:
    """Where the finger's CAD lives and how to read it.

    `length_mm` rescales the mesh to the finger's true root-to-tip length. It
    exists because a printed part rarely matches its CAD to the millimetre and
    because STL carries no units, so the file alone cannot settle the question.
    """

    path: str | None = None
    units: str = "auto"
    face_sign: int = 1
    roll_deg: float = 0.0
    length_mm: float | None = None


@dataclass(frozen=True)
class HandConfig:
    name: str = "hand"
    dictionary: str = "DICT_4X4_50"
    tag_mm: float = 20.0
    fingers: tuple[FingerDef, ...] = ()
    reference: ReferenceDef = field(default_factory=lambda: ReferenceDef(()))
    camera: CameraDef = field(default_factory=CameraDef)
    mesh: MeshDef = field(default_factory=MeshDef)
    press_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zero_samples: int = 30
    max_missing_frames: int = 8
    grip: dict[str, float] = field(default_factory=dict)

    # ---- derived views ------------------------------------------------

    @property
    def finger_tag_ids(self) -> list[int]:
        return [t for finger in self.fingers for t in finger.tag_ids]

    @property
    def all_tag_ids(self) -> list[int]:
        return list(self.reference.tag_ids) + self.finger_tag_ids

    def owner_of(self, tag_id: int) -> str:
        if tag_id in self.reference.tag_ids:
            return "reference"
        for finger in self.fingers:
            if tag_id in finger.tag_ids:
                index = finger.tag_ids.index(tag_id)
                return f"{finger.name}/{finger.station_name(index)}"
        return "unknown"

    # ---- validation ---------------------------------------------------

    def problems(self) -> list[str]:
        """Hard errors that make the configuration unusable."""
        issues: list[str] = []
        seen: dict[int, str] = {}
        for tag_id in self.all_tag_ids:
            owner = self.owner_of(tag_id)
            if tag_id in seen:
                issues.append(f"tag id {tag_id} used twice: {seen[tag_id]} and {owner}")
            seen[tag_id] = owner
        if not self.reference.tag_ids:
            issues.append("no reference tags: every measurement needs a rigid frame")
        if not self.fingers:
            issues.append("no fingers defined")
        for finger in self.fingers:
            if len(finger.tag_ids) != len(finger.stations_mm):
                issues.append(
                    f"{finger.name}: {len(finger.tag_ids)} ids but "
                    f"{len(finger.stations_mm)} stations")
            if len(finger.tag_ids) < 1:
                issues.append(f"{finger.name}: needs at least one tag")
            for station in finger.stations_mm:
                if station > finger.length_mm:
                    issues.append(
                        f"{finger.name}: station {station}mm is past the "
                        f"{finger.length_mm}mm finger")
            if list(finger.stations_mm) != sorted(finger.stations_mm):
                issues.append(f"{finger.name}: stations must run root to tip")
        if self.reference.positions_mm and \
                len(self.reference.positions_mm) != len(self.reference.tag_ids):
            issues.append("reference positions given but count does not match ids")
        return issues

    def warnings(self) -> list[str]:
        """Things that will work but are likely to bite."""
        notes: list[str] = []
        if len(self.reference.tag_ids) == 1:
            notes.append(
                "only one reference tag: if it is occluded or glared every finger "
                "reads invalid for that frame. Two or three give graceful degradation")
        for finger in self.fingers:
            if len(finger.tag_ids) < 2:
                notes.append(
                    f"{finger.name}: one tag cannot show bend shape, and the finger "
                    "axis has to be guessed rather than measured")
        try:
            import cv2
            dictionary = getattr(cv2.aruco, self.dictionary, None)
            if dictionary is not None:
                count = cv2.aruco.getPredefinedDictionary(dictionary).bytesList.shape[0]
                over = [t for t in self.all_tag_ids if t >= count]
                if over:
                    notes.append(f"{self.dictionary} only holds {count} ids; {over} are out of range")
        except Exception:
            pass
        for finger in self.fingers:
            if not finger.stations_mm:
                continue
            covered = max(finger.stations_mm) / finger.length_mm
            if covered < 0.75:
                notes.append(
                    f"{finger.name}: the outermost tag sits at "
                    f"{max(finger.stations_mm):.0f}mm of {finger.length_mm:.0f}mm, so "
                    f"{100 * (1 - covered):.0f}% of the finger is extrapolated rather "
                    "than measured. Moving it toward the tip cuts that error directly")
        return notes


# ---- YAML round trip ---------------------------------------------------


def _finger_from_dict(data: dict[str, Any], default_tag_mm: float) -> FingerDef:
    ids = data.get("tag_ids")
    if isinstance(data.get("ids"), dict):
        named = data["ids"]
        ids = [named[key] for key in STATION_ORDER if key in named]
    if not ids:
        raise ValueError(f"finger {data.get('name')!r} has no tag ids")
    stations = data.get("stations_mm")
    if stations is None:
        length = float(data.get("length_mm", 110.0))
        # Spread evenly over the outer 80% of the finger when unspecified.
        stations = list(np.linspace(0.2 * length, 0.9 * length, len(ids)))
    _ = default_tag_mm
    return FingerDef(
        name=str(data["name"]),
        tag_ids=tuple(int(v) for v in ids),
        stations_mm=tuple(float(v) for v in stations),
        base_mm=_vec(data.get("base_mm", (0.0, 0.0, 0.0)), "base_mm"),
        axis=_vec(data.get("axis", (0.0, -1.0, 0.0)), "axis"),
        length_mm=float(data.get("length_mm", 110.0)),
        face=_vec(data.get("face", (0.0, 0.0, -1.0)), "face"),
        bend_dir=_vec(data.get("bend_dir", (0.0, 0.0, -1.0)), "bend_dir"),
    )


def _resolve_mesh_path(declared: str | None, config_path: Path) -> str | None:
    """Paths inside a config file are relative to that file's project, not cwd."""
    if not declared:
        return declared
    from .paths import resolve

    given = Path(declared)
    tries = [given] if given.is_absolute() else [
        config_path.parent / given, config_path.parent.parent / given, resolve(given)]
    for candidate in tries:
        if candidate.exists():
            return str(candidate)
    return declared


def load_hand(path: str | Path) -> HandConfig:
    from .paths import resolve

    path = resolve(path)
    raw = yaml.safe_load(Path(path).read_text()) or {}
    hand = raw.get("hand", raw)
    tag_mm = float(hand.get("tag_mm", 20.0))
    fingers = tuple(_finger_from_dict(f, tag_mm) for f in hand.get("fingers", []))
    ref_raw = hand.get("reference", {}) or {}
    reference = ReferenceDef(
        tag_ids=tuple(int(v) for v in ref_raw.get("ids", ())),
        positions_mm=tuple(_vec(p, "reference position")
                           for p in ref_raw.get("positions_mm", ()) or ()),
        face=_vec(ref_raw.get("face", (0.0, 0.0, -1.0)), "reference face"),
    )
    cam_raw = raw.get("camera", hand.get("camera", {})) or {}
    camera = CameraDef(
        position_mm=_vec(cam_raw.get("position_mm", (0.0, 0.0, -195.0)), "camera position"),
        look_at_mm=_vec(cam_raw.get("look_at_mm", (0.0, -40.0, 0.0)), "camera look_at"),
        up=_vec(cam_raw.get("up", (0.0, -1.0, 0.0)), "camera up"),
        width=int(cam_raw.get("width", 1280)),
        height=int(cam_raw.get("height", 720)),
        intrinsics=cam_raw.get("intrinsics", "calibration/camera_intrinsics.json"),
    )
    mesh_raw = raw.get("mesh", hand.get("mesh", {})) or {}
    mesh = MeshDef(
        path=_resolve_mesh_path(mesh_raw.get("path"), Path(path)),
        units=str(mesh_raw.get("units", "auto")),
        face_sign=int(mesh_raw.get("face_sign", 1)),
        roll_deg=float(mesh_raw.get("roll_deg", 0.0)),
        length_mm=(float(mesh_raw["length_mm"])
                   if mesh_raw.get("length_mm") is not None else None),
    )
    return HandConfig(
        name=str(hand.get("name", "hand")),
        dictionary=str(hand.get("dictionary", "DICT_4X4_50")),
        tag_mm=tag_mm,
        fingers=fingers,
        reference=reference,
        camera=camera,
        mesh=mesh,
        press_direction=_vec(hand.get("press_direction", (0.0, 0.0, 1.0)), "press_direction"),
        zero_samples=int(hand.get("zero_samples", 30)),
        max_missing_frames=int(hand.get("max_missing_frames", 8)),
        grip={k: float(v) for k, v in (raw.get("grip", hand.get("grip", {})) or {}).items()},
    )


def dump_hand(config: HandConfig) -> str:
    data = {
        "hand": {
            "name": config.name,
            "dictionary": config.dictionary,
            "tag_mm": config.tag_mm,
            "press_direction": list(config.press_direction),
            "zero_samples": config.zero_samples,
            "max_missing_frames": config.max_missing_frames,
            "reference": {
                "ids": list(config.reference.tag_ids),
                "face": list(config.reference.face),
                **({"positions_mm": [list(p) for p in config.reference.positions_mm]}
                   if config.reference.positions_mm else {}),
            },
            "fingers": [
                {
                    "name": f.name,
                    "tag_ids": list(f.tag_ids),
                    "stations_mm": list(f.stations_mm),
                    "base_mm": list(f.base_mm),
                    "axis": list(f.axis),
                    "length_mm": f.length_mm,
                    "face": list(f.face),
                    "bend_dir": list(f.bend_dir),
                }
                for f in config.fingers
            ],
        },
        "mesh": {
            "path": config.mesh.path,
            "units": config.mesh.units,
            "face_sign": config.mesh.face_sign,
            "roll_deg": config.mesh.roll_deg,
            **({"length_mm": config.mesh.length_mm}
               if config.mesh.length_mm is not None else {}),
        },
        "camera": {
            "position_mm": list(config.camera.position_mm),
            "look_at_mm": list(config.camera.look_at_mm),
            "up": list(config.camera.up),
            "width": config.camera.width,
            "height": config.camera.height,
            "intrinsics": config.camera.intrinsics,
        },
    }
    return yaml.safe_dump(data, sort_keys=False)


# ---- adapters ----------------------------------------------------------
# These are the whole point of the file: the same declaration builds the
# simulated hand, the estimator that reads a real one, and the sheet you print.


def to_sim_rig(config: HandConfig):
    """Build a renderable rig from the declaration."""
    from .simrig import Finger, HandRig, ReferenceTag

    fingers = [
        Finger(
            name=f.name,
            base=f.base_mm,
            axis=f.axis,
            bend_dir=f.bend_dir,
            length=f.length_mm,
            tag_ids=f.tag_ids,
            tag_stations=f.stations_mm,
            tag_mm=config.tag_mm,
            face=f.face,
        )
        for f in config.fingers
    ]
    if config.reference.positions_mm:
        positions = config.reference.positions_mm
    else:
        # Spread across the hub, staggered so they do not sit on the finger columns.
        count = max(len(config.reference.tag_ids), 1)
        # Spacing must exceed the tag width or the tags overlap each other and
        # the ones behind simply never decode.
        pitch = config.tag_mm * 1.3
        span = pitch * (count - 1) / 2.0
        xs = np.linspace(-span, span, count) if count > 1 else np.array([0.0])
        # Kept close to the finger roots: every millimetre of extra vertical
        # span forces the camera back, which costs pixels on every tag.
        positions = tuple((float(x), 18.0, -12.0) for x in xs)
    references = [
        # `face` already points the way the printed tag looks, i.e. toward the
        # camera. Negating it here pointed every reference tag backwards.
        ReferenceTag(tag_id, position, normal=config.reference.face,
                     tag_mm=config.tag_mm)
        for tag_id, position in zip(config.reference.tag_ids, positions)
    ]
    return HandRig(fingers=fingers, references=references, dictionary=config.dictionary)


def to_estimator_config(config: HandConfig):
    """Build the 6-DoF estimator's configuration from the declaration."""
    from .estimator3d import FingerSpec3D, Spatial3DConfig

    return Spatial3DConfig(
        reference_ids=config.reference.tag_ids,
        fingers=tuple(FingerSpec3D(name=f.name, tag_ids=f.tag_ids)
                      for f in config.fingers),
        tag_mm=config.tag_mm,
        zero_samples=config.zero_samples,
        max_missing_frames=config.max_missing_frames,
        press_direction=config.press_direction,
    )


def tag_manifest(config: HandConfig) -> list[dict[str, Any]]:
    """Flat list of every tag and where it belongs. Drives the printable sheet."""
    rows: list[dict[str, Any]] = []
    for tag_id in config.reference.tag_ids:
        rows.append({"id": tag_id, "part": "REFERENCE", "station": "rigid hub",
                     "note": "must not flex"})
    for finger in config.fingers:
        for index, tag_id in enumerate(finger.tag_ids):
            rows.append({
                "id": tag_id,
                "part": finger.name,
                "station": finger.station_name(index),
                "note": f"{finger.stations_mm[index]:.0f} mm from root",
            })
    return rows


def to_grip_thresholds(config: HandConfig):
    """Grip thresholds from the declaration, falling back to defaults."""
    from .grip import GripThresholds

    return GripThresholds(**{k: v for k, v in config.grip.items()
                             if k in GripThresholds.__dataclass_fields__})
