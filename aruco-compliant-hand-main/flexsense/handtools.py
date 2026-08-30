"""Inspect and rehearse a hand declaration before the hardware exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .camera_calib import load_intrinsics
from .handconfig import HandConfig, tag_manifest, to_estimator_config, to_sim_rig
from .vision import MarkerDetector, require_cv2


def _intrinsics(config: HandConfig):
    """Real calibration when present, otherwise a plausible stand-in."""
    if config.camera.intrinsics:
        loaded = load_intrinsics(config.camera.intrinsics)
        if loaded is not None:
            camera_matrix, dist, size = loaded
            if tuple(size) == (config.camera.width, config.camera.height):
                return camera_matrix, dist, f"calibration {config.camera.intrinsics}"
    width, height = config.camera.width, config.camera.height
    guess = np.array([[width * 0.57, 0.0, width / 2.0],
                      [0.0, width * 0.57, height / 2.0],
                      [0.0, 0.0, 1.0]])
    return guess, np.zeros(5), "ESTIMATED intrinsics (no matching calibration)"


def check(config: HandConfig) -> dict[str, Any]:
    """Validate the declaration and predict how it will behave optically."""
    camera_matrix, dist, source = _intrinsics(config)
    rig = to_sim_rig(config)
    rvec, tvec = config.camera.pose()
    size = (config.camera.width, config.camera.height)
    frame, corners_3d = _render(rig, {}, camera_matrix, dist, size, (rvec, tvec))
    detector = MarkerDetector(config.dictionary)
    seen = detector.detect(frame)[0]

    cv2 = require_cv2()
    rotation, _ = cv2.Rodrigues(rvec)
    rows = []
    for row in tag_manifest(config):
        tag_id = row["id"]
        entry = dict(row)
        if tag_id in corners_3d:
            depth = float((rotation @ corners_3d[tag_id].mean(0) + tvec.ravel())[2])
            entry["distance_mm"] = round(depth, 1)
            entry["pixels"] = round(float(camera_matrix[0, 0]) * config.tag_mm / depth, 1)
        else:
            entry["distance_mm"] = None
            entry["pixels"] = None
        entry["visible"] = tag_id in seen
        rows.append(entry)

    notes = list(config.warnings())
    invisible = [r["id"] for r in rows if not r["visible"]]
    if invisible:
        notes.append(f"not visible from this camera position: {invisible}")
    small = [r["id"] for r in rows if r["pixels"] is not None and r["pixels"] < 50]
    if small:
        notes.append(f"below 50 px and unreliable: {small}")
    marginal = [r["id"] for r in rows
                if r["pixels"] is not None and 50 <= r["pixels"] < 90]
    if marginal:
        notes.append(f"between 50 and 90 px, usable but not good: {marginal}")
    return {
        "name": config.name,
        "intrinsics": source,
        "camera_distance_mm": round(config.camera.distance_to_target_mm, 1),
        "tags_declared": len(rows),
        "tags_visible": sum(1 for r in rows if r["visible"]),
        "problems": config.problems(),
        "warnings": notes,
        "tags": rows,
    }


def _render(rig, deflections, camera_matrix, dist, size, pose):
    from .simrig import render
    return render(rig, deflections, camera_matrix, dist, size, pose)


def preview(config: HandConfig, deflections: dict[str, float],
            output: str | Path | None = None, annotate: bool = True):
    """Render what this camera position would see, with detection overlaid."""
    cv2 = require_cv2()
    camera_matrix, dist, source = _intrinsics(config)
    rig = to_sim_rig(config)
    frame, _ = _render(rig, deflections, camera_matrix, dist,
                       (config.camera.width, config.camera.height), config.camera.pose())
    detector = MarkerDetector(config.dictionary)
    by_id, corners, ids = detector.detect(frame)
    if annotate:
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        missing = [t for t in config.all_tag_ids if t not in by_id]
        cv2.putText(frame, f"{len(by_id)}/{len(config.all_tag_ids)} tags   "
                    f"{config.camera.distance_to_target_mm:.0f} mm   {source}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (40, 200, 40) if not missing else (40, 160, 240), 2)
        if missing:
            cv2.putText(frame, f"MISSING {missing}", (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 80, 240), 2)
        y = 86
        for finger in config.fingers:
            applied = deflections.get(finger.name, 0.0)
            got = sum(1 for t in finger.tag_ids if t in by_id)
            cv2.putText(frame, f"{finger.name:9} {got}/{len(finger.tag_ids)} tags"
                        f"   deflect {applied:5.1f} mm", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
            y += 22
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)
    return frame, by_id


def rehearse(config: HandConfig, deflections: dict[str, float],
             frames: int = 40) -> dict[str, Any]:
    """Run the full pipeline in simulation and report measured vs applied.

    This is the dress rehearsal: same config, same estimator, same detector that
    will run on hardware, with ground truth available so the numbers can be
    checked rather than trusted.
    """
    from .estimator3d import SpatialDeformationEstimator

    camera_matrix, dist, source = _intrinsics(config)
    rig = to_sim_rig(config)
    pose = config.camera.pose()
    size = (config.camera.width, config.camera.height)
    detector = MarkerDetector(config.dictionary)
    estimator_config = to_estimator_config(config)
    estimator = SpatialDeformationEstimator(estimator_config, camera_matrix, dist)

    rest = rig.tag_corners({})
    for _ in range(min(frames, estimator_config.zero_samples + 2)):
        frame, _ = _render(rig, {}, camera_matrix, dist, size, pose)
        estimator.observe(detector.detect(frame)[0])
    frame, _ = _render(rig, deflections, camera_matrix, dist, size, pose)
    reading = estimator.observe(detector.detect(frame)[0])
    now = rig.tag_corners(deflections)

    results = []
    for finger in config.fingers:
        measured = reading.fingers.get(finger.name)
        truth = float(np.linalg.norm(now[finger.tip_id].mean(0)
                                     - rest[finger.tip_id].mean(0)))
        results.append({
            "finger": finger.name,
            "applied_tip_mm": deflections.get(finger.name, 0.0),
            "true_tag_motion_mm": round(truth, 3),
            "measured_mm": (None if measured is None or not measured.valid
                            else round(abs(measured.normal_mm), 3)),
            "error_mm": (None if measured is None or not measured.valid
                         else round(abs(measured.normal_mm) - truth, 3)),
            "valid": bool(measured and measured.valid),
        })
    return {
        "intrinsics": source,
        "status": reading.status,
        "reference_visible": reading.reference_visible,
        "incline_deg": (None if reading.incline_deg is None
                        else round(reading.incline_deg, 2)),
        "fingers": results,
    }
