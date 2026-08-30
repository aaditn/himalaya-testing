#!/usr/bin/env python3
"""Native-MuJoCo stability audit for the prone 30-degree crawl reset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from himalaya.tasks.four_contact_env_cfg import make_four_contact_env


def _support_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    floor = model.geom("floor").id
    support_geoms = {
        "left_foot": model.geom("left_foot").id,
        "right_foot": model.geom("right_foot").id,
        "left_hand": model.geom("left_hand_collision").id,
        "right_hand": model.geom("right_hand_collision").id,
    }
    contacts = {name: False for name in support_geoms}
    for index in range(data.ncon):
        geom1, geom2 = data.contact[index].geom
        for name, geom in support_geoms.items():
            contacts[name] |= {int(geom1), int(geom2)} == {floor, geom}
    return contacts


def audit(seconds: float, preview: Path | None = None) -> dict[str, object]:
    env = make_four_contact_env(30.0, noise_level=0.0, impl="jax")
    model = env.mj_model
    data = mujoco.MjData(model)
    qpos = np.asarray(env._init_q).copy()
    tangent = np.asarray(env.ramp_tangent)
    normal = np.asarray(env.ramp_normal)
    ramp_quat = np.asarray(env._ramp_quat)
    nominal_quat = np.asarray(env._nominal_root_quat)
    root_quat = np.empty(4)
    mujoco.mju_mulQuat(root_quat, ramp_quat, nominal_quat)
    qpos[:3] = qpos[2] * normal
    qpos[3:7] = root_quat
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(model, data)

    renderer = None
    camera = None
    if preview is not None:
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 540
        renderer = mujoco.Renderer(model, height=540, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = data.qpos[:3]
        camera.distance = 2.4
        camera.azimuth = 145
        camera.elevation = -16
        renderer.update_scene(data, camera=camera)
        preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(preview)

    initial_position = data.qpos[:3].copy()
    initial_contacts = _support_contacts(model, data)
    max_drift = 0.0
    min_height = math.inf
    min_supports = 4
    samples = round(seconds / model.opt.timestep)
    for _ in range(samples):
        mujoco.mj_step(model, data)
        displacement = data.qpos[:3] - initial_position
        max_drift = max(max_drift, abs(float(displacement @ tangent)))
        min_height = min(min_height, float(data.qpos[:3] @ normal))
        min_supports = min(min_supports, sum(_support_contacts(model, data).values()))
    final_contacts = _support_contacts(model, data)
    if renderer is not None and camera is not None:
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(
            preview.with_name(preview.stem + "_final" + preview.suffix)
        )
        renderer.close()
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    passed = (
        finite
        and all(initial_contacts.values())
        and sum(final_contacts.values()) >= 3
        and min_height >= 0.25
        and max_drift <= 0.25
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "seconds": seconds,
        "slope_degrees": 30.0,
        "pose": "four_contact_crawl",
        "initial_contacts": initial_contacts,
        "final_contacts": final_contacts,
        "minimum_support_contacts": min_supports,
        "minimum_root_normal_height_m": min_height,
        "maximum_uphill_drift_m": max_drift,
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output")
    parser.add_argument("--preview")
    args = parser.parse_args()
    report = audit(args.seconds, Path(args.preview) if args.preview else None)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
