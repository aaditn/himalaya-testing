#!/usr/bin/env python3
"""Solve a symmetric four-contact G1 pose on the 30-degree starter patch."""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from himalaya.tasks.four_contact_env_cfg import make_four_contact_env


JOINTS = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow", "wrist_pitch")


def main() -> None:
    env = make_four_contact_env(30.0, noise_level=0.0, impl="jax")
    model = env.mj_model
    data = mujoco.MjData(model)
    normal = np.asarray(env.ramp_normal)
    qpos = np.asarray(env._init_q).copy()
    root_quat = np.empty(4)
    mujoco.mju_mulQuat(
        root_quat, np.asarray(env._ramp_quat), np.asarray(env._nominal_root_quat)
    )
    qpos[:3] = qpos[2] * normal
    qpos[3:7] = root_quat
    indices = {
        joint: [
            int(model.jnt_qposadr[model.joint(f"{side}_{joint}_joint").id])
            for side in ("left", "right")
        ]
        for joint in JOINTS
    }
    initial = np.array([
        float(qpos[:3] @ normal),
        *[qpos[indices[name][0]] for name in JOINTS],
    ])
    lower = [0.35]
    upper = [0.55]
    for name in JOINTS:
        joint_range = model.jnt_range[model.joint(f"left_{name}_joint").id]
        lower.append(joint_range[0])
        upper.append(joint_range[1])
    support = [
        model.geom(name).id
        for name in ("left_foot", "right_foot", "left_hand_collision", "right_hand_collision")
    ]

    def apply(values: np.ndarray) -> None:
        qpos[:3] = values[0] * normal
        for value, name in zip(values[1:], JOINTS):
            qpos[indices[name]] = value
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = qpos[7:]
        mujoco.mj_forward(model, data)

    def residual(values: np.ndarray) -> np.ndarray:
        apply(values)
        distances = np.array([support_distance(geom) for geom in support])
        # Four slightly compressed contacts dominate; a light regularizer
        # selects the closest solution to the reviewed prone keyframe.
        regularizer = (values - initial) / np.array([0.05, 0.4, 0.4, 0.25, 0.5, 0.5, 0.5])
        # Heightfield collision tessellation gives the stock foot box roughly
        # 6 mm of compression at the reviewed pose.  The hand capsule needs
        # about 4 cm geometric overlap before its explicit hfield pair closes.
        targets = np.array([-0.006, -0.006, -0.160, -0.160])
        return np.hstack([(distances - targets) / 0.003, 0.02 * regularizer])

    def support_distance(geom: int) -> float:
        center_height = float(data.geom_xpos[geom] @ normal)
        rotation = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        geom_type = model.geom_type[geom]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            extent = float(np.abs(rotation.T @ normal) @ size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = float(size[0] + size[1] * abs(rotation[:, 2] @ normal))
        else:
            raise ValueError(f"unsupported support geom type: {geom_type}")
        return center_height - extent

    result = least_squares(
        residual, initial, bounds=(np.array(lower), np.array(upper)),
        max_nfev=1000, diff_step=1e-4,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    apply(result.x)
    distances = [support_distance(geom) for geom in support]
    print("success", result.success, result.message)
    print("root_normal_height", result.x[0])
    for name, value in zip(JOINTS, result.x[1:]):
        print(name, value)
    print("support_distances", distances)
    print("qpos", " ".join(f"{value:.8g}" for value in qpos))
    for height in np.linspace(0.46, 0.26, 11):
        probe = result.x.copy()
        probe[0] = height
        apply(probe)
        names = set()
        for contact_index in range(data.ncon):
            names.add(model.geom(int(data.contact[contact_index].geom1)).name)
            names.add(model.geom(int(data.contact[contact_index].geom2)).name)
        print(
            "probe", round(float(height), 3),
            "feet", "left_foot" in names and "right_foot" in names,
            "hands", "left_hand_collision" in names and "right_hand_collision" in names,
        )


if __name__ == "__main__":
    main()
