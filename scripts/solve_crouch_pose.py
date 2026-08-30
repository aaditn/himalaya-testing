"""Solve a feet-and-palms crouch with the G1 body suspended above the ramp."""

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from himalaya.env import Joystick, default_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torso-advance",
        type=float,
        default=0.0,
        help="uphill torso displacement relative to the planted limbs (m)",
    )
    parser.add_argument(
        "--render",
        type=Path,
        help="optional path for a static PNG of the solved pre-training pose",
    )
    args = parser.parse_args()
    config = default_config()
    config.impl = "jax"
    env = Joystick(task="climb_terrain", config=config)
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, model.keyframe("knees_bent").id
    )
    mujoco.mj_forward(model, data)
    reference = data.qpos.copy()

    def adr(name: str) -> int:
        return int(model.joint(name).qposadr[0])

    def set_pose(values: np.ndarray) -> None:
        (
            root_x,
            root_z,
            root_pitch,
            hip,
            knee,
            ankle,
            waist,
            shoulder,
            roll,
            shoulder_yaw,
            elbow,
            wrist_pitch,
        ) = values
        data.qpos[:] = reference
        data.qpos[0] = root_x
        data.qpos[2] = root_z
        data.qpos[3:7] = [
            math.cos(root_pitch / 2),
            0.0,
            math.sin(root_pitch / 2),
            0.0,
        ]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            data.qpos[adr(f"{side}_hip_pitch_joint")] = hip
            data.qpos[adr(f"{side}_knee_joint")] = knee
            data.qpos[adr(f"{side}_ankle_pitch_joint")] = ankle
            data.qpos[adr(f"{side}_shoulder_pitch_joint")] = shoulder
            data.qpos[adr(f"{side}_shoulder_roll_joint")] = sign * roll
            data.qpos[adr(f"{side}_shoulder_yaw_joint")] = sign * shoulder_yaw
            data.qpos[adr(f"{side}_elbow_joint")] = elbow
            data.qpos[adr(f"{side}_wrist_pitch_joint")] = wrist_pitch
        data.qpos[adr("waist_pitch_joint")] = waist
        mujoco.mj_forward(model, data)

    feet_ids = [model.site(name).id for name in ("left_foot", "right_foot")]
    palm_ids = [model.site(name).id for name in ("left_palm", "right_palm")]
    knee_ids = [
        model.body(name).id for name in ("left_knee_link", "right_knee_link")
    ]
    torso_id = model.body("torso_link").id
    angle = math.radians(config.climb.slope_degrees)
    slope_tangent = np.array([math.cos(angle), 0.0, math.sin(angle)])
    slope_normal = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    slope = math.tan(math.radians(config.climb.slope_degrees))

    def surface_z(x: float) -> float:
        return 0.20 + slope * x

    foot_targets = data.site_xpos[feet_ids].copy()
    palm_targets = data.site_xpos[palm_ids].copy()
    initial_torso = data.xpos[torso_id].copy()
    torso_target = np.array([
        initial_torso @ slope_tangent + args.torso_advance,
        initial_torso @ slope_normal,
    ])

    def residual(values: np.ndarray) -> np.ndarray:
        set_pose(values)
        feet = data.site_xpos[feet_ids]
        palms = data.site_xpos[palm_ids]
        knees = data.xpos[knee_ids]
        knee_clearance = knees[:, 2] - np.array(
            [surface_z(x) for x in knees[:, 0]]
        )
        clearance_error = np.maximum(0.18 - knee_clearance, 0.0)
        torso = data.xpos[torso_id]
        torso_coordinates = np.array([
            torso @ slope_tangent,
            torso @ slope_normal,
        ])
        body_error = torso_coordinates - torso_target
        return np.concatenate([
            12.0 * (feet - foot_targets).ravel(),
            12.0 * (palms - palm_targets).ravel(),
            6.0 * body_error,
            8.0 * clearance_error,
        ])

    initial = np.array([
        reference[0],
        reference[2],
        2.0 * math.atan2(reference[5], reference[3]),
        reference[adr("left_hip_pitch_joint")],
        reference[adr("left_knee_joint")],
        reference[adr("left_ankle_pitch_joint")],
        reference[adr("waist_pitch_joint")],
        reference[adr("left_shoulder_pitch_joint")],
        reference[adr("left_shoulder_roll_joint")],
        reference[adr("left_shoulder_yaw_joint")],
        reference[adr("left_elbow_joint")],
        reference[adr("left_wrist_pitch_joint")],
    ])
    lower = np.array([
        -0.2, 0.45, -1.4, -2.5, 0.1, -0.87, -0.52,
        -3.0, 0.0, -2.5, -1.0, -1e-7,
    ])
    upper = np.array([
        0.3, 0.75, 1.6, 1.2, 2.85, 0.52, 0.52,
        2.6, 1.5, 2.5, 2.09, 1e-7,
    ])
    solution = least_squares(
        residual, initial, bounds=(lower, upper), max_nfev=2000
    )
    set_pose(solution.x)
    knees = data.xpos[knee_ids]
    knee_clearance = knees[:, 2] - np.array(
        [surface_z(x) for x in knees[:, 0]]
    )
    print("success:", solution.success, "cost:", solution.cost)
    print("variables:", np.array2string(solution.x, precision=6))
    print("qpos:", " ".join(f"{value:.7f}" for value in data.qpos))
    print("ctrl:", " ".join(f"{value:.7f}" for value in data.qpos[7:]))
    print("feet:", data.site_xpos[feet_ids])
    print("palms:", data.site_xpos[palm_ids])
    print("knee clearance:", knee_clearance)
    print("initial torso:", initial_torso)
    print("solved torso:", data.xpos[torso_id])
    print(
        "torso uphill advance:",
        (data.xpos[torso_id] - initial_torso) @ slope_tangent,
    )
    up_adr = model.sensor_adr[model.sensor("upvector_torso").id]
    print("torso upvector:", data.sensordata[up_adr : up_adr + 3])
    print("contacts:", sorted({
        (model.geom(item.geom[0]).name, model.geom(item.geom[1]).name)
        for item in data.contact
    }))
    if args.render is not None:
        import mediapy

        args.render.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=960)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = data.xpos[torso_id]
        camera.distance = 1.65
        camera.azimuth = 90.0
        camera.elevation = -15.0
        renderer.update_scene(data, camera=camera)
        mediapy.write_image(str(args.render), renderer.render())
        renderer.close()
        print("render:", args.render)


if __name__ == "__main__":
    main()
