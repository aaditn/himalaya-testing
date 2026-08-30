"""Watch the climbing environment LIVE, locally. No pod, no video files.

macOS needs mjpython, NOT python: the GUI must own the main thread.

    .venv/bin/mjpython scripts/view.py                    # 35 deg climb scene
    .venv/bin/mjpython scripts/view.py --climb 25
    .venv/bin/mjpython scripts/view.py --pose all_fours   # spawn on all fours
    .venv/bin/mjpython scripts/view.py --terrain-only     # just the mountain
    .venv/bin/mjpython scripts/view.py --free             # let it fall, no hold

Controls: drag to orbit, scroll to zoom, space to pause, Esc to quit.

This builds the model the same way himalaya/env/base.py does -- same XML, same
slope quat applied to the floor BEFORE compile (MuJoCo bakes worldbody geom
orientation at compile time, so writing geom_quat afterwards silently does
nothing), same spawn placement. What you orbit here is what trains.

Deliberately does NOT import himalaya.env: that pulls jax and the whole
training stack onto a laptop for what is a CPU physics problem. The cost is
that the spawn maths below is a second copy; it is marked so it can be kept
in step with joystick.py.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# himalaya.env.scene is numpy-only on purpose, so importing it here does NOT
# pull jax onto the laptop. It owns the scene list, the slope maths, the spawn
# constants and the asset loading, which is what stops this viewer from
# drifting away from what actually trains.
from himalaya.env import scene as sc

SCENES = {"mountain": "mountain_terrain", "flat": "flat_terrain",
          "rough": "rough_terrain"}


def build(scene, slope_deg):
    slope = np.deg2rad(slope_deg)
    xml = sc.tilted_xml(SCENES[scene], slope)
    model = mujoco.MjModel.from_xml_string(xml, assets=sc.local_assets())
    return model, slope


# --- all-fours pose -------------------------------------------------------
# Order after the 7 free-joint values: 6 left leg, 6 right leg, 3 waist,
# 7 left arm, 7 right arm. See himalaya/env/g1_constants.py.
ALL_FOURS = np.array([
    # Solved numerically, not guessed: search over hip pitch, knee, ankle,
    # waist pitch, shoulder pitch/roll and elbow for a stance with the palms
    # level with the soles and well forward of them. Result: hands 0.055 m
    # above the feet, 0.28 m in front.
    #
    # Two constraints shape it. Waist pitch caps at +/-0.52 rad (30 deg), so
    # the torso CANNOT fold far at the waist -- the lean has to come from the
    # hips, which is why hip pitch sits at -0.80 with the knee near its 2.88
    # limit. And shoulder pitch is negative-forward, so reaching a palm
    # downhill needs a negative value.
    #
    #  left leg: hip p/r/y, knee, ankle p/r
    -0.801, 0.000, 0.000, 2.836, -0.216, 0.000,
    # right leg
    -0.801, 0.000, 0.000, 2.836, -0.216, 0.000,
    # waist yaw/roll/pitch, then left shoulder pitch/roll/yaw
    0.000, 0.000, 0.458, -0.574, 0.015, 0.000,
    # left elbow, left wrist r/p/y, right shoulder pitch/roll
    1.431, 0.000, 0.000, 0.000, -0.574, -0.015,
    # right shoulder yaw, right elbow, right wrist r/p/y
    0.000, 1.431, 0.000, 0.000, 0.000,
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climb", type=float, default=35.0, help="slope degrees")
    ap.add_argument("--scene", default="mountain", choices=list(SCENES))
    ap.add_argument("--pose", default="knees_bent",
                    choices=["knees_bent", "home", "all_fours"])
    ap.add_argument("--terrain-only", action="store_true",
                    help="hide the robot; look at the mountain")
    ap.add_argument("--free", action="store_true",
                    help="do not hold the pose -- let physics take it")
    ap.add_argument("--x", type=float, default=None,
                    help="spawn x up-slope; default = middle of the training range")
    ap.add_argument("--y", type=float, default=None,
                    help="spawn y; default = the training spawn for --lane")
    ap.add_argument("--lane", type=int, default=0,
                    help="which route to start at (0-3)")
    args = ap.parse_args()

    model, slope = build(args.scene, args.climb)
    data = mujoco.MjData(model)

    key = 0
    for k in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == args.pose:
            key = k
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, key)
    if args.pose == "all_fours":
        data.qpos[7:] = ALL_FOURS

    # --- spawn placement, mirroring joystick.py reset() -------------------
    if slope != 0.0 and not args.terrain_only:
        # Spawn where TRAINING spawns. Both numbers come from scene.py, so
        # changing the spawn moves the viewer and the trainer together -- the
        # split copy here is exactly what let them disagree before.
        lanes = sc.lane_mouths()
        x = args.x if args.x is not None else 0.5 * sum(sc.SPAWN_X)
        if args.y is not None:
            y = args.y
        elif lanes is not None:
            y = sc.spawn_y(lanes[args.lane % len(lanes)])
        else:
            y = 0.0
        data.qpos[0], data.qpos[1] = x, y
        data.qpos[2] = data.qpos[2] + sc.surface_z(x, slope)
        # tilt the body to stand perpendicular to the slope
        h = 0.5 * slope
        tilt = np.array([np.cos(h), 0.0, np.sin(h), 0.0])
        q = data.qpos[3:7].copy()
        mujoco.mju_mulQuat(data.qpos[3:7], tilt, q)
        mujoco.mj_forward(model, data)
        # lift until the lowest contacting geom clears the terrain
        lift = 0.0
        for _ in range(400):
            mujoco.mj_forward(model, data)
            if data.ncon and min(data.contact.dist[:data.ncon]) < -0.005:
                data.qpos[2] += 0.01
                lift += 0.01
            else:
                break
        print(f"spawn: x={x:.2f} y={y:.2f} z={data.qpos[2]:.2f} "
              f"(lifted {lift:.2f} m clear of the rock)  "
              f"[training range x {sc.SPAWN_X[0]}-{sc.SPAWN_X[1]}]")

    if args.terrain_only:
        # Park the robot far away rather than deleting it from the model.
        data.qpos[0:3] = [0.0, 0.0, 500.0]

    mujoco.mj_forward(model, data)
    hold = data.qpos[7:].copy()
    if model.nu and not args.free:
        data.ctrl[:] = hold

    print(f"scene={args.scene} slope={args.climb} deg pose={args.pose} "
          f"{'FREE' if args.free else 'holding pose'}")
    print(f"model: nq={model.nq} nu={model.nu} "
          f"mass={model.body_mass.sum():.1f} kg")
    print("drag=orbit  scroll=zoom  space=pause  Esc=quit")

    with mujoco.viewer.launch_passive(model, data) as v:
        if args.terrain_only:
            v.cam.lookat[:] = [0.0, 0.0, 0.0]
            v.cam.distance = 18.0
        else:
            v.cam.lookat[:] = data.qpos[0:3]
            v.cam.distance = 4.0
        v.cam.azimuth, v.cam.elevation = 90.0, -15.0
        while v.is_running():
            t0 = time.time()
            if model.nu and not args.free:
                data.ctrl[:] = hold
            mujoco.mj_step(model, data)
            v.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
