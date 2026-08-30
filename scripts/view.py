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
# Load scene.py BY PATH, not as himalaya.env.scene: the package __init__
# imports joystick, which imports jax, so a normal import would pull the entire
# training stack onto a laptop that only needs mujoco. scene.py itself is numpy
# only, which is what lets this viewer share the trainer's constants instead of
# keeping its own drifting copy.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "himalaya_scene", ROOT / "himalaya" / "env" / "scene.py")
sc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(sc)

SCENES = {"mountain": "mountain_terrain", "flat": "flat_terrain",
          "rough": "rough_terrain"}


def _terrain_height(model, world_xyz, slope_rad):
    """Relief above the mean plane at a world point, along the slope normal.

    Mirrors G1Env.terrain_height_at (himalaya/env/base.py). Kept here rather
    than in scene.py because it needs the compiled model's hfield data, which
    scene.py deliberately does not load.
    """
    if model.nhfield == 0:
        return 0.0
    nr = int(model.hfield_nrow[0]); nc = int(model.hfield_ncol[0])
    z = float(model.hfield_size[0][2])
    hx = float(model.hfield_size[0][0]); hy = float(model.hfield_size[0][1])
    g = np.array(model.hfield_data[: nr * nc]).reshape(nr, nc) * z
    R = np.array([[np.cos(slope_rad), 0.0, np.sin(slope_rad)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(slope_rad), 0.0, np.cos(slope_rad)]])
    local = R.T @ np.asarray(world_xyz, dtype=float)
    fx = np.clip((local[0] + hx) / (2 * hx) * (nc - 1), 0, nc - 1)
    fy = np.clip((local[1] + hy) / (2 * hy) * (nr - 1), 0, nr - 1)
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, nc - 1), min(y0 + 1, nr - 1)
    tx, ty = fx - x0, fy - y0
    top = g[y0, x0] * (1 - tx) + g[y0, x1] * tx
    bot = g[y1, x0] * (1 - tx) + g[y1, x1] * tx
    return float(top * (1 - ty) + bot * ty)


def build(scene, slope_deg):
    slope = np.deg2rad(slope_deg)
    xml = sc.tilted_xml(SCENES[scene], slope)
    model = mujoco.MjModel.from_xml_string(xml, assets=sc.local_assets())
    return model, slope


# --- all-fours pose -------------------------------------------------------
# Order after the 7 free-joint values: 6 left leg, 6 right leg, 3 waist,
# 7 left arm, 7 right arm. See himalaya/env/g1_constants.py.
CROUCH = np.array([
    # Deep crouch with the palms planted BESIDE the feet -- not a quadruped
    # stance, which this robot cannot assume. Measured: the arm is 0.460 m
    # shoulder-to-palm, the leg 0.657 m hip-to-sole, and the shoulder sits
    # 0.394 m above the hip, leaving a 0.631 m deficit to the ground. Waist
    # pitch caps at 30 degrees and its segment is only 0.291 m long, so folding
    # there drops the shoulder just 0.038 m. Only the hip has the authority,
    # and folding at the hip swings the feet backward as fast as it brings the
    # hands forward. Searching 400k poses, 35k of them self-collision free, the
    # maximum forward reach of palm past foot is 0.144 m -- 0.077 m with the
    # palms at sole level. A bear-crawl needs 0.4-0.5 m.
    #
    # So the hands go to the SIDES, where they can brace against the corridor
    # banks, which is what the walls are for.
    #
    # This pose was solved with self-intersection as a hard constraint; the
    # previous one put the hands through the thighs by 0.086 m.
    #
    #  left leg: hip p/r/y, knee, ankle p/r
    -2.498, 0.947, 0.000, 1.024, -0.584, 0.000,
    # right leg
    -2.498, -0.524, 0.000, 1.024, -0.584, 0.000,
    # waist yaw/roll/pitch, then left shoulder pitch/roll/yaw
    0.000, 0.000, 0.498, -1.753, 0.116, 0.107,
    # left elbow, left wrist r/p/y, right shoulder pitch/roll
    1.604, 0.000, 0.000, 0.000, -1.753, -0.116,
    # right shoulder yaw, right elbow, right wrist r/p/y
    -0.107, 1.604, 0.000, 0.000, 0.000,
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climb", type=float, default=35.0, help="slope degrees")
    ap.add_argument("--scene", default="mountain", choices=list(SCENES))
    ap.add_argument("--pose", default="knees_bent",
                    choices=["knees_bent", "home", "crouch"])
    ap.add_argument("--terrain-only", action="store_true",
                    help="hide the robot; look at the mountain")
    ap.add_argument("--free", action="store_true",
                    help="do not hold the pose -- let physics take it")
    ap.add_argument("--x", type=float, default=None,
                    help="spawn x up-slope; default = the training spawn")
    ap.add_argument("--y", type=float, default=None,
                    help="spawn y; default = the training spawn")
    args = ap.parse_args()

    model, slope = build(args.scene, args.climb)
    data = mujoco.MjData(model)

    key = 0
    for k in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == args.pose:
            key = k
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, key)
    if args.pose == "crouch":
        data.qpos[7:] = CROUCH

    # --- spawn placement, mirroring joystick.py reset() -------------------
    if slope != 0.0 and not args.terrain_only:
        # Spawn where TRAINING spawns. Both numbers come from scene.py, so
        # changing the spawn moves the viewer and the trainer together -- the
        # split copy here is exactly what let them disagree before.
        # ONE definition of the spawn, shared with joystick.py reset(). Do not
        # reimplement any part of it here -- that split is what made this viewer
        # show a different heading and height than training used.
        init_h = float(model.keyframe("knees_bent").qpos[2])
        data.qpos[:] = sc.spawn_pose(
            data.qpos, slope, init_h,
            lambda p: _terrain_height(model, p, slope))
        if args.x is not None:
            data.qpos[0] = args.x
        if args.y is not None:
            data.qpos[1] = args.y
        x, y = float(data.qpos[0]), float(data.qpos[1])
        mujoco.mj_forward(model, data)
        # Lift until nothing is inside the FLOOR.
        #
        # Only floor contacts count. A self-intersecting pose -- arms through
        # thighs, say -- never clears however high you lift, so counting every
        # contact ran this loop to its ceiling and reported a 4 m lift.
        floor_gid = model.geom("floor").id
        lift, stuck = 0.0, False
        for i in range(400):
            mujoco.mj_forward(model, data)
            worst = 0.0
            for c in range(data.ncon):
                con = data.contact[c]
                if floor_gid in (con.geom1, con.geom2):
                    worst = min(worst, con.dist)
            if worst >= -0.005:
                break
            data.qpos[2] += 0.01
            lift += 0.01
        else:
            stuck = True
        # Report self-intersection rather than hiding it in a silent lift.
        selfcon = []
        for c in range(data.ncon):
            con = data.contact[c]
            if floor_gid not in (con.geom1, con.geom2) and con.dist < -0.002:
                selfcon.append((
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2),
                    con.dist))
        if stuck:
            print("  WARNING: never cleared the floor in 400 steps")
        for g1, g2, dist in selfcon:
            print(f"  SELF-INTERSECTION: {g1} <-> {g2} by {-dist:.3f} m")
        print(f"spawn: x={x:.2f} y={y:.2f} z={data.qpos[2]:.2f} "
              f"(lifted {lift:.2f} m clear of the rock)")

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
