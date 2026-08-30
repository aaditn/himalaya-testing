"""Render the SPAWN with no policy: physics only, holding the reset pose.

Answers what a reward curve cannot: does the robot stay where it is put, and
in the pose we intended. Zero ACTION is not zero torque -- step() commands
_default_pose + action*scale, so action=0 snaps the robot to the clean keyframe
even though reset() spawned it at a jittered one. --hold commands the pose the
robot actually spawned in, which is the honest passive test.
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

ap = argparse.ArgumentParser()
ap.add_argument("--climb", type=float, default=35.0)
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--out", default="videos/spawn_check.mp4")
ap.add_argument("--hold", action="store_true",
                help="command the pose the robot spawned in, not _default_pose")
ap.add_argument("--no-push", action="store_true", help="disable random pushes")
ap.add_argument("--no-jitter", action="store_true",
                help="spawn at the exact keyframe pose, no joint scaling")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=640)
a = ap.parse_args()

import jax, jax.numpy as jp, numpy as np, mediapy, mujoco
from himalaya.env import Joystick, climb_config

cfg = climb_config(a.climb)
if a.no_push:
    cfg.push_config.enable = False
env = Joystick(task="mountain_terrain", config=cfg)
if a.no_jitter:
    env._reset_jitter_off = True   # read by the patch below
reset, step = jax.jit(env.reset), jax.jit(env.step)
n_up = jp.array(env._slope_normal)

per = int(a.seconds / env.dt)
rollout, stats = [], []
for ep in range(a.episodes):
    st = reset(jax.random.PRNGKey(ep))
    # The action that commands the SPAWN pose: action*scale = q_spawn - default
    hold_act = ((st.data.qpos[7:] - env._default_pose)
                / env._config.action_scale) if a.hold else jp.zeros(env.action_size)
    # CLEARANCE above the terrain underfoot, not displacement along the normal.
    #
    # dot(qpos, slope_normal) confounds sliding with falling: a robot that
    # slides 60 m down a 35 degree incline, in contact the whole way, registers
    # as a 50 m "drop" while never leaving the ground. That is how a pure
    # sliding failure was first misread here as falling through the world.
    def clearance(state):
        p = state.data.qpos[0:3]
        return float(jp.dot(p, n_up) - env.terrain_height_at(p))

    c0 = clearance(st)
    lo, done_at = c0, None
    slid = 0.0
    p0 = np.array(st.data.qpos[0:3])
    for t in range(per):
        st = step(st, hold_act)
        lo = min(lo, clearance(st))
        slid = max(slid, float(np.linalg.norm(np.array(st.data.qpos[0:3]) - p0)))
        rollout.append(st)
        if float(st.done) and done_at is None:
            done_at = t
    stats.append((ep, c0, lo, slid, done_at))

mode = ("hold-spawn-pose" if a.hold else "zero-action(=default pose)")
print(f"passive spawn: {mode}"
      f"{', pushes off' if a.no_push else ''}"
      f"{', no joint jitter' if a.no_jitter else ''}")
surv = 0
for ep, c0, lo, slid, d in stats:
    ok = d is None
    surv += ok
    print(f"  ep{ep}: clearance {c0:+.2f} -> {lo:+.2f} m, travelled {slid:5.2f} m, "
          f"{'survived ' + str(a.seconds) + 's' if ok else 'terminated at step ' + str(d)}")
print(f"  survived: {surv}/{len(stats)}")
print("  (clearance = height above the terrain underfoot; travelled = distance "
      "from the spawn point. Large travel with steady clearance is SLIDING, "
      "not falling.)")

m = env.mj_model
d = mujoco.MjData(m)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, -10.0
r = mujoco.Renderer(m, height=a.height, width=a.width)
frames = []
for st in rollout:
    d.qpos[:] = np.array(st.data.qpos); d.qvel[:] = np.array(st.data.qvel)
    mujoco.mj_forward(m, d)
    cam.lookat[0], cam.lookat[1] = float(d.qpos[0]), float(d.qpos[1])
    cam.lookat[2] = float(d.qpos[2])
    r.update_scene(d, camera=cam)
    frames.append(r.render())
r.close()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
mediapy.write_video(str(out), frames, fps=1.0 / env.dt)
print(f"wrote {out} ({len(frames)} frames)")
