"""Does the 23-DOF G1 stand on flat ground holding its default pose?

No policy, no terrain, no rewards -- hold the nominal pose with the PD
controllers and see whether the robot stays upright. If this fails, the model
is wrong and no amount of reward tuning will help.

Usage:  .venv/bin/python scripts/inspect_model.py
"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from himalaya.mjx.g1_model import DEFAULT_POSE, STANDING_HEIGHT, load  # noqa: E402


def main():
    model, qpos0 = load()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)

    print(f"total mass      {model.body_mass.sum():.2f} kg")
    print(f"standing height {STANDING_HEIGHT:.3f} m (measured from foot geometry)")
    print(f"actuators       {model.nu}")

    # hold the nominal pose; ctrl for position servos is the target angle
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        ctrl[a] = DEFAULT_POSE.get(names[jid], 0.0)
    data.ctrl[:] = ctrl

    print(f"\n{'t(s)':>6} {'pelvis_z':>9} {'upright':>8} {'com_vel':>8}")
    dt = model.opt.timestep
    for step in range(int(3.0 / dt)):
        mujoco.mj_step(model, data)
        if step % int(0.5 / dt) == 0:
            z = data.qpos[2]
            # body-frame z of the world up-axis: 1.0 upright, 0 on its side
            upright = data.xmat[1].reshape(3, 3)[2, 2]
            vel = np.linalg.norm(data.qvel[:3])
            print(f"{step * dt:6.2f} {z:9.3f} {upright:8.3f} {vel:8.3f}")

    z, upright = data.qpos[2], data.xmat[1].reshape(3, 3)[2, 2]
    print()
    if z > 0.6 and upright > 0.85:
        print(f"VERDICT: STANDS   (z={z:.3f}, upright={upright:.3f})")
        return 0
    if z < 0.35:
        print(f"VERDICT: COLLAPSED   (z={z:.3f})")
    else:
        print(f"VERDICT: UNSTABLE   (z={z:.3f}, upright={upright:.3f})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
