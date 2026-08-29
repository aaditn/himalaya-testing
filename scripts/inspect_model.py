"""Does the 29-DOF G1 stand on flat ground holding its default pose?

No policy, no terrain, no rewards -- hold the standing keyframe with the
model's own actuators and see whether the robot stays upright. If this fails,
the model is wrong and no amount of reward tuning will help.

Usage:  .venv/bin/python scripts/inspect_model.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from himalaya.mjx import g1_29dof  # noqa: E402


def main():
    # Full-fidelity scene, NOT the MJX one: the MJX scene runs 5 solver
    # iterations, which cannot hold a humanoid up open-loop. Judging the
    # model on it would report a collapse that is a solver artifact.
    model, qpos0, ctrl0 = g1_29dof.load(mjx=False)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)

    n_act = sum(1 for j in range(model.njnt)
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE)
    print(f"total mass      {model.body_mass.sum():.2f} kg")
    print(f"standing height {data.qpos[2]:.3f} m (Menagerie keyframe)")
    print(f"scene           {g1_29dof.scene_path(mjx=False).split('/')[-1]}")
    print(f"actuators       {model.nu}   actuated joints {n_act}")

    # Menagerie ships the matching actuator targets with the keyframe; use
    # them rather than rebuilding from qpos.
    data.ctrl[:] = ctrl0

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
