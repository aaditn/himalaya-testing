"""Gravity-compensate the spawn pose for servo sag.

The G1's actuators are position servos with kp=75, so holding a static load
requires a steady-state deflection of tau/kp -- 0.12 to 0.27 rad at the torques
a crawl demands. Commanding the desired joint angles therefore SETTLES
somewhere else: measured 0.107 m of pelvis sag on flat ground and ~1.0 m on a
37.5 deg slope, where the shifted COM then slid the stance out.

This solves the inverse problem: find ctrl targets whose settled pose IS the
pose we want. Simple fixed-point iteration -- command, settle, measure the
error, push the target further by that error, repeat.

    .venv/bin/python scripts/compensate_stance.py --slope 37.5 --friction 1.0
"""
import argparse, sys
from pathlib import Path
import mujoco, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ic", str(Path(__file__).resolve().parent / "inspect_climb.py"))
ic = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ic)


def settle(model, data, qpos0, ctrl, normal, seconds=2.0):
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), float(np.linalg.norm(data.qvel[0:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slope", type=float, default=37.5)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=25)
    args = ap.parse_args()

    slope = np.deg2rad(args.slope)
    model = ic.load(args.slope > 0, slope, args.friction)
    data = mujoco.MjData(model)
    normal = ic.ry(-slope) @ np.array([0.0, 0.0, 1.0])

    qpos0 = np.load(f"/tmp/quad_qpos_{args.slope:g}.npy")
    _, normal2 = ic.place(model, data, slope, qpos0)
    qpos0 = data.qpos.copy()
    des = qpos0[7:].copy()

    ctrl = des.copy()
    best = (np.inf, None, None)
    for r in range(args.rounds):
        q, speed = settle(model, data, qpos0, ctrl, normal)
        err = des - q[7:]
        h = float(q[0:3] @ normal)
        score = float(np.abs(err).max())
        if score < best[0]:
            best = (score, ctrl.copy(), (h, speed))
        if r % 5 == 0 or r == args.rounds - 1:
            print(f"  round {r:2d}  max joint err {score:.4f} rad   "
                  f"pelvis {h:+.3f} m   speed {speed:.3f} m/s")
        ctrl = np.clip(ctrl + 0.7 * err,
                       model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    score, ctrl, (h, speed) = best
    print(f"\nbest: max joint error {score:.4f} rad, pelvis {h:.3f} m above plane, speed {speed:.3f} m/s")
    np.save(f"/tmp/quad_ctrl_{args.slope:g}.npy", ctrl)
    print(f"saved /tmp/quad_ctrl_{args.slope:g}.npy")
    print('ctrl="' + " ".join(f"{x:.4f}" for x in ctrl) + '"')


if __name__ == "__main__":
    main()
