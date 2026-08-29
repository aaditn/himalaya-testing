"""Watch the G1 in MuJoCo's native viewer, locally. No pod, no VNC.

macOS needs mjpython, NOT python: the GUI must own the main thread.

Usage:
    .venv/bin/mjpython scripts/view.py              # Playground rough terrain
    .venv/bin/mjpython scripts/view.py --flat       # Playground flat terrain
    .venv/bin/mjpython scripts/view.py --menagerie  # bare Menagerie G1 standing

Controls: drag to orbit, scroll to zoom, space to pause, Esc to quit.
"""
import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true", help="flat terrain instead of rough")
    ap.add_argument("--menagerie", action="store_true", help="bare Menagerie model")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until closed")
    args = ap.parse_args()

    if args.menagerie:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from himalaya.mjx import g1_29dof
        # Full-fidelity scene: this one actually holds a pose open-loop.
        model = mujoco.MjModel.from_xml_path(g1_29dof.scene_path(mjx=False))
        label = "menagerie unitree_g1 (29-DOF)"
    else:
        from mujoco_playground import registry
        name = "G1JoystickFlatTerrain" if args.flat else "G1JoystickRoughTerrain"
        model = registry.load(name).mj_model
        label = name

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # hold the spawn pose so the robot stands rather than flopping
    if model.nu:
        ctrl = np.zeros(model.nu)
        for a in range(model.nu):
            jid = model.actuator_trnid[a, 0]
            if jid >= 0:
                ctrl[a] = data.qpos[model.jnt_qposadr[jid]]
        data.ctrl[:] = ctrl

    print(f"{label}: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"mass={model.body_mass.sum():.1f}kg")
    print("drag=orbit  scroll=zoom  space=pause  Esc=quit")

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            sys.exit(
                "On macOS this must run under mjpython, not python:\n"
                "    .venv/bin/mjpython scripts/view.py " + " ".join(sys.argv[1:])
            )
        raise

    with viewer:
        start = time.time()
        while viewer.is_running():
            if args.seconds and time.time() - start > args.seconds:
                break
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # keep it near real time
            lag = model.opt.timestep - (time.time() - step_start)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    sys.exit(main())
