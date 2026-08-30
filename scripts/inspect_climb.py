"""Does the G1 hold a four-point stance -- on flat ground, and on the slope?

No policy, no rewards. Hold the climb keyframe with the model's own position
actuators and see whether all four contacts stay loaded. If this fails, the
scene or the stance is wrong and no reward tuning will save it.

    .venv/bin/python scripts/inspect_climb.py            # flat control
    .venv/bin/python scripts/inspect_climb.py --incline  # 35 deg rough slope
    .venv/bin/python scripts/inspect_climb.py --incline --slope 45 --friction 0.5

Like scripts/inspect_model.py this runs the FULL-FIDELITY solver, not the 3
iterations the MJX scene ships. A low-iteration solver cannot hold a 33 kg body
open-loop, so a collapse there would be an artifact rather than a finding.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from himalaya.env.base import get_assets  # noqa: E402

XMLS = Path(__file__).resolve().parent.parent / "himalaya" / "env" / "xmls"
# The four limb-tip contacts we score. Resolved BY NAME, never by index:
# adding the leg pairs reordered the table so the feet moved from 0,1 to 2,5.
GROUND_PAIRS = ["left_foot_floor", "right_foot_floor",
                "left_hand_floor", "right_hand_floor"]
# every geom-vs-floor pair, for setting friction consistently
ALL_FLOOR_PAIRS = GROUND_PAIRS + [
    "left_thigh_floor", "right_thigh_floor", "left_shin_floor", "right_shin_floor"]

# Solved by scripts/find_quadruped_pose.py on flat ground.
CRAWL_PITCH = 0.8249
JOINTS = np.array([
    -2.3185, 0.0000, 0.0000, 2.6716, -0.8438, 0.0000,
    -2.3185, 0.0000, 0.0000, 2.6716, -0.8438, 0.0000,
    0.0000, 0.0000, 0.0000, -1.0447, 0.1500, 0.0000,
    1.4236, 0.0000, 0.0000, 0.0000, -1.0447, -0.1500,
    0.0000, 1.4236, 0.0000, 0.0000, 0.0000,
])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load(incline, slope_rad, friction, plane=False):
    """plane=True tilts the FLAT scene's smooth plane instead of using the
    heightfield -- the control that separates 'the slope is hard' from 'the
    heightfield is the problem'."""
    if plane:
        xml = XMLS / "scene_mjx_climb_flat.xml"
        text = xml.read_text().replace(
            '<geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>',
            f'<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" euler="0 {-slope_rad:.6f} 0"/>')
    else:
        xml = XMLS / ("scene_mjx_climb_incline.xml" if incline else "scene_mjx_climb_flat.xml")
        text = xml.read_text()
        if incline:
            text = text.replace('euler="0 -0.6109 0"', f'euler="0 {-slope_rad:.6f} 0"')
    model = mujoco.MjModel.from_xml_string(text, assets=get_assets())
    # Full-fidelity solver for an open-loop judgement (see module docstring).
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    if friction is not None:
        for name in ALL_FLOOR_PAIRS:
            pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
            if pid >= 0:
                model.pair_friction[pid, 0:2] = friction
    return model


def place(model, data, slope_rad, qpos0):
    """Drop the solved stance along the slope normal until it just touches.

    The pose comes from scripts/find_quadruped_pose.py solved AT THIS SLOPE.
    Rotating a flat-ground stance onto the slope instead leaves the body level
    with the world rather than braced against gravity, which measured 0.62 m of
    sag and a 2 m slide.
    """
    normal = ry(-slope_rad) @ np.array([0.0, 0.0, 1.0])

    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def n_floor_contacts():
        # Count ONLY floor contacts. The model also declares self-collision
        # pairs (left_hand_thigh, left_foot_right_foot); a crawl pose can touch
        # those at any height, which made the height search report "already in
        # contact" and abort.
        return sum(1 for i in range(data.ncon)
                   if floor_gid in (data.contact[i].geom1, data.contact[i].geom2))

    def set_at(offset):
        data.qpos[:] = qpos0
        data.qpos[0:3] = qpos0[0:3] + normal * offset
        mujoco.mj_forward(model, data)

    # binary search the offset at which contact first appears
    lo, hi = -0.30, 0.80
    set_at(hi)
    if n_floor_contacts() > 0:
        raise SystemExit("robot already touching the floor at max offset -- bad scene")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        set_at(mid)
        if n_floor_contacts() > 0:
            lo = mid
        else:
            hi = mid
    set_at(hi)
    return hi, normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incline", action="store_true")
    ap.add_argument("--slope", type=float, default=35.0, help="degrees")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--plane", action="store_true", help="smooth tilted plane instead of hfield")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    slope_rad = np.deg2rad(args.slope) if (args.incline or args.plane) else 0.0
    model = load(args.incline, slope_rad, args.friction, plane=args.plane)
    data = mujoco.MjData(model)

    pair_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, n)
                for n in GROUND_PAIRS}
    print(f"scene      {'tilted PLANE (control)' if args.plane else ('incline hfield' if args.incline else 'flat (control)')}"
          f"   slope {args.slope if (args.incline or args.plane) else 0.0:.1f} deg")
    print(f"friction   {model.pair_friction[pair_ids['left_foot_floor'], 0]:.3f}"
          f"   (mu needed to not slide: {np.tan(slope_rad):.3f})")
    print(f"solver     {model.opt.iterations} iterations (full fidelity)")

    qfile = Path(f"/tmp/quad_qpos_{args.slope if (args.incline or args.plane) else 0:g}.npy")
    if not qfile.exists():
        raise SystemExit(f"no solved stance for {args.slope} deg: run\n"
                         f"  .venv/bin/python scripts/find_quadruped_pose.py --slope {args.slope:g}")
    qpos0 = np.load(qfile)
    offset, normal = place(model, data, slope_rad, qpos0)
    data.ctrl[:] = qpos0[7:]
    start = data.qpos[0:3].copy()
    spawn_h = float(data.qpos[0:3] @ normal)

    print(f"\n{'t(s)':>6} {'height':>8} {'Lf':>3} {'Rf':>3} {'Lh':>3} {'Rh':>3} {'slide':>7}")
    dt = model.opt.timestep
    survived = args.seconds
    for step in range(int(args.seconds / dt)):
        mujoco.mj_step(model, data)
        if survived == args.seconds and step * dt > 0.25:
            on_now = contacts(model, data, pair_ids)
            if sum(on_now.values()) < 3 or np.linalg.norm(data.qvel[0:3]) > 0.6:
                survived = step * dt
        if step % int(0.5 / dt) == 0:
            on = contacts(model, data, pair_ids)
            h = float(data.qpos[0:3] @ normal)
            slide = float(np.linalg.norm(data.qpos[0:3] - start))
            print(f"{step*dt:6.2f} {h:8.3f} "
                  + " ".join(f"{int(on[n]):>3}" for n in GROUND_PAIRS)
                  + f" {slide:7.3f}")

    on = contacts(model, data, pair_ids)
    n_on = sum(on.values())
    h = float(data.qpos[0:3] @ normal)
    slide = float(np.linalg.norm(data.qpos[0:3] - start))
    # what matters is whether it is STILL moving, not how far it settled
    speed = float(np.linalg.norm(data.qvel[0:3]))
    print()
    print(f"contacts loaded at end: {n_on}/4  ({', '.join(n for n in GROUND_PAIRS if on[n]) or 'none'})")
    print(f"pelvis height above surface: {h:.3f} m  (spawned {spawn_h:.3f}, sag {spawn_h-h:+.3f})")
    print(f"travelled {slide:.3f} m   residual speed {speed:.4f} m/s")
    print(f"SURVIVAL {survived:.2f}s of {args.seconds:.1f}s (>=3 contacts and under 0.6 m/s)")
    if n_on == 4 and h > 0.15 and speed < 0.05:
        print("VERDICT: FOUR-POINT STANCE HOLDS")
        return 0
    if n_on < 4:
        print(f"VERDICT: LOST CONTACTS ({n_on}/4) -- stance or scene is wrong")
    elif speed >= 0.05:
        print(f"VERDICT: SLIDING at {speed:.3f} m/s")
    else:
        print(f"VERDICT: COLLAPSED (pelvis {h:.3f} m)")
    return 1


def contacts(model, data, pair_ids):
    on = {n: False for n in pair_ids}
    inv = {v: k for k, v in pair_ids.items()}
    for i in range(data.ncon):
        c = data.contact[i]
        # geom pair -> which declared <pair> produced it
        for name, pid in pair_ids.items():
            g1, g2 = model.pair_geom1[pid], model.pair_geom2[pid]
            if {c.geom1, c.geom2} == {g1, g2}:
                on[name] = True
    return on


if __name__ == "__main__":
    sys.exit(main())
