"""Solve a four-point stance for the G1 on a slope of a given angle.

Two things make this more than picking joint angles by eye.

1. ACTUATOR LIMITS. The shoulder_pitch and elbow are limited to +-25 Nm (real
   G1 numbers, in g1_mjx_feetonly.xml). A crawl that reaches the hands far
   ahead of the shoulders needs load x moment-arm past that, the arms saturate,
   and the front end sinks. The search scores static torque demand.

2. THE SLOPE FRAME IS NOT THE GRAVITY FRAME. Taking a stance solved on flat
   ground and rotating it to sit on the slope leaves the body level with the
   WORLD, not braced against gravity, and the centre of mass falls outside the
   support polygon -- measured: 0.107 m of sag on flat became 0.62 m at 35 deg
   and the robot slid 1-2 m and fell through the terrain. Contact heights and
   the COM projection are therefore measured along the true vertical and the
   slope normal separately.

    .venv/bin/python scripts/find_quadruped_pose.py --slope 37.5

Prints a <key> block and saves the qpos to /tmp/quad_qpos_<slope>.npy.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAC_LO, FRAC_HI = 0.30, 0.60
HEAD_H = 0.0   # head clearance target; set by --head-h
XMLS = Path(__file__).resolve().parent.parent / "himalaya" / "env" / "xmls"

# qpos: 3 pos + 4 quat, then 29 joints.
L_HIP_P, L_KNEE, L_ANK_P = 0, 3, 4
R_HIP_P, R_KNEE, R_ANK_P = 6, 9, 10
WAIST_P = 14
L_SHO_P, L_SHO_R, L_ELBOW = 15, 16, 18
R_SHO_P, R_SHO_R, R_ELBOW = 22, 23, 25
CONTACTS = ["left_foot", "right_foot", "left_hand_collision", "right_hand_collision"]


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def extreme_along(model, data, geom_name, direction):
    """Signed distance of a geom's most-negative point along `direction`."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    pos, R, size = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
    d = float(pos @ direction)
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        return d - float(np.abs(R.T @ direction) @ size[:3])
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = R[:, 2]
        return d - abs(float(axis @ direction)) * float(size[1]) - float(size[0])
    raise SystemExit(f"unhandled geom type for {geom_name}")


def build_qpos(p):
    pitch, pz, px, sho_p, elbow, hip_p, knee, ank_p, waist_p = p
    q = np.zeros(36)
    q[0], q[2] = px, pz
    q[3], q[5] = np.cos(pitch / 2), np.sin(pitch / 2)
    j = q[7:]
    j[L_HIP_P] = j[R_HIP_P] = hip_p
    j[L_KNEE] = j[R_KNEE] = knee
    j[L_ANK_P] = j[R_ANK_P] = ank_p
    j[L_SHO_P] = j[R_SHO_P] = sho_p
    j[L_SHO_R], j[R_SHO_R] = 0.15, -0.15
    j[L_ELBOW] = j[R_ELBOW] = elbow
    j[WAIST_P] = waist_p
    return q


def evaluate(model, data, p, normal, uphill, slope):
    q = build_qpos(p)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    # (a) all four contacts on the inclined plane (measured along its normal)
    d = [extreme_along(model, data, g, normal) for g in CONTACTS]
    cost = float(np.sum(np.square(d))) * 400.0

    # (b) centre of mass, dropped along TRUE VERTICAL, must land inside the
    #     support polygon -- this is what rotating the flat stance got wrong.
    com = np.array(data.subtree_com[0])
    z = np.array([0.0, 0.0, 1.0])
    t = float(com @ normal) / float(z @ normal)
    ground_pt = com - t * z
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    u_feet = float(np.mean([data.geom_xpos[gid(g)] @ uphill for g in CONTACTS[:2]]))
    u_hands = float(np.mean([data.geom_xpos[gid(g)] @ uphill for g in CONTACTS[2:]]))
    u_com = float(ground_pt @ uphill)
    span = u_hands - u_feet
    frac = (u_com - u_feet) / span if abs(span) > 1e-6 else 0.5
    # keep the COM between the limbs, biased slightly uphill (into the hill)
    cost += max(0.0, FRAC_LO - frac) ** 2 * 600.0
    cost += max(0.0, frac - FRAC_HI) ** 2 * 600.0
    cost += max(0.0, 0.30 - span) ** 2 * 200.0   # a real stance, not a huddle

    # (c) static torque demand at the weak joints
    mass = float(model.body_mass.sum())
    per_hand = float(np.clip(frac, 0, 1)) * mass * 9.81 / 2.0
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    lh = data.geom_xpos[gid("left_hand_collision")]
    horiz = lambda a, b: float(np.linalg.norm((a - b)[:2]))
    tau_sh = per_hand * horiz(lh, data.xpos[bid("left_shoulder_pitch_link")])
    tau_el = per_hand * horiz(lh, data.xpos[bid("left_elbow_link")])
    cost += max(0.0, tau_sh - 15.0) ** 2
    cost += max(0.0, tau_el - 15.0) ** 2

    # (d) pelvis clear of the ground along the normal
    h = float(q[0:3] @ normal)
    cost += max(0.0, 0.28 - h) ** 2 * 200.0

    # (e) head up. Without this the search flattens the torso along the slope
    # (arms overhead at sho_p ~ -2.75) because a long reach happens to satisfy
    # (a)-(d) -- and the head ends up dragged along the ground. The head sits
    # ~0.4 m above torso_link along its z axis; ask that point to clear the
    # slope by HEAD_H.
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    head = data.xpos[tb] + data.xmat[tb].reshape(3, 3) @ np.array([0.0, 0.0, 0.4])
    head_h = float(head @ normal)
    cost += max(0.0, HEAD_H - head_h) ** 2 * 600.0

    # (f) SOLES FLAT on the plane, not edges. (a) only asks the box's lowest
    # point to touch, which a foot balanced on its toe edge satisfies -- and
    # an edge stance that is statically perfect tips over dynamically (a
    # head-up solve without this fell in 27-46 steps under a null policy).
    flat = 0.0
    for g in CONTACTS:
        Rg = data.geom_xmat[gid(g)].reshape(3, 3)
        flat += (1.0 - float(Rg[:, 2] @ normal)) ** 2
    cost += flat * 50.0
    return cost, dict(d=d, frac=frac, span=span, tau_sh=tau_sh, tau_el=tau_el,
                      h=h, head_h=head_h, flat=flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slope", type=float, default=37.5, help="degrees")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--frac-lo", type=float, default=0.30)
    ap.add_argument("--frac-hi", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head-h", type=float, default=0.0,
                    help="minimum head height above the slope, metres. "
                         "0 reproduces the old flat-torso stances.")
    ap.add_argument("--sho-lo", type=float, default=-3.05,
                    help="shoulder_pitch search floor; raise toward -2.0 "
                         "to keep the arms under the shoulders (bear "
                         "crawl) instead of overhead (superman)")
    ap.add_argument("--init", default=None,
                    help="npy of a previous parameter vector; refine locally "
                         "around it instead of searching globally. Used by "
                         "build_stances.py so stances at neighbouring angles "
                         "form ONE continuous family that can be interpolated "
                         "-- independent global searches land in different "
                         "local optima and cannot be blended.")
    args = ap.parse_args()

    global FRAC_LO, FRAC_HI, HEAD_H
    FRAC_LO, FRAC_HI = args.frac_lo, args.frac_hi
    HEAD_H = args.head_h
    slope = np.deg2rad(args.slope)
    from himalaya.env.base import get_assets
    text = (XMLS / "scene_mjx_climb_flat.xml").read_text().replace(
        '<geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>',
        f'<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" euler="0 {-slope:.6f} 0"/>')
    model = mujoco.MjModel.from_xml_string(text, assets=get_assets())
    data = mujoco.MjData(model)

    R = ry(-slope)
    normal = R @ np.array([0.0, 0.0, 1.0])
    uphill = R @ np.array([1.0, 0.0, 0.0])
    print(f"slope {args.slope:.1f} deg   normal {np.round(normal,3)}   uphill {np.round(uphill,3)}")
    print(f"mu needed to not slide: {np.tan(slope):.3f}")

    #        pitch  pz    px    sho_p  elbow  hip_p  knee  ank_p  waist_p
    lo = np.array([-0.2, 0.25, -0.5, args.sho_lo, -1.0, -2.45, 0.0, -0.85, -0.50])
    hi = np.array([1.60, 0.75, 0.5, 0.50, 2.09, 0.20, 2.85, 0.52, 0.50])
    rng = np.random.default_rng(args.seed)
    best, best_p = np.inf, None
    if args.init:
        best_p = np.clip(np.load(args.init), lo, hi)
        best, _ = evaluate(model, data, best_p, normal, uphill, slope)
    else:
        for _ in range(args.iters):
            p = rng.uniform(lo, hi)
            c, _ = evaluate(model, data, p, normal, uphill, slope)
            if c < best:
                best, best_p = c, p.copy()
    step = (hi - lo) * (0.02 if args.init else 0.08)
    for _ in range(args.iters):
        cand = np.clip(best_p + rng.normal(0, 1, len(lo)) * step, lo, hi)
        c, _ = evaluate(model, data, cand, normal, uphill, slope)
        if c < best:
            best, best_p = c, cand
        step *= 0.99985

    cost, info = evaluate(model, data, best_p, normal, uphill, slope)
    names = ["pelvis_pitch", "pelvis_z", "pelvis_x", "shoulder_pitch", "elbow",
             "hip_pitch", "knee", "ankle_pitch", "waist_pitch"]
    print(f"\ncost {cost:.6f}")
    for n, v in zip(names, best_p):
        print(f"  {n:15s} {v:+.4f}")
    print(f"\ncontact offsets from plane (want ~0): {np.round(info['d'],5)}")
    print(f"COM between limbs at {info['frac']*100:.0f}% (0=feet, 100=hands)   stance span {info['span']:.3f} m")
    print(f"shoulder {info['tau_sh']:.1f} Nm   elbow {info['tau_el']:.1f} Nm   (limit 25)")
    print(f"pelvis above plane {info['h']:.3f} m   head above plane {info['head_h']:.3f} m")

    q = build_qpos(best_p)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    shift = min(extreme_along(model, data, g, normal) for g in CONTACTS)
    q[0:3] -= shift * normal
    np.save(f"/tmp/quad_qpos_{args.slope:g}.npy", q)
    np.save(f"/tmp/quad_params_{args.slope:g}.npy", best_p)
    fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
    print(f"\nsaved /tmp/quad_qpos_{args.slope:g}.npy")
    print(f'qpos="{fmt(q[:3])}  {fmt(q[3:7])}  {fmt(q[7:36])}"')


if __name__ == "__main__":
    main()
