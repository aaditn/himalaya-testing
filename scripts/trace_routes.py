"""Trace the climbing corridors by RAY-CASTING the compiled scene.

    .venv/bin/python scripts/trace_routes.py --climb 15

Writes himalaya/env/xmls/assets/mountain_lines.npy: (n_routes, steps, 2) of
world x,y, which scene.route_lines() loads and the progress_uphill reward
follows.

WHY RAY-CASTS AND NOT THE PNG
-----------------------------
Every earlier version read mountain.png directly and indexed it as
r=(y+6)/12*res, c=(x+6)/12*res. That is the heightfield's LOCAL frame. The
floor geom is rotated about +Y by the slope angle, so local != world, and at 15
degrees the error is over a metre -- enough to trace corridors through banks.
Worse, the x and y errors are different, which rotates the traced line relative
to the terrain and is what produced a route pointing 90 degrees off the visible
corridor.

mj_ray fires at the compiled geometry, the same geometry the viewer draws and
the physics steps. It cannot disagree with what you see, which is the whole
point: the terrain, the viewer, and the reward now derive from one source.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import view as viewmod  # noqa: E402  (builds the tilted scene the same way)

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "himalaya_scene", ROOT / "himalaya" / "env" / "scene.py")
sc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def make_surface_fn(model, data):
    """World surface height at (x, y), or None off the map."""
    def surf(x, y):
        pnt = np.array([float(x), float(y), 30.0])
        vec = np.array([0.0, 0.0, -1.0])
        g = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, g)
        if dist is None or dist < 0:
            return None
        return 30.0 - dist
    return surf


CLIMB_W = 3.0   # how much a metre of height gained is worth against a metre
                # of relief. Higher = straighter up the fall line.


def trace(surf, slope_rad, extent, n_routes, steps, step_len, drift_bias,
          climb_w=None):
    """Walk each route up the hill, preferring low ground.

    At every step the candidate headings are those with an uphill component;
    the one whose landing point sits lowest ABOVE THE MEAN PLANE wins. Height
    above the plane, not raw world z, is what distinguishes a corridor from the
    hillside -- raw z always falls as you descend, so minimising it walks
    downhill.
    """
    half = extent / 2.0
    tan = np.tan(slope_rad)
    cw = CLIMB_W if climb_w is None else climb_w
    lines = []
    # Spread the starts across the downhill edge so the routes do not overlap.
    for r in range(n_routes):
        y = -half + extent * (r + 0.5) / n_routes
        x = half - 1.0                     # start low on the hill (large x)
        pts = [(x, y)]
        heading = np.pi                    # -x is uphill
        for _ in range(steps):
            best, best_h, best_p = None, None, None
            for dth in np.linspace(-1.2, 1.2, 25):
                th = heading + dth
                nx, ny = x + np.cos(th) * step_len, y + np.sin(th) * step_len
                if abs(nx) > half - 0.4 or abs(ny) > half - 0.4:
                    continue
                z = surf(nx, ny)
                if z is None:
                    continue
                # Height above the tilted mean plane at that point.
                rel = z - (-nx * tan)
                # Trade relief against PROGRESS explicitly.
                #
                # Minimising relief alone finds the flattest ground, which on a
                # hillside is sideways: measured, pure-relief routes sat at
                # 0.006-0.047 m (beautifully in the troughs) but climbed only
                # 0.14-0.36 m over the whole map, because a contour is the
                # flattest line there is. Penalising turns instead just made
                # them straight. What a route actually wants is to gain height
                # while staying low relative to the ground beside it, so the
                # cost pays for uphill progress and charges for relief.
                gain = (x - nx) * tan          # height gained on the plane
                cost = rel - cw * gain + drift_bias * abs(dth)
                if best is None or cost < best:
                    best, best_h, best_p = cost, th, (nx, ny)
            if best_p is None:
                break
            x, y = best_p
            heading = best_h
            pts.append((x, y))
        lines.append(pts)
    n = min(len(p) for p in lines)
    return np.array([p[:n] for p in lines])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climb", type=float, default=15.0)
    ap.add_argument("--routes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--step-len", type=float, default=0.12)
    ap.add_argument("--climb-w", type=float, default=3.0,
                    help="value of a metre climbed against a metre of relief; "
                         "higher = straighter up the fall line")
    ap.add_argument("--drift-bias", type=float, default=0.0,
                    help="metres of relief a turn must save to be worth it; "
                         "higher = straighter routes")
    a = ap.parse_args()

    model, slope = viewmod.build("mountain", a.climb)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surf = make_surface_fn(model, data)

    lines = trace(surf, slope, 12.0, a.routes, a.steps, a.step_len,
                  a.drift_bias, a.climb_w)
    out = ROOT / "himalaya" / "env" / "xmls" / "assets" / "mountain_lines.npy"
    np.save(out.as_posix(), lines)

    tan = np.tan(slope)
    up = sc.uphill(slope)[:2]
    up = up / np.linalg.norm(up)
    print(f"wrote {out.relative_to(ROOT)}  {lines.shape}")
    for r in range(lines.shape[0]):
        p = lines[r]
        rel = [surf(*q) - (-q[0] * tan) for q in p[::10]
               if surf(*q) is not None]
        climb = (-p[-1, 0] * tan) - (-p[0, 0] * tan)
        t = p[10] - p[0]
        direct = float(np.dot(t / (np.linalg.norm(t) + 1e-9), up))
        print(f"  route {r}: climbs {climb:+.2f} m   "
              f"mean relief {np.mean(rel):.3f} m   "
              f"directness {direct:.2f}   "
              f"spans y {p[:,1].min():+.1f} to {p[:,1].max():+.1f}")


if __name__ == "__main__":
    main()
