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
    """Extract the carved channels by RAY-CAST, column by column.

    Not a greedy walk. A walker will not step down into a 1.1 m trench -- it
    ran along the wall tops at 0.29-0.57 m relief instead of the 0.05 m floor.
    And the channel positions cannot be read back from make_route.py's grid
    indices without redoing the row/column mapping that has already produced a
    90-degree error twice today.

    So: sample each x-column, find the runs of low ground, and link runs
    between adjacent columns into continuous channels. Everything comes from
    mj_ray against the compiled geometry, so it cannot disagree with what the
    viewer draws.
    """
    half = extent / 2.0
    tan = np.tan(slope_rad)
    xs = np.linspace(-half + 0.5, half - 0.5, 90)
    ys = np.linspace(-half + 0.5, half - 0.5, 120)

    # Per column, the y-centres of each contiguous low-ground run.
    cols = []
    for x in xs:
        rel = []
        for y in ys:
            z = surf(x, y)
            rel.append(1e9 if z is None else z - (-x * tan))
        rel = np.array(rel)
        thr = 0.20
        runs, cur = [], []
        for k, r in enumerate(rel):
            if r < thr:
                cur.append(k)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        cols.append([float(np.mean(ys[np.array(run)])) for run in runs
                     if len(run) >= 2])

    # Link runs across columns, seeding from the column that has the most.
    # Seeding from column 0 finds nothing: the map edge is not where the
    # channels are widest, and a channel that has not started yet has no run
    # to seed from. Walk out in BOTH directions from the seed column.
    seed_j = max(range(len(cols)), key=lambda j: len(cols[j]))
    chans = []
    for seed in cols[seed_j]:
        # SKIP empty columns rather than stopping at them. 11 of 90 columns
        # have no run at all -- the channel passes under a bridge of terrain or
        # the sampling misses it -- and breaking on the first gap ended every
        # chain immediately.
        def walk(rng_):
            out_, y_, miss = [], seed, 0
            for j in rng_:
                if not cols[j]:
                    miss += 1
                    if miss > 3:
                        break
                    continue
                nxt = min(cols[j], key=lambda v: abs(v - y_))
                if abs(nxt - y_) > 0.6 * (miss + 1) + 0.6:
                    miss += 1
                    if miss > 3:
                        break
                    continue
                y_, miss = nxt, 0
                out_.append((xs[j], y_))
            return out_
        left = walk(range(seed_j - 1, -1, -1))
        right = walk(range(seed_j + 1, len(xs)))
        pts = left[::-1] + [(xs[seed_j], seed)] + right
        if len(pts) > 20:
            chans.append(pts)

    chans.sort(key=len, reverse=True)
    chans = chans[:n_routes]
    if not chans:
        raise RuntimeError("no channels found; is CHANNEL_DEPTH 0?")
    n = min(len(c) for c in chans)
    out = np.array([c[:n] for c in chans])
    # Order every channel so index 0 is the DOWNHILL end (large x), so the
    # tangent runs uphill.
    for r in range(out.shape[0]):
        if out[r, 0, 0] < out[r, -1, 0]:
            out[r] = out[r][::-1]
    return out


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
