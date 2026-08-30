"""Generate a BANK of corridor terrains, one heightfield per variant.

    .venv/bin/python scripts/make_terrain_bank.py --n 16

Writes himalaya/env/xmls/assets/terrain_bank.npy of shape (n, nrow*ncol),
flattened exactly like mujoco's hfield_data, plus terrain_bank_meta.npz
recording each variant's corridor width and wall height.

WHY A BANK
----------
randomize.py can give every environment its own hfield: hfield_data is a jax
array and vmaps cleanly to (n_envs, 65536) through tree_replace -- verified,
despite make_route.py's comment claiming the MJX schema has no per-world
dimension for it. But generating terrain inside jax means rewriting a numpy
generator that already works. A bank sidesteps that: generate offline with the
existing code, load the stack, and index it per environment at reset.

WHY IT MATTERS
--------------
The single baked map gave the legs a median 2.35 m of corridor floor against a
0.24 m stance -- ten times the room they need, so the arms were never worth
using. These variants run 0.45-0.9 m, where the lateral margin drops to
0.11-0.33 m: at the narrow end that is less than one foot length, so a stumble
puts a foot off the floor and a hand on the wall is the only recovery.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "himalaya" / "env" / "xmls" / "assets"

_sspec = importlib.util.spec_from_file_location(
    "himalaya_scene", ROOT / "himalaya" / "env" / "scene.py")
_scene = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_scene)
SPAWN_X, SPAWN_Y = _scene.SPAWN
SPAWN_PAD = _scene.SPAWN_PAD

_spec = importlib.util.spec_from_file_location(
    "make_route", ROOT / "himalaya" / "env" / "terrain" / "make_route.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--z-scale", type=float, default=2.30,
                    help="must match the scene XML hfield size z")
    ap.add_argument("--width-min", type=float, default=0.45)
    ap.add_argument("--width-max", type=float, default=0.90)
    ap.add_argument("--wall-min", type=float, default=0.85)
    ap.add_argument("--wall-max", type=float, default=1.35)
    ap.add_argument("--base-relief", type=float, default=1.00,
                    help="height of the terrain the channels are cut into. "
                         "Above ~1.4 the map becomes mostly wall.")
    ap.add_argument("--rough-min", type=float, default=0.06)
    ap.add_argument("--rough-max", type=float, default=0.18)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    fields, widths, walls, roughs = [], [], [], []
    for i in range(a.n):
        w = float(rng.uniform(a.width_min, a.width_max))
        wall = float(rng.uniform(a.wall_min, a.wall_max))
        rough = float(rng.uniform(a.rough_min, a.rough_max))
        # CHANNEL_W is the flat floor the robot walks on -- the number that
        # decides whether the legs have room to spare.
        mr.CHANNEL_W = w
        mr.CHANNEL_DEPTH = wall
        # Base terrain height is NOT the z_scale budget. Setting it to the
        # full budget made the map 64% wall by area; the channel depth is what
        # provides bracing height, cut down from a modest surface.
        mr._PEAK_M = a.base_relief
        h, _ = mr.route(res=a.res, lane_w=max(w, 0.6), rough=rough,
                        lane_rough=rough * 0.8, seed=i)
        # SHIFT the map so a corridor passes through the spawn.
        #
        # One fixed SPAWN cannot serve terrains whose channels sit in different
        # places: measured before this, 4 of 16 variants put the robot on a
        # wall at ~1.0 m relief, and those episodes died on step 1. Rolling the
        # field laterally is free -- the terrain is statistically homogeneous
        # across y -- and guarantees every variant is spawnable.
        # Roll ROWS so a channel passes through the spawn.
        #
        # Channels run along COLUMNS (world x, up the slope) and vary in row, so
        # the free axis is the row axis: rolling rows slides the whole set of
        # corridors sideways without disturbing their up-slope continuity.
        sr = int((SPAWN_Y + 6.0) / 12.0 * a.res)
        sc_ = int((SPAWN_X + 6.0) / 12.0 * a.res)
        # Pick the roll that leaves the SPAWN CELL lowest, rather than rolling
        # the column's global minimum onto the spawn row. With four channels a
        # column has four minima, and the deepest is rarely the one nearest the
        # spawn -- measured, that put 13 of 16 variants on a wall.
        # Shift the field in BOTH axes so the spawn cell sits on a channel
        # floor at the global minimum height, not merely at the low point of a
        # column that happens to be high everywhere. Rolling rows alone left
        # 13 of 16 variants with 0.6 m of relief under the robot, because the
        # spawn's column was elevated along its whole length.
        flat = int(np.argmin(h))
        lo_r, lo_c = flat // a.res, flat % a.res
        h = np.roll(h, sr - lo_r, axis=0)
        h = np.roll(h, sc_ - lo_c, axis=1)
        # FLAT PAD under the spawn. The roll above puts the field's global
        # minimum at the spawn cell; this flattens a SPAWN_PAD-metre square
        # around it to that height, so the robot starts on level corridor
        # floor in every variant. Run L spawned on wall flanks and fell 29
        # times in 20 s; a check that the spawn is not on a wall can be fooled
        # (an upright robot ON TOP of a wall passes every posture test), but a
        # square that is flat by construction cannot.
        cell = 12.0 / a.res
        half = int(round(SPAWN_PAD / 2.0 / cell))
        floor = float(h[sr, sc_])
        r0, r1 = max(sr - half, 0), min(sr + half + 1, a.res)
        c0, c1 = max(sc_ - half, 0), min(sc_ + half + 1, a.res)
        h[r0:r1, c0:c1] = floor
        pad = h[r0:r1, c0:c1]
        assert np.ptp(pad) == 0.0, f"variant {i}: pad not flat, ptp={np.ptp(pad)}"
        assert h[sr, sc_] == floor, f"variant {i}: spawn cell off the pad"
        assert (r1 - r0) * cell >= SPAWN_PAD - 2 * cell, \
            f"variant {i}: pad clipped by the map edge"
        # Normalise to the z budget LAST. The carve digs below zero and adds to
        # the total range, so a field that measured 1.0 m before carving can
        # exceed the scene's z_scale afterwards and clip in the PNG.
        h = h - h.min()
        if h.max() > 0:
            h = h * (a.z_scale * 0.95 / h.max())
        # Store NORMALISED, the way mujoco reads hfield_data: 0..1 scaled by
        # the geom's z size. Writing metres here would silently multiply the
        # relief by z_scale again.
        fields.append((h / a.z_scale).astype(np.float32).ravel())
        widths.append(w); walls.append(wall); roughs.append(rough)

    bank = np.stack(fields)
    np.save((ASSETS / "terrain_bank.npy").as_posix(), bank)
    np.savez((ASSETS / "terrain_bank_meta.npz").as_posix(),
             width=np.array(widths), wall=np.array(walls),
             rough=np.array(roughs), z_scale=a.z_scale)
    print(f"wrote terrain_bank.npy  {bank.shape}  (n, nrow*ncol)")
    print(f"  corridor width {min(widths):.2f}-{max(widths):.2f} m")
    print(f"  wall depth     {min(walls):.2f}-{max(walls):.2f} m")
    print(f"  roughness      {min(roughs):.2f}-{max(roughs):.2f} m")
    print(f"  relief in metres: {bank.max()*a.z_scale:.2f} (z_scale {a.z_scale})")
    print(f"  spawn pad: {SPAWN_PAD:.2f} m square, flat in {a.n}/{a.n} variants")


if __name__ == "__main__":
    main()
