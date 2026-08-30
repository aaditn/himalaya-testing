"""Design the mountain. Change a number, see the mountain. No pod, no training.

macOS needs mjpython, NOT python: the viewer must own the main thread.

    .venv/bin/mjpython scripts/map.py                       # look at it
    .venv/bin/mjpython scripts/map.py --wall-h 1.1          # taller walls
    .venv/bin/mjpython scripts/map.py --lane-w 1.8 --seed 3
    .venv/bin/mjpython scripts/map.py --robot               # drop the G1 in
    .venv/bin/mjpython scripts/map.py --stats               # numbers, no window

Every flag regenerates the heightfield and rebuilds the scene before the
window opens, so the loop is: edit the command, press up-arrow, enter.

WHAT THE KNOBS DO

  --lane-w      walkable corridor width, metres. The lane is where the feet go.
  --wall-h      how far the banks rise above the lane. Reach is ~0.34 m from
                the shoulder, so above ~0.9 m the hands cannot use the top.
  --wall-w      how wide the rise is spread. THIS is what makes a bank a bank:
                the same 0.85 m over 1.6 m is leanable, over 0.9 m it is a
                cliff you fall off.
  --rough       bump amplitude on the banks and outer ground.
  --lane-rough  bump amplitude in the lane. This is the footing the legs have
                to adapt to; 0 is a dead-flat floor that teaches nothing.
  --wall-steep  profile exponent. ABOVE 1 eases out of the lane and steepens;
                BELOW 1 goes vertical at the lane edge and gives you blades.
  --blur        final smoothing passes over the whole field, in cells. 1 kills
                single-cell spikes; 2 also flattens the corridor.
  --peak        total relief, metres. Must stay under the scene's hfield
                z_scale (1.10) or the PNG clips and the summits become mesas.
  --routes      how many separate corridors share the field.
  --seed        a different mountain with the same character.

WHAT TO LOOK FOR

  lane face angle   the slope underfoot INSIDE the corridor. This adds to the
                    base tilt: 9 deg of lane on a 35 deg slope means a foot
                    meets anywhere from 26 to 44 deg. Above ~12 the unlucky
                    direction exceeds what friction holds (mu >= tan(angle)).
  >45 / >60 deg     how much of the map is too steep to stand on at all.
  contrast          height difference across a cross-section. Under ~0.4 m the
                    corridor stops reading as a corridor.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
XMLS = ROOT / "himalaya" / "env" / "xmls"
PNG = XMLS / "assets" / "mountain.png"

# Import the generator WITHOUT importing himalaya.env, which would pull in jax
# and the whole training stack for what is a numpy problem.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "make_route", ROOT / "himalaya" / "env" / "terrain" / "make_route.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def measure(h, extent=12.0):
    cell = extent / h.shape[0]
    gy, gx = np.gradient(h, cell)
    face = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    lap = np.abs(
        np.gradient(np.gradient(h, cell, axis=0), cell, axis=0)
        + np.gradient(np.gradient(h, cell, axis=1), cell, axis=1))
    row = h[h.shape[0] // 2]
    return {
        "relief": h.max(),
        "mean_h": h.mean(),
        "face_med": np.percentile(face, 50),
        "face_p99": np.percentile(face, 99),
        "over45": 100 * (face > 45).mean(),
        "over60": 100 * (face > 60).mean(),
        "spike_p99": np.percentile(lap, 99),
        "contrast": row.max() - row.min(),
    }


def report(h, stats, m, base_deg):
    print()
    print(f"  lane face angle   {stats['lane_face']:5.1f} deg"
          f"   -> underfoot {base_deg - stats['lane_face']:.0f}-"
          f"{base_deg + stats['lane_face']:.0f} deg on a {base_deg:.0f} deg slope")
    print(f"  lane covers       {stats['lane_frac']:5.1f} % of the map")
    print(f"  corridor contrast {m['contrast']:5.2f} m"
          f"   {'(too flat to read as a route)' if m['contrast'] < 0.4 else ''}")
    print(f"  relief            {m['relief']:5.2f} m   mean {m['mean_h']:.2f} m")
    print(f"  too steep to stand  >45 deg {m['over45']:4.1f} %"
          f"   >60 deg {m['over60']:4.1f} %")
    print(f"  spikiness (p99)   {m['spike_p99']:5.1f}"
          f"   {'(SPIKY -- raise --blur or --wall-steep)' if m['spike_p99'] > 20 else ''}")
    print()
    # Three cuts, not one: a single row can land between corridors and show a
    # flat floor on a map that has four of them.
    for frac, label in ((0.25, "lower"), (0.5, "middle"), (0.75, "upper")):
        row = h[int(h.shape[0] * frac)]
        lo, span = row.min(), max(row.max() - row.min(), 1e-6)
        print(f"  cross-section, {label} third (left-right across the slope):")
        for band in range(5, -1, -1):
            line = "".join("#" if (v - lo) / span * 6 >= band else " "
                           for v in row[::4])
            print(f"    {lo + span * band / 6:5.2f} |{line}")
        print("           " + "-" * len(row[::4]))
    print()


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--lane-w", type=float, default=1.4)
    ap.add_argument("--wall-h", type=float, default=0.85)
    ap.add_argument("--wall-w", type=float, default=1.6)
    ap.add_argument("--rough", type=float, default=0.16)
    ap.add_argument("--lane-rough", type=float, default=0.14)
    ap.add_argument("--wall-steep", type=float, default=2.1)
    ap.add_argument("--blur", type=int, default=1)
    ap.add_argument("--peak", type=float, default=1.05)
    ap.add_argument("--routes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--despike", type=int, default=2,
                    help="median-filter passes: removes spikes WITHOUT "
                         "flattening walls (a blur would do both)")
    ap.add_argument("--lane-grain", default="coarse", choices=["coarse", "fine"],
                    help="coarse = bumps wider than a foot; fine = the original "
                         "res/4 texture, bumps smaller than a foot")
    ap.add_argument("--climb", type=float, default=35.0,
                    help="base tilt to view it at")
    ap.add_argument("--robot", action="store_true",
                    help="put the G1 on the map for scale")
    ap.add_argument("--pose", default="all_fours",
                    choices=["knees_bent", "home", "all_fours"])
    ap.add_argument("--stats", action="store_true",
                    help="print the numbers and exit, no window")
    ap.add_argument("--keep", action="store_true",
                    help="write the PNG (default: also writes it)")
    a = ap.parse_args()

    # The generator reads these two as module-level knobs.
    mr._PEAK_M = a.peak
    mr._WALL_STEEP = a.wall_steep
    mr._BLUR_K = a.blur
    mr._LANE_GRAIN = a.lane_grain
    mr._DESPIKE_PASSES = a.despike

    h, stats = mr.write_png(
        PNG, z_scale=1.10,
        lane_w=a.lane_w, wall_h=a.wall_h, wall_w=a.wall_w,
        rough=a.rough, lane_rough=a.lane_rough,
        n_routes=a.routes, seed=a.seed)
    m = measure(h)

    print(f"wrote {PNG.relative_to(ROOT)}")
    report(h, stats, m, a.climb)

    if a.stats:
        return

    cmd = [str(ROOT / ".venv" / "bin" / "mjpython"),
           str(ROOT / "scripts" / "view.py"), "--climb", str(a.climb)]
    if a.robot:
        cmd += ["--pose", a.pose]
    else:
        cmd += ["--terrain-only"]
    print("  opening viewer:", " ".join(cmd[1:]))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
