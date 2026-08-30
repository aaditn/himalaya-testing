"""INCOMPLETE -- does not currently produce a usable table. Kept for the finding.

Build a slope-indexed table of four-point stances that ACTUALLY HOLD.

STATUS: the search does not reliably find holding stances. Measured over 6
seeds per angle: at 30 deg only one seed reached 4/4 contacts (and at 2.24 m/s,
i.e. sliding, not holding); at 35 deg none did. Solving each angle independently
and warm-starting from a neighbour (continuation) both produced stances that
failed open-loop. The blocker is that the static cost is a weak proxy for
dynamic stability, so this needs a search over SIMULATED survival -- which is
too slow on CPU at the resolution required. Do not wire this into reset() until
every angle in the table is verified to hold.

Why this is not just "solve the cost function at each angle":

  The static cost (contacts coplanar, COM inside the support, shoulder/elbow
  torque under the real 25 Nm limit) has many minima, and most of them are
  dynamically unstable -- they satisfy every static condition and then fall
  over. Measured: solving each angle independently gave stances whose null-
  policy hold was 1/4, 0/4 and 0/4 contacts, while a different minimum at the
  SAME angle held 4/4 at 0.025 m/s. Warm-starting one angle from the previous
  (continuation) made it worse, not better.

  So candidates are SIMULATED and kept only if they hold. The selection
  criterion is survival, not cost.

Why a table rather than one reference stance:

  reset() rotates the spawn stance onto each environment's slope. That is exact
  for contact geometry but does not re-place the centre of mass against
  gravity. The 37.5 deg stance rotated to 45 deg collapses to 2/4 contacts,
  which is why the trained policy climbed cleanly to 40 deg and fell above it.

    .venv/bin/python scripts/build_stances.py
"""
import subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANGLES = [30.0, 35.0, 40.0, 45.0]
SEEDS = [0, 1, 2, 3, 4, 5]


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def holds(angle, qpos, seconds=3.0, friction=1.5):
    """Simulate the stance open-loop; return (n_contacts, residual speed)."""
    import mujoco, importlib.util
    spec = importlib.util.spec_from_file_location("ic", str(ROOT/"scripts"/"inspect_climb.py"))
    ic = importlib.util.module_from_spec(spec); spec.loader.exec_module(ic)
    slope = np.deg2rad(angle)
    model = ic.load(True, slope, friction)
    data = mujoco.MjData(model)
    ic.place(model, data, slope, qpos)
    data.ctrl[:] = qpos[7:]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    pair_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, n) for n in ic.GROUND_PAIRS}
    on = ic.contacts(model, data, pair_ids)
    return sum(on.values()), float(np.linalg.norm(data.qvel[0:3]))


def main():
    poses, chosen = [], []
    for a in ANGLES:
        best = None
        for seed in SEEDS:
            subprocess.run([sys.executable, str(ROOT/"scripts"/"find_quadruped_pose.py"),
                            "--slope", str(a), "--frac-lo", "0.15", "--frac-hi", "0.35",
                            "--iters", "20000", "--seed", str(seed)],
                           check=True, capture_output=True)
            q = np.load(f"/tmp/quad_qpos_{a:g}.npy")
            n, speed = holds(a, q)
            print(f"  {a:5.1f} deg seed {seed}: {n}/4 contacts, {speed:.3f} m/s")
            if n == 4 and (best is None or speed < best[1]):
                best = (q.copy(), speed, seed)
            if best and best[1] < 0.05:
                break
        if best is None:
            raise SystemExit(f"no holding stance found at {a} deg")
        q, speed, seed = best
        print(f"  -> {a} deg: seed {seed}, {speed:.3f} m/s")
        th = np.deg2rad(a)
        pos_sf = ry(th) @ q[0:3]
        quat_sf = quat_mul(np.array([np.cos(th/2), 0, np.sin(th/2), 0]), q[3:7])
        poses.append(np.concatenate([pos_sf, quat_sf, q[7:36]]))
        chosen.append(seed)
    out = ROOT/"himalaya"/"env"/"climb_stances.npz"
    np.savez(out, angles_deg=np.array(ANGLES), poses=np.array(poses))
    print(f"\nwrote {out}  seeds {chosen}")


if __name__ == "__main__":
    main()
