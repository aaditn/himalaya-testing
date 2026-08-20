#!/usr/bin/env python3
"""Pre-flight checks. Run before ANY long training job.

Three cheap tests that catch the expensive mistakes:
  1. zero-action   -- physics sanity, ~30s
  2. overfit-one   -- obs/action wiring sanity, ~5min
  3. reward-audit  -- per-term magnitude balance, ~1min

Usage:
    python scripts/preflight.py --config configs/g1_rough.yaml [--test all]

Exit code 0 = clear to launch. Non-zero = do not start the run.
"""
import argparse
import sys

import yaml


class Result:
    def __init__(self, name):
        self.name = name
        self.failures = []
        self.warnings = []

    def fail(self, msg):
        self.failures.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def passed(self):
        return not self.failures

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"[{status}] {self.name}")
        for w in self.warnings:
            print(f"       warn: {w}")
        for f in self.failures:
            print(f"       fail: {f}")


def test_zero_action(env, cfg):
    """Step the sim with zero actions. The robot should fall over plausibly.

    Catches: bad contact params, wrong timestep, broken collision meshes,
    robot spawned inside the ground plane.
    """
    r = Result("zero-action physics sanity")
    obs, _ = env.reset()
    zero = env.zero_actions()

    heights, max_speed, max_accel = [], 0.0, 0.0
    prev_vel = None
    for _ in range(int(2.0 * cfg["control"]["policy_hz"])):
        obs, _, _, _, info = env.step(zero)
        h = env.base_height()
        v = env.base_lin_vel()
        heights.append(h)
        max_speed = max(max_speed, abs(v).max())
        if prev_vel is not None:
            max_accel = max(max_accel, abs(v - prev_vel).max() * cfg["control"]["policy_hz"])
        prev_vel = v

    if not all(map(_finite, heights)):
        r.fail("non-finite base height -- sim diverged (NaN/inf)")
    if min(heights) < -0.5:
        r.fail(f"base fell to {min(heights):.2f}m -- robot sinking through ground")
    if max_speed > 50.0:
        r.fail(f"peak speed {max_speed:.1f} m/s -- physics explosion")
    if max_accel > 500.0:
        r.fail(f"peak accel {max_accel:.0f} m/s^2 -- contact jitter, check solver/timestep")
    if max(heights) - min(heights) < 0.02:
        r.warn("base barely moved under zero action -- is the robot actually free?")
    return r


def test_overfit_one(env, cfg, minutes=5):
    """Single env, flat, one fixed command. Tracking should climb fast.

    Catches: obs/action index mismatch, wrong action scale, reward sign errors,
    frozen or unmapped joints. If a policy cannot overfit ONE env on flat
    ground, nothing downstream will work.
    """
    r = Result("overfit single env")
    curve = env.train_single(minutes=minutes, command=(0.5, 0.0, 0.0))

    if len(curve) < 10:
        r.fail("too few iterations logged -- training loop not stepping")
        return r

    start = sum(curve[:5]) / 5
    end = sum(curve[-5:]) / 5
    if end <= start:
        r.fail(f"tracking reward did not improve ({start:.3f} -> {end:.3f}) "
               "-- suspect obs/action wiring or reward sign")
    elif end < 0.5:
        r.warn(f"tracking only reached {end:.3f} on a single flat env; "
               "expected >0.5. Check action_scale and PD gains.")

    moved = env.arm_joint_travel()
    if moved is not None and moved < 1e-3:
        r.warn("arm joints essentially static -- expected for the arms-free "
               "baseline, but confirms the AM term will likely be needed")
    return r


def test_reward_audit(env, cfg):
    """Roll out a random policy, log each reward term separately.

    Catches the single most common failure: one term dominating the sum.
    Invisible if you only ever look at total reward.
    """
    r = Result("reward term balance")
    terms = env.sample_reward_terms(steps=200)

    active = {k: v for k, v in terms.items() if abs(v) > 1e-9}
    if not active:
        r.fail("all reward terms are zero -- reward fn not wired")
        return r

    print("       per-term mean magnitude:")
    biggest = max(active.values(), key=abs)
    for k, v in sorted(active.items(), key=lambda kv: -abs(kv[1])):
        share = abs(v) / abs(biggest) if biggest else 0.0
        print(f"         {k:<20} {v:>10.4f}  ({share:5.1%} of largest)")

    tracking = abs(active.get("track_lin_vel", 0.0))
    for k, v in active.items():
        if k == "track_lin_vel":
            continue
        if tracking > 0 and abs(v) > 10 * tracking:
            r.fail(f"'{k}' is {abs(v)/tracking:.0f}x the velocity-tracking term "
                   "-- it will swamp the primary objective")
        elif tracking > 0 and abs(v) > 3 * tracking:
            r.warn(f"'{k}' is {abs(v)/tracking:.1f}x velocity tracking -- watch it")

    zeroed = [k for k, v in terms.items() if abs(v) <= 1e-9]
    if zeroed:
        r.warn(f"zero-valued terms (intentional?): {', '.join(zeroed)}")
    return r


def _finite(x):
    return x == x and abs(x) != float("inf")


TESTS = {
    "zero-action": test_zero_action,
    "overfit-one": test_overfit_one,
    "reward-audit": test_reward_audit,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1_rough.yaml")
    ap.add_argument("--test", default="all", choices=["all", *TESTS])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from himalaya.tasks.g1_rough import make_preflight_env  # noqa: PLC0415
    env = make_preflight_env(cfg)

    chosen = TESTS if args.test == "all" else {args.test: TESTS[args.test]}
    results = []
    for name, fn in chosen.items():
        print(f"\n--- {name} ---")
        results.append(fn(env, cfg))

    print("\n" + "=" * 52)
    for r in results:
        r.report()

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED -- do not launch the run.")
        return 1
    print("\nAll pre-flight checks passed. Clear to launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
