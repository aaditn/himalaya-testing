#!/usr/bin/env python3
"""Fast contract and one-step physics checks before a long PPO run."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jp
import numpy as np

from himalaya.tasks.g1_cfg import (
    G1_ACTION_SIZE,
    G1_ACTOR_OBSERVATION_SIZE,
    G1_PRIVILEGED_OBSERVATION_SIZE,
)
from himalaya.tasks.himalaya_env_cfg import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", type=float, default=15.0)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    args = parser.parse_args()

    env = make_env(args.slope, noise_level=0.0, impl=args.impl)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    checks = {
        "action size": env.action_size == G1_ACTION_SIZE,
        "actor observation unchanged": (
            state.obs["state"].shape == (G1_ACTOR_OBSERVATION_SIZE,)
        ),
        "privileged slope descriptor": (
            state.obs["privileged_state"].shape
            == (G1_PRIVILEGED_OBSERVATION_SIZE,)
        ),
        "finite reset": bool(np.all(np.isfinite(np.asarray(state.data.qpos)))),
        "pushes disabled": not env._config.push_config.enable,
    }

    next_state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    checks["finite one-step state"] = bool(
        np.all(np.isfinite(np.asarray(next_state.data.qpos)))
    )
    checks["all requested diagnostics"] = all(
        key in next_state.metrics
        for key in (
            "validation/progress_m",
            "validation/uphill_speed_mps",
            "validation/planted_slip_mps",
            "validation/fall",
            "validation/com_height_m",
            "validation/peak_knee_torque_nm",
        )
    )
    checks["Stage-I ZMP reward"] = "reward/terrain_zmp" in next_state.metrics

    failed = []
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            failed.append(name)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
