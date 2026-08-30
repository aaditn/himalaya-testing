#!/usr/bin/env python3
"""Compile and verify the reviewed 30/35-degree crawl contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import mjx_env

from himalaya.env import joystick as g1_joystick
from himalaya.tasks.four_contact_env_cfg import make_four_contact_env
from himalaya.tasks.g1_cfg import (
    FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,
    FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,
    G1_ACTION_SIZE,
)


ACTION_ORDER = tuple(
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "hip_pitch", "hip_roll", "hip_yaw", "knee",
            "ankle_pitch", "ankle_roll",
        )
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
    ]
)


def _stock_hand_envelope_matches(env, stock) -> bool:
    fields = ("geom_pos", "geom_quat", "geom_size", "geom_type")
    for name in ("left_hand_collision", "right_hand_collision"):
        env_id = env.mj_model.geom(name).id
        stock_id = stock.mj_model.geom(name).id
        for field in fields:
            if not np.allclose(
                np.asarray(getattr(env.mj_model, field)[env_id]),
                np.asarray(getattr(stock.mj_model, field)[stock_id]),
            ):
                return False
    return True


def _pair_friction(env, name: str) -> np.ndarray:
    return np.asarray(env.mj_model.pair_friction[env.mj_model.pair(name).id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    mjx_env.ensure_menagerie_exists()
    stock_cfg = g1_joystick.default_config()
    with stock_cfg.unlocked():
        stock_cfg.impl = "jax"
        stock_cfg.noise_config.level = 0.0
    stock = g1_joystick.Joystick(task="flat_terrain", config=stock_cfg)

    checks: dict[str, bool] = {}
    for slope in (30.0, 35.0):
        env = make_four_contact_env(slope, noise_level=0.0, impl="jax")
        label = f"{slope:g}deg"
        expected_quat = np.array([
            math.cos(math.radians(slope) / 2), 0.0,
            -math.sin(math.radians(slope) / 2), 0.0,
        ])
        floor_id = env.mj_model.geom("floor").id
        action_order = tuple(
            env.mj_model.actuator(index).name
            for index in range(env.mj_model.nu)
        )
        checks.update({
            f"{label} action order": action_order == ACTION_ORDER,
            f"{label} action size": env.action_size == G1_ACTION_SIZE,
            f"{label} terrain transform": np.allclose(
                np.asarray(env.mj_model.geom_quat[floor_id]), expected_quat
            ),
            f"{label} stock hand envelope": _stock_hand_envelope_matches(env, stock),
            f"{label} nominal hand friction": np.allclose(
                _pair_friction(env, "left_hand_floor")[:2], [0.9, 0.9]
            ) and np.allclose(
                _pair_friction(env, "right_hand_floor")[:2], [0.9, 0.9]
            ),
            f"{label} crampon foot friction": np.allclose(
                _pair_friction(env, "left_foot_floor")[:2], [1.0, 1.0]
            ) and np.allclose(
                _pair_friction(env, "right_foot_floor")[:2], [1.0, 1.0]
            ),
        })
        # The two grades share one shape-identical runtime. Compile a reset/step
        # once at 30 degrees; retain static XML/config checks for both grades.
        if slope == 30.0 and not args.static_only:
            state = jax.jit(env.reset)(jax.random.PRNGKey(30))
            next_state = jax.jit(env.step)(state, jp.zeros(env.action_size))
            checks.update({
                "compiled actor observations": state.obs["state"].shape
                == (FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,),
                "compiled critic observations": state.obs["privileged_state"].shape
                == (FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,),
                "compiled actor is privileged prefix only": np.allclose(
                    np.asarray(state.obs["state"]),
                    np.asarray(state.obs["privileged_state"][:103]),
                ),
                "compiled finite reset": bool(
                    np.isfinite(np.asarray(state.data.qpos)).all()
                ),
                "compiled finite step": bool(
                    np.isfinite(np.asarray(next_state.data.qpos)).all()
                ),
            })
        required_diagnostics = (
            "hand_contact_left", "hand_contact_right", "double_hand_contact",
            "hand_slip_mps", "foot_slip_mps", "leg_positive_power_w",
            "arm_positive_power_w", "waist_positive_power_w",
            "leg_propulsion_fraction", "peak_hand_force_n",
            "peak_wrist_moment_nm", "peak_leg_torque_nm",
            "peak_ankle_torque_nm", "actuator_saturation_ratio",
        )
        if slope == 30.0 and not args.static_only:
            checks["compiled active diagnostics"] = all(
                f"validation/{name}" in next_state.metrics
                for name in required_diagnostics
            )
            checks["compiled stable metric dtypes"] = all(
                np.issubdtype(np.asarray(value).dtype, np.floating)
                for key, value in next_state.metrics.items()
                if key.startswith("validation/")
            )
            checks["compiled finite diagnostics"] = all(
                np.isfinite(np.asarray(value)).all()
                for key, value in next_state.metrics.items()
                if key.startswith("validation/")
            )

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    report = {
        "schema_version": 1,
        "passed": all(bool(value) for value in checks.values()),
        "duration_seconds": time.monotonic() - started,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "compiled_slopes_degrees": [] if args.static_only else [30.0],
        "statically_checked_slopes_degrees": [30.0, 35.0],
        "dynamic_coverage_delegated_to_smoke": args.static_only,
        "checks": {name: bool(value) for name, value in checks.items()},
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
