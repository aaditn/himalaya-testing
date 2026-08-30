"""Deterministic policy validation on the five uniform ramp grades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import jax
import jax.numpy as jp
import numpy as np

from .tasks.himalaya_env_cfg import HimalayaG1UphillEnv


@dataclass(frozen=True)
class SlopeMetrics:
    slope_degrees: float
    trials: int
    successes: int
    success_rate: float
    mean_uphill_speed_mps: float
    mean_planted_slip_mps: float
    falls: int
    terminations: int
    mean_com_height_m: float
    peak_knee_torque_nm: float
    mean_progress_m: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_policy(
    env: HimalayaG1UphillEnv,
    inference_fn: Callable,
    *,
    trials: int = 64,
    seed: int = 0,
) -> SlopeMetrics:
    """Evaluate one policy without auto-resetting completed trials."""

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    batched_policy = jax.jit(jax.vmap(inference_fn))

    root_rng = jax.random.PRNGKey(seed)
    root_rng, reset_rng = jax.random.split(root_rng)
    state = reset(jax.random.split(reset_rng, trials))

    active = np.ones(trials, dtype=bool)
    successes = np.zeros(trials, dtype=bool)
    falls = np.zeros(trials, dtype=bool)
    terminations = np.zeros(trials, dtype=bool)
    final_progress = np.zeros(trials, dtype=np.float64)
    speed_sum = np.zeros(trials, dtype=np.float64)
    slip_sum = np.zeros(trials, dtype=np.float64)
    height_sum = np.zeros(trials, dtype=np.float64)
    sample_count = np.zeros(trials, dtype=np.int64)
    peak_knee_torque = np.zeros(trials, dtype=np.float64)

    for _ in range(env._config.episode_length):
        root_rng, action_rng = jax.random.split(root_rng)
        action_keys = jax.random.split(action_rng, trials)
        action = batched_policy(state.obs, action_keys)[0]
        action = jp.where(jp.asarray(active)[:, None], action, jp.zeros_like(action))
        state = step(state, action)

        progress = np.asarray(state.metrics["validation/progress_m"])
        success_now = np.asarray(state.metrics["validation/success"]) > 0.5
        fall_now = np.asarray(state.metrics["validation/fall"]) > 0.5
        done_now = np.asarray(state.done) > 0.5

        uphill_speed = np.asarray(
            state.metrics["validation/uphill_speed_mps"]
        )
        planted_slip = np.asarray(
            state.metrics["validation/planted_slip_mps"]
        )
        com_height = np.asarray(state.metrics["validation/com_height_m"])
        knee_torque = np.asarray(
            state.metrics["validation/peak_knee_torque_nm"]
        )

        speed_sum[active] += uphill_speed[active]
        slip_sum[active] += planted_slip[active]
        height_sum[active] += com_height[active]
        sample_count[active] += 1
        peak_knee_torque[active] = np.maximum(
            peak_knee_torque[active], knee_torque[active]
        )
        final_progress[active] = progress[active]

        successes |= active & success_now
        falls |= active & fall_now
        terminations |= active & done_now
        active &= ~(success_now | done_now)
        if not np.any(active):
            break

    denom = np.maximum(sample_count, 1)
    return SlopeMetrics(
        slope_degrees=env.slope_degrees,
        trials=trials,
        successes=int(np.sum(successes)),
        success_rate=float(np.mean(successes)),
        mean_uphill_speed_mps=float(np.mean(speed_sum / denom)),
        mean_planted_slip_mps=float(np.mean(slip_sum / denom)),
        falls=int(np.sum(falls)),
        terminations=int(np.sum(terminations)),
        mean_com_height_m=float(np.mean(height_sum / denom)),
        peak_knee_torque_nm=float(np.max(peak_knee_torque)),
        mean_progress_m=float(np.mean(final_progress)),
    )


def write_report(
    metrics: Iterable[SlopeMetrics], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.to_dict() for item in metrics]
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
