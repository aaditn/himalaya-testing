"""Deterministic validation for the four-contact slope curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import jax
import jax.numpy as jp
import numpy as np

from .tasks.four_contact_env_cfg import HimalayaG1FourContactEnv


@dataclass(frozen=True)
class FourContactMetrics:
    slope_degrees: float
    trials: int
    successes: int
    success_rate: float
    mean_progress_m: float
    mean_traversal_time_s: float | None
    mean_uphill_speed_mps: float
    left_hand_contact_ratio: float
    right_hand_contact_ratio: float
    four_contact_ratio: float
    alternating_hand_contact_ratio: float
    continuous_double_hand_contact_ratio: float
    mean_hand_slip_mps: float
    mean_foot_slip_mps: float
    mean_hand_load_share: float
    leg_propulsion_fraction: float
    mean_leg_positive_work_j: float
    mean_arm_positive_work_j: float
    mean_waist_positive_work_j: float
    peak_hand_force_n: float
    peak_wrist_moment_nm: float
    peak_wrist_actuator_torque_nm: float
    peak_leg_torque_nm: float
    peak_ankle_torque_nm: float
    peak_actuator_saturation_ratio: float
    actuator_saturation_ratio: float
    falls: int
    prohibited_body_contact_terminations: int
    nonfinite_terminations: int
    other_terminations: int
    terminations: int
    mean_com_height_m: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_four_contact_policy(
    env: HimalayaG1FourContactEnv,
    inference_fn: Callable,
    *,
    trials: int = 64,
    seed: int = 0,
    max_hand_slip_mps: float | None = None,
    max_foot_slip_mps: float | None = None,
) -> FourContactMetrics:
    """Success requires stable, safe uphill progress on rough terrain.

    Hand-use, alternation, and four-contact occupancy remain diagnostics, not
    promotion gates.  This lets terrain mechanics select the useful gait.
    """

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    policy = jax.jit(jax.vmap(inference_fn))
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset(jax.random.split(reset_rng, trials))

    active = np.ones(trials, dtype=bool)
    successes = np.zeros(trials, dtype=bool)
    falls = np.zeros(trials, dtype=bool)
    terminations = np.zeros(trials, dtype=bool)
    prohibited = np.zeros(trials, dtype=bool)
    nonfinite = np.zeros(trials, dtype=bool)
    progress = np.zeros(trials)
    traversal_time = np.full(trials, np.nan)
    sums = {name: np.zeros(trials) for name in (
        "speed", "left", "right", "four", "alternating", "double",
        "hand_slip", "foot_slip", "load", "height", "leg_power",
        "arm_power", "waist_power", "saturated",
    )}
    count = np.zeros(trials, dtype=np.int64)
    peaks = {name: np.zeros(trials) for name in (
        "hand_force", "wrist_moment", "wrist_actuator", "leg_torque",
        "ankle_torque", "saturation",
    )}

    for _ in range(env._config.episode_length):
        rng, action_rng = jax.random.split(rng)
        action = policy(state.obs, jax.random.split(action_rng, trials))[0]
        action = jp.where(jp.asarray(active)[:, None], action, jp.zeros_like(action))
        state = step(state, action)
        metrics = {k: np.asarray(v) for k, v in state.metrics.items()}

        progress[active] = metrics["validation/progress_m"][active]
        sums["speed"][active] += metrics["validation/uphill_speed_mps"][active]
        sums["left"][active] += metrics["validation/hand_contact_left"][active]
        sums["right"][active] += metrics["validation/hand_contact_right"][active]
        sums["four"][active] += metrics["validation/four_contact"][active]
        sums["alternating"][active] += metrics[
            "validation/alternating_hand_contact"
        ][active]
        sums["double"][active] += metrics[
            "validation/double_hand_contact"
        ][active]
        sums["hand_slip"][active] += metrics["validation/hand_slip_mps"][active]
        sums["foot_slip"][active] += metrics["validation/foot_slip_mps"][active]
        sums["load"][active] += metrics["validation/hand_load_share"][active]
        sums["height"][active] += metrics["validation/com_height_m"][active]
        sums["leg_power"][active] += metrics[
            "validation/leg_positive_power_w"
        ][active]
        sums["arm_power"][active] += metrics[
            "validation/arm_positive_power_w"
        ][active]
        sums["waist_power"][active] += metrics[
            "validation/waist_positive_power_w"
        ][active]
        sums["saturated"][active] += metrics[
            "validation/actuator_saturated"
        ][active]
        count[active] += 1
        peaks["hand_force"][active] = np.maximum(
            peaks["hand_force"][active], metrics["validation/peak_hand_force_n"][active]
        )
        peaks["wrist_moment"][active] = np.maximum(
            peaks["wrist_moment"][active], metrics["validation/peak_wrist_moment_nm"][active]
        )
        peaks["wrist_actuator"][active] = np.maximum(
            peaks["wrist_actuator"][active],
            metrics["validation/peak_wrist_actuator_torque_nm"][active],
        )
        peaks["leg_torque"][active] = np.maximum(
            peaks["leg_torque"][active], metrics["validation/peak_leg_torque_nm"][active]
        )
        peaks["ankle_torque"][active] = np.maximum(
            peaks["ankle_torque"][active], metrics["validation/peak_ankle_torque_nm"][active]
        )
        peaks["saturation"][active] = np.maximum(
            peaks["saturation"][active],
            metrics["validation/actuator_saturation_ratio"][active],
        )
        denom = np.maximum(count, 1)
        elapsed = denom * env.dt
        mean_speed = progress / elapsed
        hand_slip_limit = (
            env._config.validation_max_hand_slip_mps
            if max_hand_slip_mps is None else max_hand_slip_mps
        )
        foot_slip_limit = (
            env._config.validation_max_foot_slip_mps
            if max_foot_slip_mps is None else max_foot_slip_mps
        )
        success_now = (
            progress >= env._config.validation_success_distance_m
        ) & (
            mean_speed >= env._config.validation_min_speed_mps
        ) & (
            mean_speed <= env._config.validation_max_speed_mps
        ) & (
            sums["hand_slip"] / denom <= hand_slip_limit
        ) & (
            sums["foot_slip"] / denom <= foot_slip_limit
        ) & (
            peaks["hand_force"] <= env._config.reward_config.max_hand_force
        ) & (
            peaks["wrist_moment"] <= env._config.reward_config.max_wrist_moment
        )
        done_now = np.asarray(state.done) > 0.5
        fall_now = metrics["validation/fall"] > 0.5
        prohibited_now = metrics["validation/prohibited_body_contact"] > 0.5
        nonfinite_now = metrics["validation/nonfinite"] > 0.5
        newly_successful = active & success_now & ~successes
        traversal_time[newly_successful] = count[newly_successful] * env.dt
        successes |= active & success_now
        falls |= active & fall_now
        prohibited |= active & prohibited_now
        nonfinite |= active & nonfinite_now
        terminations |= active & done_now
        active &= ~(success_now | done_now)
        if not np.any(active):
            break

    denom = np.maximum(count, 1)
    total_leg_work = sums["leg_power"] * env.dt
    total_arm_work = sums["arm_power"] * env.dt
    total_waist_work = sums["waist_power"] * env.dt
    propulsion_fraction = total_leg_work / (
        total_leg_work + total_arm_work + 1.0e-6
    )
    successful_times = traversal_time[np.isfinite(traversal_time)]
    classified = falls | prohibited | nonfinite
    return FourContactMetrics(
        slope_degrees=env.slope_degrees,
        trials=trials,
        successes=int(successes.sum()),
        success_rate=float(successes.mean()),
        mean_progress_m=float(progress.mean()),
        mean_traversal_time_s=(
            float(successful_times.mean()) if successful_times.size else None
        ),
        mean_uphill_speed_mps=float(np.mean(sums["speed"] / denom)),
        left_hand_contact_ratio=float(np.mean(sums["left"] / denom)),
        right_hand_contact_ratio=float(np.mean(sums["right"] / denom)),
        four_contact_ratio=float(np.mean(sums["four"] / denom)),
        alternating_hand_contact_ratio=float(
            np.mean(sums["alternating"] / denom)
        ),
        continuous_double_hand_contact_ratio=float(
            np.mean(sums["double"] / denom)
        ),
        mean_hand_slip_mps=float(np.mean(sums["hand_slip"] / denom)),
        mean_foot_slip_mps=float(np.mean(sums["foot_slip"] / denom)),
        mean_hand_load_share=float(np.mean(sums["load"] / denom)),
        leg_propulsion_fraction=float(np.mean(propulsion_fraction)),
        mean_leg_positive_work_j=float(np.mean(total_leg_work)),
        mean_arm_positive_work_j=float(np.mean(total_arm_work)),
        mean_waist_positive_work_j=float(np.mean(total_waist_work)),
        peak_hand_force_n=float(peaks["hand_force"].max()),
        peak_wrist_moment_nm=float(peaks["wrist_moment"].max()),
        peak_wrist_actuator_torque_nm=float(peaks["wrist_actuator"].max()),
        peak_leg_torque_nm=float(peaks["leg_torque"].max()),
        peak_ankle_torque_nm=float(peaks["ankle_torque"].max()),
        peak_actuator_saturation_ratio=float(peaks["saturation"].max()),
        actuator_saturation_ratio=float(np.mean(sums["saturated"] / denom)),
        falls=int(falls.sum()),
        prohibited_body_contact_terminations=int(prohibited.sum()),
        nonfinite_terminations=int(nonfinite.sum()),
        other_terminations=int((terminations & ~classified).sum()),
        terminations=int(terminations.sum()),
        mean_com_height_m=float(np.mean(sums["height"] / denom)),
    )


def write_four_contact_report(
    metrics: Iterable[FourContactMetrics], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.to_dict() for item in metrics], indent=2) + "\n",
        encoding="utf-8",
    )
