"""Warm-started PPO curriculum for the Himalaya G1 uphill environment."""

from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path
import time
from typing import Callable

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict

from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from .evaluation import SlopeMetrics, evaluate_policy, write_report
from .tasks.g1_cfg import CURRICULUM_SLOPES_DEG, validate_slope
from .tasks.himalaya_env_cfg import default_config, HimalayaG1UphillEnv


def ppo_config(num_timesteps: int = 40_000_000) -> config_dict.ConfigDict:
    """Use DeepMind's tuned G1 PPO settings, changing only stage duration."""

    cfg = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    cfg.num_timesteps = int(num_timesteps)
    return cfg


def latest_checkpoint(path: str | Path) -> Path:
    root = Path(path)
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"no numeric PPO checkpoints found under {root}")
    return max(candidates, key=lambda p: int(p.name))


def train_stage(
    slope_degrees: float,
    output_dir: Path,
    *,
    restore_checkpoint: Path | None,
    num_timesteps: int,
    seed: int,
    impl: str,
    num_envs: int | None,
) -> tuple[Callable, Path]:
    """Train one fixed-grade stage and return deterministic inference."""

    env_cfg = default_config()
    with env_cfg.unlocked():
        env_cfg.slope_degrees = validate_slope(slope_degrees)
        env_cfg.impl = impl
    env = HimalayaG1UphillEnv(config=env_cfg)

    params = ppo_config(num_timesteps)
    if num_envs is not None:
        params.num_envs = int(num_envs)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **params.network_factory
    )
    training_params = dict(params)
    del training_params["network_factory"]

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(step: int, metrics: dict[str, float]) -> None:
        reward = metrics.get("eval/episode_reward", float("nan"))
        elapsed = time.monotonic() - started
        print(
            f"slope={slope_degrees:g} step={step:,} "
            f"reward={reward:.3f} elapsed={elapsed / 60.0:.1f}m",
            flush=True,
        )

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=seed,
        restore_checkpoint_path=restore_checkpoint,
        save_checkpoint_path=checkpoint_dir,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=params.get("num_eval_envs", 128),
    )
    make_inference_fn, trained_params, _ = train_fn(
        environment=env,
        eval_env=env,
        progress_fn=progress,
    )
    inference_fn = make_inference_fn(trained_params, deterministic=True)
    return inference_fn, latest_checkpoint(checkpoint_dir)


def load_inference_fn(
    env: HimalayaG1UphillEnv, checkpoint: str | Path, *, seed: int = 1
) -> Callable:
    """Restore a deterministic Brax PPO policy without invoking PPO training."""

    del seed

    checkpoint_path = Path(checkpoint).resolve()
    if checkpoint_path.is_dir() and not checkpoint_path.name.isdigit():
        checkpoint_path = latest_checkpoint(checkpoint_path)
    params = ppo_config(num_timesteps=0)
    network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        **params.network_factory,
    )
    restored_params = ppo_checkpoint.load(checkpoint_path)
    make_inference_fn = ppo_networks.make_inference_fn(network)
    return make_inference_fn(restored_params, deterministic=True)


def validate_stage(
    slope_degrees: float,
    inference_fn: Callable,
    *,
    trials: int,
    seed: int,
    impl: str,
) -> SlopeMetrics:
    cfg = default_config()
    with cfg.unlocked():
        cfg.slope_degrees = validate_slope(slope_degrees)
        cfg.noise_config.level = 0.0
        cfg.command_stand_probability = 0.0
        cfg.impl = impl
    env = HimalayaG1UphillEnv(config=cfg)
    return evaluate_policy(env, inference_fn, trials=trials, seed=seed)


def _write_csv(metrics: list[SlopeMetrics], output: Path) -> None:
    if not metrics:
        return
    rows = [m.to_dict() for m in metrics]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_curriculum(args: argparse.Namespace) -> int:
    target = validate_slope(args.target_slope)
    slopes = [s for s in CURRICULUM_SLOPES_DEG if s <= target]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    restore = Path(args.restore).resolve() if args.restore else None
    all_metrics: list[SlopeMetrics] = []

    for index, slope in enumerate(slopes):
        stage_dir = output / f"stage_{index:02d}_{slope:g}deg"
        print(f"\n=== curriculum stage {index}: {slope:g} degrees ===")
        inference_fn, restore = train_stage(
            slope,
            stage_dir,
            restore_checkpoint=restore,
            num_timesteps=args.timesteps_per_stage,
            seed=args.seed + index,
            impl=args.impl,
            num_envs=args.num_envs,
        )
        metrics = validate_stage(
            slope,
            inference_fn,
            trials=args.validation_trials,
            seed=args.seed + 10_000 + index,
            impl=args.impl,
        )
        all_metrics.append(metrics)
        write_report(all_metrics, output / "validation.json")
        _write_csv(all_metrics, output / "validation.csv")
        print(metrics)

        if metrics.success_rate < args.promotion_success_rate:
            print(
                f"stage held at {slope:g} degrees: success rate "
                f"{metrics.success_rate:.1%} is below "
                f"{args.promotion_success_rate:.1%}"
            )
            return 2

    print(f"curriculum complete through {target:g} degrees")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the MuJoCo-only Himalaya G1 uphill curriculum."
    )
    parser.add_argument("--output", default="runs/g1_uphill_stage1")
    parser.add_argument("--target-slope", type=float, default=15.0)
    parser.add_argument(
        "--timesteps-per-stage", type=int, default=40_000_000
    )
    parser.add_argument("--validation-trials", type=int, default=64)
    parser.add_argument(
        "--promotion-success-rate", type=float, default=0.90
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--restore")
    return parser


def main() -> None:
    raise SystemExit(run_curriculum(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
