"""Train and validate 30-degree rough-terrain four-contact balance."""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
import time
from typing import Callable

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from .four_contact_evaluation import (
    FourContactMetrics,
    evaluate_four_contact_policy,
    write_four_contact_report,
)
from .tasks.four_contact_env_cfg import (
    HimalayaG1FourContactEnv,
    default_four_contact_config,
)
from .tasks.g1_cfg import (
    FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,
    FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,
    G1_ACTION_SIZE,
    validate_four_contact_slope,
)


def ppo_config(num_timesteps: int = 40_000_000):
    """Retain DeepMind's tuned G1 PPO settings and network sizes."""

    cfg = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    cfg.num_timesteps = int(num_timesteps)
    return cfg


def latest_checkpoint(path: str | Path) -> Path:
    root = Path(path)
    candidates = [
        p for p in root.rglob("*") if p.is_dir() and p.name.isdigit()
    ]
    if root.is_dir() and root.name.isdigit():
        candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"no numeric PPO checkpoints found under {root}")
    return max(candidates, key=lambda p: (int(p.name), p.stat().st_mtime_ns))


def resolve_restore_checkpoint(path: str | Path, *, expected_slope: float) -> Path:
    """Resolve a relocatable stage artifact and verify its provenance."""

    root = Path(path).resolve()
    search_root = root if root.is_dir() and not root.name.isdigit() else root.parent
    manifests = sorted(search_root.rglob("stage_result.json"))
    matching = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            float(payload.get("slope_degrees", -1.0)) == expected_slope
            and payload.get("promotion_passed") is True
            and payload.get("action_size") == G1_ACTION_SIZE
            and payload.get("actor_observation_size")
            == FOUR_CONTACT_ACTOR_OBSERVATION_SIZE
            and payload.get("critic_observation_size")
            == FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE
        ):
            matching.append(manifest)
    if not matching:
        raise ValueError(
            f"restore artifact must contain stage_result.json for {expected_slope:g} degrees"
        )
    return latest_checkpoint(root)


def make_env(slope: float, *, impl: str, validation: bool = False):
    cfg = default_four_contact_config()
    with cfg.unlocked():
        cfg.slope_degrees = validate_four_contact_slope(slope)
        cfg.impl = impl
        if validation:
            cfg.noise_config.level = 0.0
            cfg.command_stand_probability = 0.0
    return HimalayaG1FourContactEnv(config=cfg)


def train_stage(
    slope: float, output: Path, *, restore: Path | None,
    timesteps: int, seed: int, impl: str, num_envs: int | None,
) -> tuple[Callable, Path]:
    env = make_env(slope, impl=impl)
    params = ppo_config(timesteps)
    if num_envs is not None:
        params.num_envs = int(num_envs)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **params.network_factory
    )
    training_params = dict(params)
    del training_params["network_factory"]
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(step: int, metrics: dict[str, float]) -> None:
        print(
            f"slope={slope:g} step={step:,} "
            f"reward={metrics.get('eval/episode_reward', float('nan')):.3f} "
            f"elapsed={(time.monotonic() - started) / 60:.1f}m",
            flush=True,
        )

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=seed,
        restore_checkpoint_path=restore,
        save_checkpoint_path=checkpoint_dir,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=params.get("num_eval_envs", 128),
    )
    make_inference_fn, trained_params, _ = train_fn(
        environment=env, eval_env=env, progress_fn=progress
    )
    return (
        make_inference_fn(trained_params, deterministic=True),
        latest_checkpoint(checkpoint_dir),
    )


def load_inference_fn(env, checkpoint: str | Path) -> Callable:
    path = Path(checkpoint).resolve()
    if path.is_dir() and not path.name.isdigit():
        path = latest_checkpoint(path)
    params = ppo_config(0)
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, **params.network_factory
    )
    restored = ppo_checkpoint.load(path)
    return ppo_networks.make_inference_fn(network)(restored, deterministic=True)


def _write_csv(metrics: list[FourContactMetrics], output: Path) -> None:
    rows = [item.to_dict() for item in metrics]
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stage(args: argparse.Namespace) -> int:
    slope = validate_four_contact_slope(args.slope)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if slope == 30.0 and args.restore:
        raise ValueError("30-degree acquisition must start from scratch")
    if slope == 35.0 and not args.restore:
        raise ValueError("35-degree training requires a reviewed 30-degree restore artifact")
    restore = (
        resolve_restore_checkpoint(args.restore, expected_slope=30.0)
        if args.restore else None
    )
    stage = output / f"stage_{slope:g}deg"
    print(f"\n=== four-contact balance stage: {slope:g} degrees ===")
    inference, checkpoint = train_stage(
        slope, stage, restore=restore,
        timesteps=args.timesteps,
        seed=args.seed, impl=args.impl, num_envs=args.num_envs,
    )
    checkpoint_relative = checkpoint.relative_to(output)
    (output / "latest_checkpoint.txt").write_text(
        checkpoint_relative.as_posix() + "\n", encoding="utf-8"
    )
    marker = {
        "slope_degrees": slope,
        "checkpoint": checkpoint_relative.as_posix(),
        "warm_started_from": str(restore) if restore else None,
    }
    (output / "latest_stage.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    metrics = evaluate_four_contact_policy(
        make_env(slope, impl=args.impl, validation=True),
        inference, trials=args.validation_trials,
        seed=args.seed + 10_000,
    )
    results = [metrics]
    write_four_contact_report(results, output / "validation.json")
    _write_csv(results, output / "validation.csv")
    promoted = (
        metrics.four_contact_ratio >= args.promotion_success_rate
        and metrics.nonfinite_terminations == 0
    )
    result = {
        **marker,
        "action_size": G1_ACTION_SIZE,
        "actor_observation_size": FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,
        "critic_observation_size": FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,
        "success_rate": metrics.success_rate,
        "four_contact_ratio": metrics.four_contact_ratio,
        "promotion_success_rate": args.promotion_success_rate,
        "promotion_passed": promoted,
    }
    (output / "stage_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics)
    if not promoted:
        print(
            f"balance stage held at {slope:g} degrees: {metrics.four_contact_ratio:.1%} "
            f"< {args.promotion_success_rate:.1%}"
        )
        return 2
    print(f"balance stage passed at {slope:g} degrees")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/g1_four_contact_balance")
    parser.add_argument("--slope", type=float, required=True, choices=(30.0, 35.0))
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--validation-trials", type=int, default=64)
    parser.add_argument("--promotion-success-rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--restore")
    return parser


def main() -> None:
    raise SystemExit(run_stage(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
