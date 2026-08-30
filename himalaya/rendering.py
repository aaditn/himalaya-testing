"""Render deterministic MuJoCo rollouts from a trained PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import jax
import mediapy as media
import mujoco

from .tasks.g1_cfg import validate_slope
from .tasks.himalaya_env_cfg import default_config, HimalayaG1UphillEnv
from .training import load_inference_fn


def _configure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable; install ffmpeg or imageio-ffmpeg"
        ) from exc


def _make_env(slope_degrees: float) -> HimalayaG1UphillEnv:
    cfg = default_config()
    with cfg.unlocked():
        cfg.slope_degrees = validate_slope(slope_degrees)
        cfg.noise_config.level = 0.0
        cfg.command_stand_probability = 0.0
        cfg.impl = "jax"
    return HimalayaG1UphillEnv(config=cfg)


def render_rollout(
    env: HimalayaG1UphillEnv,
    inference_fn,
    output: Path,
    *,
    seed: int,
    seconds: float,
    width: int,
    height: int,
) -> None:
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    policy = jax.jit(inference_fn)
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset(reset_rng)

    trajectory = []
    max_steps = min(
        env._config.episode_length, max(1, round(seconds / env.dt))
    )
    for _ in range(max_steps):
        trajectory.append(state)
        rng, action_rng = jax.random.split(rng)
        action = policy(state.obs, action_rng)[0]
        state = step(state, action)
        if float(state.done) > 0.5:
            trajectory.append(state)
            break

    render_every = 2
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    frames = env.render(
        trajectory[::render_every],
        height=height,
        width=width,
        camera="track",
        scene_option=scene_option,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _configure_ffmpeg()
    media.write_video(output, frames, fps=1.0 / env.dt / render_every, qp=18)
    print(f"wrote {output} ({len(frames)} frames)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="videos/latest")
    parser.add_argument(
        "--slopes", nargs="+", type=float, default=[0.0, 5.0, 10.0, 15.0]
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    load_env = _make_env(args.slopes[0])
    inference_fn = load_inference_fn(load_env, args.checkpoint, seed=args.seed)
    for index, slope in enumerate(args.slopes):
        env = _make_env(slope)
        render_rollout(
            env,
            inference_fn,
            output / f"g1_uphill_{slope:g}deg.mp4",
            seed=args.seed + index,
            seconds=args.seconds,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
