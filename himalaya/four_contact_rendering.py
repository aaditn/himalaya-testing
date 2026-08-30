"""Render a deterministic four-contact policy rollout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import jax
import mediapy as media
import mujoco

from .four_contact_training import load_inference_fn, make_env


def render_policy(
    checkpoint: str | Path, output: str | Path, *, slope: float = 30.0,
    seed: int = 2026, seconds: float = 20.0,
) -> Path:
    env = make_env(slope, impl="jax", validation=True)
    inference = load_inference_fn(env, checkpoint)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    policy = jax.jit(inference)
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset(reset_rng)
    trajectory = []
    for _ in range(min(env._config.episode_length, round(seconds / env.dt))):
        trajectory.append(state)
        rng, action_rng = jax.random.split(rng)
        state = step(state, policy(state.obs, action_rng)[0])
        if float(state.done) > 0.5:
            trajectory.append(state)
            break
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    frames = env.render(
        trajectory[::2], height=540, width=960,
        camera="track", scene_option=option,
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError as exc:
            raise RuntimeError("ffmpeg is required to render the audit") from exc
    media.write_video(target, frames, fps=1.0 / env.dt / 2, qp=18)
    print(f"wrote {target} ({len(frames)} frames)", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slope", type=float, default=30.0, choices=(30.0, 35.0))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    render_policy(
        args.checkpoint, args.output, slope=args.slope,
        seconds=args.seconds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
