"""Run the exact Brax evaluator/PPO/checkpoint/render path on a tiny workload."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from .four_contact_rendering import render_policy
from .four_contact_training import latest_checkpoint, make_env, ppo_config


def _verify_video(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"smoke video is missing or empty: {path}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    packets = int(streams[0].get("nb_read_packets", 0))
    duration = float(streams[0].get("duration", 0.0))
    if packets < 1 or duration <= 0.0:
        raise RuntimeError(
            f"invalid smoke video: packets={packets}, duration={duration}"
        )
    return {"packets": packets, "duration_seconds": duration}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_smoke(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    output = Path(args.output).resolve()
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    env = make_env(args.slope, impl="jax")

    params = dict(ppo_config(args.timesteps))
    params.update({
        "num_timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        # Two evaluations force the same pre-training evaluator scan that
        # previously exposed the bool/float metric carry mismatch.
        "num_evals": 2,
        "episode_length": 16,
        "unroll_length": 4,
        "batch_size": 8,
        "num_minibatches": 8,
        "num_updates_per_batch": 1,
    })
    network_config = params.pop("network_factory")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_config
    )

    progress_reports: list[dict[str, float | int]] = []

    def progress(step: int, metrics: dict[str, float]) -> None:
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        if not math.isfinite(reward):
            raise RuntimeError(f"non-finite smoke reward at step {step}: {reward}")
        progress_reports.append({"step": int(step), "episode_reward": reward})
        print(
            f"smoke step={step:,} "
            f"reward={reward:.3f}",
            flush=True,
        )

    make_inference_fn, trained_params, _ = ppo.train(
        environment=env,
        eval_env=env,
        progress_fn=progress,
        network_factory=network_factory,
        seed=args.seed,
        save_checkpoint_path=checkpoint_root,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **params,
    )
    # Constructing inference exercises the returned parameter tree before reload.
    make_inference_fn(trained_params, deterministic=True)
    checkpoint = latest_checkpoint(checkpoint_root)
    (output / "latest_checkpoint.txt").write_text(
        str(checkpoint.resolve()) + "\n", encoding="utf-8"
    )

    video = render_policy(
        checkpoint,
        output / "videos" / "smoke.mp4",
        slope=args.slope,
        seed=args.seed + 1,
        seconds=0.5,
    )
    video_probe = _verify_video(video)
    preflight = Path(args.preflight_manifest).resolve()
    preflight_report = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_report.get("passed") is not True:
        raise RuntimeError("compiled preflight manifest does not report passed=true")
    report = {
        "schema_version": 2,
        "passed": True,
        "source_revision": args.source_revision,
        "source_digest": args.source_digest,
        "runtime_digest": args.runtime_digest,
        "image_ref": args.image_ref,
        "slope_degrees": args.slope,
        "num_envs": args.num_envs,
        "timesteps": args.timesteps,
        "ppo_progress": progress_reports,
        "ppo_metrics_finite": bool(progress_reports),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_path(checkpoint),
        "video": str(video),
        "video_sha256": _sha256_path(video),
        "video_probe": video_probe,
        "preflight_manifest": str(preflight),
        "preflight_manifest_sha256": _sha256_path(preflight),
        "preflight_duration_seconds": preflight_report.get("duration_seconds"),
        "smoke_duration_seconds": time.monotonic() - started,
    }
    marker = output / "smoke_pass.json"
    marker.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"SMOKE PASS: {marker}", flush=True)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--slope", type=float, default=30.0, choices=(30.0, 35.0))
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
