"""Train the four-limb climb curriculum with the native Himalaya trainer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = [
        path for path in root.glob("*")
        if path.is_dir() and path.name.isdigit()
    ]
    return max(checkpoints, key=lambda path: int(path.name)) if checkpoints else None


def selected_checkpoint(run_root: Path) -> Path | None:
    """Return the checkpoint chosen by the ascent objective, with safe fallback."""
    selection = run_root / "best_checkpoint.json"
    if selection.is_file():
        step = int(json.loads(selection.read_text(encoding="utf-8"))["step"])
        checkpoint = run_root / "checkpoints" / f"{step:012d}"
        if checkpoint.is_dir():
            return checkpoint
    return latest_checkpoint(run_root / "checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=Path("configs/curriculum.json"))
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--prefix", default="g1_climb")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stages = json.loads(args.curriculum.read_text(encoding="utf-8"))["stages"]
    restore = args.restore
    for index, stage in enumerate(stages):
        if index < args.start_stage:
            continue
        name = f"{args.prefix}_{index:02d}_{stage['name']}"
        command = [
            sys.executable,
            "scripts/train.py",
            "--climb",
            "--name", name,
            "--runs-dir", str(args.runs_dir),
            "--envs", str(args.envs),
            "--timesteps", str(stage["num_timesteps"]),
            "--slope", str(stage["slope_degrees"]),
            "--roughness", str(stage["roughness_m"]),
            "--spike-friction", str(stage["spike_friction"]),
            "--foot-friction", str(2.0 * stage["spike_friction"]),
            "--hand-load", str(stage["target_hand_load_share"]),
            "--speed", str(stage["target_uphill_speed"]),
            "--num-evals", str(stage["num_evals"]),
            "--eval-envs", str(stage.get("eval_envs", 32)),
            "--action-scale", str(stage.get("action_scale", 0.35)),
            "--learning-rate", str(stage.get("learning_rate", 1e-4)),
            "--entropy-cost", str(stage.get("entropy_cost", 0.002)),
            "--updates-per-batch", str(stage.get("updates_per_batch", 3)),
            "--seed", str(stage.get("terrain_seed", args.seed + index)),
        ]
        if restore:
            command.extend(["--restore", str(restore.resolve())])
        if not stage.get("boulders_enabled", True):
            command.append("--no-boulders")
        if not stage.get("domain_randomization", True):
            command.append("--no-randomization")
        subprocess.run(command, check=True)
        restore = selected_checkpoint(args.runs_dir / name)
        if restore is None:
            raise RuntimeError(f"no checkpoint written for {name}")

    print(f"curriculum complete: {restore}")


if __name__ == "__main__":
    main()
