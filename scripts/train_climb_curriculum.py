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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=Path("configs/curriculum.json"))
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--prefix", default="g1_climb")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stages = json.loads(args.curriculum.read_text(encoding="utf-8"))["stages"]
    restore = args.restore
    for index, stage in enumerate(stages):
        name = f"{args.prefix}_{index:02d}_{stage['name']}"
        command = [
            sys.executable,
            "scripts/train.py",
            "--climb",
            "--name", name,
            "--envs", str(args.envs),
            "--timesteps", str(stage["num_timesteps"]),
            "--slope", str(stage["slope_degrees"]),
            "--roughness", str(stage["roughness_m"]),
            "--spike-friction", str(stage["spike_friction"]),
            "--foot-friction", str(2.0 * stage["spike_friction"]),
            "--hand-load", str(stage["target_hand_load_share"]),
            "--speed", str(stage["target_uphill_speed"]),
            "--num-evals", str(stage["num_evals"]),
            "--seed", str(stage.get("terrain_seed", args.seed + index)),
        ]
        if restore:
            command.extend(["--restore", str(restore.resolve())])
        if not stage.get("boulders_enabled", True):
            command.append("--no-boulders")
        if not stage.get("domain_randomization", True):
            command.append("--no-randomization")
        subprocess.run(command, check=True)
        restore = latest_checkpoint(Path("runs") / name / "checkpoints")
        if restore is None:
            raise RuntimeError(f"no checkpoint written for {name}")

    print(f"curriculum complete: {restore}")


if __name__ == "__main__":
    main()
