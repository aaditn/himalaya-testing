"""Command-line evaluation across every Stage-I slope level."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .evaluation import evaluate_policy, write_report
from .tasks.g1_cfg import CURRICULUM_SLOPES_DEG
from .tasks.himalaya_env_cfg import default_config, HimalayaG1UphillEnv
from .training import load_inference_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Himalaya G1 policy at 0, 5, 10, 15, and 20 degrees."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="validation/all_slopes.json")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    args = parser.parse_args()

    load_cfg = default_config()
    with load_cfg.unlocked():
        load_cfg.slope_degrees = 0.0
        load_cfg.noise_config.level = 0.0
        load_cfg.command_stand_probability = 0.0
        load_cfg.impl = args.impl
    load_env = HimalayaG1UphillEnv(config=load_cfg)
    inference_fn = load_inference_fn(load_env, args.checkpoint, seed=args.seed)

    results = []
    for index, slope in enumerate(CURRICULUM_SLOPES_DEG):
        cfg = default_config()
        with cfg.unlocked():
            cfg.slope_degrees = slope
            cfg.noise_config.level = 0.0
            cfg.command_stand_probability = 0.0
            cfg.impl = args.impl
        env = HimalayaG1UphillEnv(config=cfg)
        result = evaluate_policy(
            env, inference_fn, trials=args.trials, seed=args.seed + index
        )
        results.append(result)
        print(result)

    output = Path(args.output).resolve()
    write_report(results, output)
    csv_path = output.with_suffix(".csv")
    rows = [r.to_dict() for r in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {csv_path}")


if __name__ == "__main__":
    main()
