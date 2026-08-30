#!/usr/bin/env python3
"""Run cheap, non-JIT sanity checks after a reusable smoke gate is verified."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from himalaya.tasks.four_contact_env_cfg import make_four_contact_env
from himalaya.tasks.g1_cfg import (
    FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,
    FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,
    G1_ACTION_SIZE,
)
from runtime_fingerprint import runtime_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    expected_runtime = os.environ["RUNTIME_DIGEST"]
    actual_runtime = runtime_manifest(Path("."))["runtime_digest"]
    env = make_four_contact_env(30.0, noise_level=0.0, impl="jax")
    checks = {
        "runtime digest": actual_runtime == expected_runtime,
        "gpu backend": jax.default_backend() == "gpu",
        "action size": env.action_size == G1_ACTION_SIZE == 29,
        "actor observation contract": FOUR_CONTACT_ACTOR_OBSERVATION_SIZE == 103,
        "critic observation contract": FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE == 241,
        "left hand friction": np.allclose(
            np.asarray(env.mj_model.pair_friction[env.mj_model.pair("left_hand_floor").id])[:2],
            [0.9, 0.9],
        ),
        "right hand friction": np.allclose(
            np.asarray(env.mj_model.pair_friction[env.mj_model.pair("right_hand_floor").id])[:2],
            [0.9, 0.9],
        ),
        "left foot friction": np.allclose(
            np.asarray(env.mj_model.pair_friction[env.mj_model.pair("left_foot_floor").id])[:2],
            [1.0, 1.0],
        ),
        "right foot friction": np.allclose(
            np.asarray(env.mj_model.pair_friction[env.mj_model.pair("right_foot_floor").id])[:2],
            [1.0, 1.0],
        ),
    }
    report = {
        "schema_version": 1,
        "passed": all(bool(value) for value in checks.values()),
        "duration_seconds": time.monotonic() - started,
        "source_revision": os.environ.get("SOURCE_REVISION"),
        "runtime_digest": actual_runtime,
        "image_ref": os.environ.get("IMAGE_REF"),
        "checks": {name: bool(value) for name, value in checks.items()},
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
