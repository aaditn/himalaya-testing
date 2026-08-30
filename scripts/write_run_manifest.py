#!/usr/bin/env python3
"""Write the effective, reproducible configuration of an HF job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform


TRACKED_ENV = (
    "HF_REPO_ID",
    "IMAGE_REF",
    "JOB_MODE",
    "NUM_ENVS",
    "PROMOTION_SUCCESS_RATE_30",
    "PROMOTION_SUCCESS_RATE_35",
    "REMOTE_OUTPUT_PATH",
    "RUN_ID",
    "SKIP_SMOKE_GATE",
    "SOURCE_DIGEST",
    "SOURCE_REVISION",
    "RUNTIME_DIGEST",
    "SMOKE_GATE_PATH",
    "HUMAN_AUDIT_APPROVED_BY",
    "HUMAN_AUDIT_APPROVAL_REF",
    "HUMAN_AUDIT_APPROVED_AT",
    "TRAINING_TIMESTEPS_30",
    "TRAINING_TIMESTEPS_35",
    "VALIDATION_TRIALS",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", default="starting")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()

    target = Path(args.output)
    now = datetime.now(timezone.utc)
    previous = {}
    if target.is_file():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    started_at = previous.get("started_at_utc", now.isoformat())
    try:
        elapsed = (now - datetime.fromisoformat(started_at)).total_seconds()
    except (TypeError, ValueError):
        started_at = now.isoformat()
        elapsed = 0.0
    packages = Path("/opt/himalaya-image/installed-packages.txt")
    package_bytes = packages.read_bytes() if packages.is_file() else b""
    report = {
        "status": args.status,
        "exit_code": args.exit_code,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in TRACKED_ENV},
        "installed_packages_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "started_at_utc": started_at,
        "updated_at_utc": now.isoformat(),
        "elapsed_seconds": elapsed,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
