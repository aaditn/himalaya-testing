#!/usr/bin/env python3
"""Verify reusable smoke evidence against runtime and immutable image digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate")
    parser.add_argument("--repo-id")
    parser.add_argument("--gate-path")
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.gate:
        gate_file = Path(args.gate)
    elif args.repo_id and args.gate_path:
        from huggingface_hub import hf_hub_download

        gate_file = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=args.gate_path,
            repo_type="model",
        ))
    else:
        parser.error("pass --gate or both --repo-id and --gate-path")
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    problems = []
    if gate.get("passed") is not True:
        problems.append("smoke marker does not report passed=true")
    if not gate.get("source_revision"):
        problems.append("smoke marker has no audited source revision")
    if gate.get("runtime_digest") != args.runtime_digest:
        problems.append("smoke marker runtime digest does not match")
    if gate.get("image_ref") != args.image_ref:
        problems.append("smoke marker image digest does not match")
    if gate.get("slope_degrees") not in (30, 30.0):
        problems.append("smoke marker did not exercise the 30-degree environment")
    video_probe = gate.get("video_probe", {})
    if not isinstance(video_probe, dict):
        video_probe = {}
    if not isinstance(video_probe.get("packets"), int) or video_probe["packets"] < 1:
        problems.append("smoke marker has no verified video packets")
    duration = video_probe.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration <= 0.0:
        problems.append("smoke marker has no verified video duration")
    for field in (
        "checkpoint_sha256", "video_sha256", "preflight_manifest_sha256"
    ):
        value = gate.get(field)
        if not isinstance(value, str) or len(value) != 64:
            problems.append(f"smoke marker has no valid {field}")
    verification = {
        "schema_version": 1,
        "passed": not problems,
        "audited_source_revision": gate.get("source_revision"),
        "current_source_revision": args.source_revision,
        "runtime_digest": args.runtime_digest,
        "image_ref": args.image_ref,
        "gate_path": args.gate_path or str(gate_file),
        "gate": gate,
        "problems": problems,
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if problems:
        raise SystemExit("; ".join(problems))
    print(
        f"verified smoke gate from source {gate.get('source_revision')} for "
        f"runtime {args.runtime_digest} and image {args.image_ref}"
    )


if __name__ == "__main__":
    main()
