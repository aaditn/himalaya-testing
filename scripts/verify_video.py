#!/usr/bin/env python3
"""Require a non-empty, decodable video stream with positive duration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output")
    args = parser.parse_args()
    video = Path(args.video)
    if not video.is_file() or video.stat().st_size < 1024:
        raise SystemExit(f"video is missing or empty: {video}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets,duration",
            "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise SystemExit("ffprobe found no video stream")
    packets = int(streams[0].get("nb_read_packets", 0))
    duration = float(streams[0].get("duration", 0.0))
    if packets < 1 or duration <= 0.0:
        raise SystemExit(
            f"invalid video stream: packets={packets}, duration={duration}"
        )
    report = {
        "video": str(video.resolve()),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "packets": packets,
        "duration_seconds": duration,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
