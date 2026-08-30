#!/usr/bin/env python3
"""Print checkpoint and slope from a curriculum latest-stage marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker")
    args = parser.parse_args()
    data = json.loads(Path(args.marker).read_text(encoding="utf-8"))
    checkpoint = Path(data["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path(args.marker).resolve().parent / checkpoint
    if not checkpoint.is_dir() or not checkpoint.name.isdigit():
        raise SystemExit(f"invalid numeric PPO checkpoint: {checkpoint}")
    print(checkpoint)
    print(float(data["slope_degrees"]))


if __name__ == "__main__":
    main()
