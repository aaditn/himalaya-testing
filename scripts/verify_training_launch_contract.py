#!/usr/bin/env python3
"""Execute the repository's persistent HF training-launch contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / ".cursor" / "rules" / "hf-training-launch.mdc"
REQUIRED_RULE_TEXT = (
    "REQUIRED_LAUNCH_CONTRACT: DOCKERFILE_HF_ONLY",
    "/opt/himalaya-image/menagerie-ready",
    "/opt/himalaya-image/provenance",
    "Dockerfile.hf",
)
IMAGE_PATTERN = re.compile(
    r"/himalaya(?:-g1)?-hf@sha256:[0-9a-fA-F]{64}$"
)


def verify(image: str) -> None:
    rule = RULE.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_RULE_TEXT if token not in rule]
    if missing:
        raise RuntimeError(f"HF launch rule is incomplete: {missing}")
    if not IMAGE_PATTERN.search(image):
        raise RuntimeError(
            "image must be a digest-pinned himalaya-hf image built from Dockerfile.hf"
        )
    dockerfile = (ROOT / "Dockerfile.hf").read_text(encoding="utf-8")
    if "/opt/himalaya-image/provenance" not in dockerfile:
        raise RuntimeError("Dockerfile.hf does not create the provenance marker")
    print(f"verified {RULE.relative_to(ROOT)} for {image}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    verify(args.image)


if __name__ == "__main__":
    main()

