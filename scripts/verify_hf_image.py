#!/usr/bin/env python3
"""Fail fast unless the job is using the pinned, GPU-capable HF image."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

import jax


EXPECTED = {
    "brax": "0.14.2",
    "flax": "0.11.2",
    "huggingface-hub": "1.29.0",
    "imageio-ffmpeg": "0.6.0",
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "mediapy": "1.2.7",
    "mujoco": "3.12.0",
    "mujoco-mjx": "3.12.0",
    "optax": "0.2.8",
    "orbax-checkpoint": "0.12.4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    problems: list[str] = []
    versions: dict[str, str] = {}
    for package, expected in EXPECTED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        versions[package] = actual
        if actual != expected:
            problems.append(f"{package}: expected {expected}, found {actual}")

    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python 3.11 required, found {sys.version.split()[0]}")
    if not Path("/opt/himalaya-image/menagerie-ready").is_file():
        problems.append("pinned MuJoCo Menagerie assets are not baked into the image")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        problems.append("ffmpeg and ffprobe must be baked into the image")

    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    if backend != "gpu" and not args.allow_cpu:
        problems.append(f"JAX GPU backend required, found {backend}")

    report = {
        "passed": not problems,
        "python": sys.version.split()[0],
        "jax_backend": backend,
        "jax_devices": devices,
        "versions": versions,
        "problems": problems,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
