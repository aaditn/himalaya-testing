#!/usr/bin/env python3
"""Small, credential-free wrapper for this project's Hugging Face Jobs remote."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "hf_jobs.json"


def _load_config() -> dict[str, str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = {"name", "namespace", "runs_volume"}
    missing = required.difference(config)
    if missing:
        raise SystemExit(f"Missing myremote settings: {', '.join(sorted(missing))}")
    return config


def _hf_cli() -> str:
    override = os.environ.get("HF_CLI")
    candidates = [
        override,
        str(ROOT / ".venv" / "Scripts" / "hf.exe"),
        str(ROOT / ".venv" / "bin" / "hf"),
        shutil.which("hf"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("Hugging Face CLI not found. Install huggingface_hub[cli] or set HF_CLI.")


def _has_option(args: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in args)


def main(argv: list[str]) -> int:
    config = _load_config()
    command = argv[0] if argv else "list"
    args = argv[1:] if argv else []

    if command == "config":
        print(json.dumps(config, indent=2))
        return 0

    if command in {"list", "ls", "ps"}:
        hf_command = "list" if command in {"ls", "ps"} else command
        injected = [] if _has_option(args, "--namespace") else ["--namespace", config["namespace"]]
        cmd = [_hf_cli(), "jobs", hf_command, *injected, *args]
    elif command == "run":
        injected = []
        if not _has_option(args, "--namespace"):
            injected += ["--namespace", config["namespace"]]
        if "--no-runs-volume" in args:
            args = [arg for arg in args if arg != "--no-runs-volume"]
        else:
            injected += ["--volume", config["runs_volume"]]
        if not _has_option(args, "--label", "-l"):
            injected += ["--label", f"remote={config['name']}"]
        cmd = [_hf_cli(), "jobs", "run", *injected, *args]
    else:
        # Job-id operations (logs, inspect, cancel, wait, stats, ssh) do not
        # need a namespace: the Hub resolves the job from its globally unique ID.
        cmd = [_hf_cli(), "jobs", command, *args]

    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
