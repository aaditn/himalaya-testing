#!/usr/bin/env python3
"""Fingerprint only files that can change the HF training/runtime outcome."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path


RUNTIME_PATTERNS = (
    "himalaya/*.py",
    "himalaya/**/*.py",
    "himalaya/*.xml",
    "himalaya/**/*.xml",
    "scripts/*.py",
    "scripts/*.sh",
    "requirements-hf.txt",
    "pyproject.toml",
    "Dockerfile.hf",
)


def runtime_manifest(root: Path) -> dict[str, object]:
    root = root.resolve()
    files: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in RUNTIME_PATTERNS):
            continue
        content = path.read_bytes()
        file_digest = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "sha256": file_digest, "bytes": len(content)})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
        aggregate.update(b"\0")
    if not files:
        raise RuntimeError(f"no runtime files found below {root}")
    return {
        "schema_version": 1,
        "runtime_digest": aggregate.hexdigest(),
        "patterns": list(RUNTIME_PATTERNS),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = runtime_manifest(Path(args.root))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered, end="")


if __name__ == "__main__":
    main()
