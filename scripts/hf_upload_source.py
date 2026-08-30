#!/usr/bin/env python3
"""Upload one immutable source snapshot and print its exact Hub revision."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from runtime_fingerprint import runtime_manifest


IGNORE_PATTERNS = [
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    "**/__pycache__/**",
    "*.pyc",
    "artifacts/**",
    "local_preview/**",
    "runs/**",
    "validation/**",
]


def _ignored(relative: str) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in IGNORE_PATTERNS)


def _source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        if _ignored(relative):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source", default=".")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    source_digest = _source_digest(source)
    runtime = runtime_manifest(source)
    source_branch = f"source-{source_digest[:16]}"
    api.create_branch(
        repo_id=args.repo_id,
        repo_type="model",
        branch=source_branch,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(source),
        path_in_repo=".",
        ignore_patterns=IGNORE_PATTERNS,
        delete_patterns=["*"],
        commit_message="Upload immutable G1 four-contact training source",
        revision=source_branch,
    )
    revision = commit.oid or api.repo_info(
        args.repo_id, repo_type="model", revision=source_branch
    ).sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a source revision")
    if _source_digest(source) != source_digest:
        raise RuntimeError(
            "local source changed during upload; refusing to launch from a mixed snapshot"
        )
    print(json.dumps({
        "repo_id": args.repo_id,
        "source_revision": revision,
        "source_branch": source_branch,
        "source_digest": source_digest,
        "runtime_digest": runtime["runtime_digest"],
        "runtime_file_count": len(runtime["files"]),
        "commit_url": str(commit.commit_url),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
