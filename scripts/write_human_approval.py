#!/usr/bin/env python3
"""Persist the operator's explicit approval bound to the real run evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    required = (
        "HUMAN_AUDIT_APPROVED_BY",
        "HUMAN_AUDIT_APPROVAL_REF",
        "HUMAN_AUDIT_APPROVED_AT",
        "SOURCE_REVISION",
        "SOURCE_DIGEST",
        "RUNTIME_DIGEST",
        "IMAGE_REF",
        "SMOKE_GATE_PATH",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("missing human approval fields: " + ", ".join(missing))
    report = {
        "schema_version": 1,
        "approved": True,
        "approved_by": os.environ["HUMAN_AUDIT_APPROVED_BY"],
        "approval_reference": os.environ["HUMAN_AUDIT_APPROVAL_REF"],
        "approved_at_utc": os.environ["HUMAN_AUDIT_APPROVED_AT"],
        "real_source_revision": os.environ["SOURCE_REVISION"],
        "real_source_digest": os.environ["SOURCE_DIGEST"],
        "runtime_digest": os.environ["RUNTIME_DIGEST"],
        "image_ref": os.environ["IMAGE_REF"],
        "smoke_gate_path": os.environ["SMOKE_GATE_PATH"],
        "smoke_gate_explicitly_waived": os.environ.get("SKIP_SMOKE_GATE") == "1",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
