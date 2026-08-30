"""Small, inspectable datasets for supervised grip classification.

Each keypress stores the untouched camera frame plus the complete FlexSense
assessment that was visible when the human assigned GOOD or BAD.  JSON Lines
keeps the dataset append-friendly and makes an undo a transparent pop of the
last record instead of hiding state in a database.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .grip import GripAssessment
from .vision import require_cv2

VALID_LABELS = ("GOOD", "BAD")
SCHEMA_VERSION = 1


class GripLabelDataset:
    """Append and undo human-labeled camera samples on local disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames = self.root / "frames"
        self.manifest = self.root / "labels.jsonl"
        self._records = self._read_records()

    def _read_records(self) -> list[dict]:
        if not self.manifest.exists():
            return []
        records = []
        for number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") not in VALID_LABELS:
                raise ValueError(
                    f"{self.manifest}:{number} has unsupported label {record.get('label')!r}")
            records.append(record)
        return records

    @property
    def counts(self) -> Counter:
        return Counter(record["label"] for record in self._records)

    @property
    def summary(self) -> str:
        counts = self.counts
        return f"dataset  GOOD {counts['GOOD']}  BAD {counts['BAD']}"

    def save(self, frame: np.ndarray, label: str, assessment: GripAssessment,
             calibration: str = "") -> dict:
        label = label.upper()
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        if assessment is None:
            raise ValueError("cannot label before the grip assessment is available")

        captured = datetime.now(timezone.utc)
        sample_id = f"{captured.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        relative_image = Path("frames") / f"{sample_id}.png"
        image_path = self.root / relative_image
        self.frames.mkdir(parents=True, exist_ok=True)

        cv2 = require_cv2()
        if not cv2.imwrite(str(image_path), frame):
            raise OSError(f"could not write labeled frame to {image_path}")

        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "captured_at": captured.isoformat(),
            "label": label,
            "image": relative_image.as_posix(),
            "calibration": calibration,
            "assessment": assessment.to_dict(),
        }
        self._records.append(record)
        try:
            self._write_records()
        except Exception:
            self._records.pop()
            image_path.unlink(missing_ok=True)
            raise
        return record

    def undo(self) -> dict | None:
        if not self._records:
            return None
        record = self._records.pop()
        self._write_records()
        image = self.root / record["image"]
        image.unlink(missing_ok=True)
        return record

    def _write_records(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest.with_suffix(".jsonl.tmp")
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in self._records)
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.manifest)
