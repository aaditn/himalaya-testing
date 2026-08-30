from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flexsense.grip import FingerGrip, GripAssessment
from flexsense.labeling import GripLabelDataset


class GripLabelDatasetTests(unittest.TestCase):
    def assessment(self) -> GripAssessment:
        finger = FingerGrip(name="left", state="wrapping", signed_bend_deg=-7.5,
                            tags_seen=2)
        return GripAssessment(verdict="good", reason="one test finger",
                              fingers={"left": finger}, wrapping=1)

    def test_save_records_raw_frame_and_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = GripLabelDataset(directory)
            record = dataset.save(
                np.full((12, 16, 3), 80, np.uint8), "good", self.assessment(),
                "calibration/camera_intrinsics.json (project)")

            self.assertEqual(record["label"], "GOOD")
            self.assertTrue((Path(directory) / record["image"]).is_file())
            lines = (Path(directory) / "labels.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            stored = json.loads(lines[0])
            self.assertEqual(stored["assessment"]["fingers"]["left"]["tags_seen"], 2)
            self.assertEqual(dataset.summary, "dataset  GOOD 1  BAD 0")

    def test_undo_removes_latest_record_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = GripLabelDataset(directory)
            first = dataset.save(np.zeros((8, 8, 3), np.uint8), "GOOD", self.assessment())
            second = dataset.save(np.zeros((8, 8, 3), np.uint8), "BAD", self.assessment())

            removed = dataset.undo()

            self.assertEqual(removed["sample_id"], second["sample_id"])
            self.assertFalse((Path(directory) / second["image"]).exists())
            self.assertTrue((Path(directory) / first["image"]).exists())
            self.assertEqual(dataset.summary, "dataset  GOOD 1  BAD 0")

    def test_undo_empty_dataset_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(GripLabelDataset(directory).undo())


if __name__ == "__main__":
    unittest.main()
