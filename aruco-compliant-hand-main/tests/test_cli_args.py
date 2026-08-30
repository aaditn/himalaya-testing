"""Guard that every args.<name> read in cli.py is actually declared by a parser.

A subcommand losing an argument while its call site keeps passing it fails only
at runtime, after the user has already set up hardware. This catches it offline.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from flexsense.cli import build_parser

CLI_SOURCE = Path(__file__).resolve().parents[1] / "flexsense" / "cli.py"

MINIMAL_ARGV = {
    "markers": ["markers"],
    "track": ["track"],
    "camera-board": ["camera-board"],
    "camera-calibrate": ["camera-calibrate"],
    "screen-calibrate": ["screen-calibrate"],
    "camera-refit": ["camera-refit"],
    "watch": ["watch"],
    "grip": ["grip"],
    "label": ["label"],
    "hand-check": ["hand-check"],
    "hand-preview": ["hand-preview"],
    "hand-rehearse": ["hand-rehearse"],
    "calibrate": ["calibrate", "--sensor", "left_finger", "--axis", "normal",
                  "--loads-g", "0,50"],
    "finray": ["finray"],
}


class TestCliArguments(unittest.TestCase):
    def test_every_referenced_arg_is_declared(self) -> None:
        referenced = set(re.findall(r"\bargs\.([a-zA-Z_][a-zA-Z0-9_]*)", CLI_SOURCE.read_text()))
        parser = build_parser()
        available: set[str] = set()
        for argv in MINIMAL_ARGV.values():
            available |= vars(parser.parse_args(argv)).keys()
        missing = referenced - available
        self.assertEqual(missing, set(), f"cli.py reads args.{missing} but no parser declares it")

    def test_every_subcommand_parses(self) -> None:
        parser = build_parser()
        for name, argv in MINIMAL_ARGV.items():
            with self.subTest(command=name):
                self.assertEqual(parser.parse_args(argv).command, name)

    def test_camera_calibrate_defaults(self) -> None:
        args = build_parser().parse_args(["camera-calibrate"])
        self.assertEqual(args.square_mm, None)
        self.assertEqual(args.min_coverage, 0.55)
        self.assertEqual(args.views, 20)
        self.assertTrue(args.frames_dir)

    def test_label_dataset_default(self) -> None:
        args = build_parser().parse_args(["label"])
        self.assertEqual(args.dataset, "data/grip_labels")


if __name__ == "__main__":
    unittest.main()
