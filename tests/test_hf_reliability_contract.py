"""Dependency-free regression checks for the guarded HF launch path."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HuggingFaceReliabilityContractTest(unittest.TestCase):
    def test_image_bakes_dependencies_and_menagerie(self) -> None:
        dockerfile = (ROOT / "Dockerfile.hf").read_text(encoding="utf-8")
        self.assertIn("requirements-hf.txt", dockerfile)
        self.assertIn("ensure_menagerie_exists", dockerfile)
        self.assertIn("menagerie-ready", dockerfile)

    def test_runner_uses_exact_smoke_and_strict_final_sync(self) -> None:
        runner = (ROOT / "scripts" / "hf_four_contact_job.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/smoke_four_contact.py", runner)
        self.assertIn("sync_output_strict", runner)
        self.assertIn("latest_stage.json", runner)
        self.assertIn("--slope 30", runner)
        self.assertNotIn("--slope 35", runner)
        self.assertNotIn("--start-slope", runner)
        self.assertNotIn("pip install --upgrade", runner)
        self.assertIn("/opt/himalaya-image/provenance", runner)
        self.assertNotIn("apt-get install", runner)

    def test_submitter_requires_digest_and_matching_smoke_gate(self) -> None:
        submitter = (
            ROOT / "scripts" / "submit_hf_four_contact_job.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("@sha256:", submitter)
        self.assertIn("verify_smoke_gate.py", submitter)
        self.assertIn("runtimeDigest", submitter)
        self.assertIn("SOURCE_REVISION", submitter)
        self.assertIn("REMOTE_OUTPUT_PATH", submitter)
        self.assertIn("TRAINING_TIMESTEPS_30", submitter)
        self.assertNotIn('ValidateSet("Smoke", "Audit", "Real")', submitter)
        self.assertIn("built from Dockerfile.hf", submitter)

    def test_all_hf_jobs_require_baked_runtime(self) -> None:
        for name in ("hf_job.sh", "hf_video_job.sh"):
            runner = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("/opt/himalaya-image/provenance", runner)
            self.assertNotIn("apt-get install", runner)
            self.assertNotIn("pip install --upgrade", runner)
        for name in ("submit_hf_job.ps1", "submit_hf_video_job.ps1"):
            submitter = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("built from Dockerfile.hf", submitter)
            self.assertNotIn("pytorch/pytorch:", submitter)

    def test_persistent_launch_rule_is_executed_everywhere(self) -> None:
        rule = ROOT / ".cursor" / "rules" / "hf-training-launch.mdc"
        self.assertTrue(rule.is_file())
        rule_text = rule.read_text(encoding="utf-8")
        self.assertIn("alwaysApply: true", rule_text)
        self.assertIn("DOCKERFILE_HF_ONLY", rule_text)
        verifier = "scripts/verify_training_launch_contract.py"
        for name in ("hf_four_contact_job.sh", "hf_job.sh", "hf_video_job.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn(verifier, text)
        for name in (
            "submit_hf_four_contact_job.ps1",
            "submit_hf_job.ps1",
            "submit_hf_video_job.ps1",
        ):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn(verifier, text)

    def test_training_publishes_last_completed_stage(self) -> None:
        training = (
            ROOT / "himalaya" / "four_contact_training.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"latest_stage.json"', training)
        self.assertIn('"checkpoint": checkpoint_relative.as_posix()', training)
        self.assertIn('"slope_degrees": slope', training)
        self.assertIn("30-degree acquisition must start from scratch", training)
        self.assertIn("35-degree training requires a reviewed 30-degree", training)

    def test_smoke_gate_binds_source_image_and_video(self) -> None:
        marker = {
            "passed": True,
            "source_revision": "source-sha",
            "runtime_digest": "runtime-sha",
            "image_ref": "registry/image@sha256:" + "a" * 64,
            "slope_degrees": 30.0,
            "video_probe": {"packets": 10, "duration_seconds": 2.0},
            "checkpoint_sha256": "b" * 64,
            "video_sha256": "c" * 64,
            "preflight_manifest_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "smoke_pass.json"
            gate.write_text(json.dumps(marker), encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "verify_smoke_gate.py"),
                "--gate", str(gate),
                "--source-revision", "source-sha",
                "--runtime-digest", "runtime-sha",
                "--image-ref", marker["image_ref"],
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            rejected = subprocess.run(
                [*command[:-1], "registry/image@sha256:" + "b" * 64],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_runtime_gate_ignores_docs_but_invalidates_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "himalaya").mkdir()
            runtime = root / "himalaya" / "task.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("first\n", encoding="utf-8")

            def digest() -> str:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "runtime_fingerprint.py"),
                     "--root", str(root)],
                    check=True, capture_output=True, text=True,
                )
                return json.loads(result.stdout)["runtime_digest"]

            initial = digest()
            readme.write_text("docs only\n", encoding="utf-8")
            self.assertEqual(initial, digest())
            runtime.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(initial, digest())


if __name__ == "__main__":
    unittest.main()
