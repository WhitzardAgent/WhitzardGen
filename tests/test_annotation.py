import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whitzard.annotation.service import annotate_run
from whitzard.run_store import load_run_dataset_records


def _build_summary(*, run_id: str, output_dir: Path, export_path: Path):
    return type(
        "Summary",
        (),
        {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "export_path": str(export_path),
            "status": "completed",
            "to_dict": lambda self: {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "export_path": str(export_path),
                "status": "completed",
            },
        },
    )()


class AnnotationTests(unittest.TestCase):
    def test_load_run_dataset_records_reads_export_from_manifest(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        run_root = tmpdir / "runs" / "run_001"
        export_path = run_root / "exports" / "dataset.jsonl"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps({"record_id": "rec_00000001", "prompt_id": "p001", "artifact_type": "image"})
            + "\n",
            encoding="utf-8",
        )
        (run_root / "run_manifest.json").write_text(
            json.dumps({"run_id": "run_001", "export_path": str(export_path)}),
            encoding="utf-8",
        )

        records = load_run_dataset_records("run_001", runs_root=tmpdir / "runs")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], "rec_00000001")

    def test_annotate_run_builds_requests_and_writes_bundle(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        bundle_dir = tmpdir / "annotation_bundle"
        source_record = {
            "record_id": "rec_00000001",
            "run_id": "run_001",
            "task_id": "task_000001",
            "prompt_id": "p001",
            "prompt": "a calm lake at sunrise",
            "negative_prompt": "low quality",
            "language": "en",
            "model_name": "Z-Image",
            "task_type": "t2i",
            "artifact_type": "image",
            "artifact_path": "/tmp/source.png",
            "artifact_metadata": {"format": "png"},
            "generation_params": {"width": 1024},
            "prompt_metadata": {
                "prompt_template": "photorealistic_base",
                "prompt_style_family": "detailed_sentence",
                "annotation_hints": {"scene_category": "Nature"},
            },
        }

        def fake_run_single_model(**kwargs):
            prompt_file = Path(kwargs["prompt_file"])
            payload = json.loads(prompt_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("a calm lake at sunrise", payload["prompt"])
            self.assertIn('"scene_category": "Nature"', payload["prompt"])
            self.assertIn("/tmp/source.png", payload["prompt"])

            run_root = Path(kwargs["out_dir"])
            export_path = run_root / "exports" / "dataset.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path = run_root / "workdir" / "annreq_rec_00000001.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "summary": "A realistic outdoor scene.",
                        "labels": ["photorealistic"],
                        "confidence": 0.95,
                        "rationale": "Metadata and prompt are aligned.",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            export_path.write_text(
                json.dumps(
                    {
                        "prompt_id": "annreq_rec_00000001",
                        "artifact_path": str(artifact_path),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / "failures.json").write_text("[]", encoding="utf-8")
            return _build_summary(run_id="annotation_run_001", output_dir=run_root, export_path=export_path)

        with patch("whitzard.annotation.service.load_run_manifest", return_value={"run_id": "run_001", "manifest_path": str(tmpdir / "runs" / "run_001" / "run_manifest.json"), "export_path": str(tmpdir / "runs" / "run_001" / "exports" / "dataset.jsonl")}):
            with patch("whitzard.annotation.service.load_run_dataset_records", return_value=[source_record]):
                with patch("whitzard.annotation.service.run_single_model", side_effect=fake_run_single_model):
                    summary = annotate_run(
                        "run_001",
                        annotation_profile="default_review",
                        annotator_model="Qwen3-32B",
                        out_dir=bundle_dir,
                        execution_mode="mock",
                    )

        annotations_path = Path(summary.annotations_path)
        manifest_path = Path(summary.manifest_path)
        self.assertTrue(annotations_path.exists())
        self.assertTrue(manifest_path.exists())
        annotations = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(summary.annotated_count, 1)
        self.assertEqual(annotations[0]["source_record_id"], "rec_00000001")
        self.assertEqual(annotations[0]["annotation"]["summary"], "A realistic outdoor scene.")
        self.assertEqual(
            annotations[0]["source_prompt_metadata"]["annotation_hints"]["scene_category"],
            "Nature",
        )

    def test_annotate_run_skips_existing_annotation_records(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        bundle_dir = tmpdir / "annotation_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "annotations.jsonl").write_text(
            json.dumps({"source_record_id": "rec_00000001", "annotation": {"summary": "done"}}) + "\n",
            encoding="utf-8",
        )
        source_record = {
            "record_id": "rec_00000001",
            "prompt_id": "p001",
            "artifact_type": "image",
            "artifact_path": "/tmp/source.png",
            "prompt_metadata": {},
        }

        with patch("whitzard.annotation.service.load_run_manifest", return_value={"run_id": "run_001", "manifest_path": str(tmpdir / "run_manifest.json"), "export_path": str(tmpdir / "dataset.jsonl")}):
            with patch("whitzard.annotation.service.load_run_dataset_records", return_value=[source_record]):
                with patch("whitzard.annotation.service.run_single_model") as run_single_model:
                    summary = annotate_run(
                        "run_001",
                        annotation_profile="default_review",
                        annotator_model="Qwen3-32B",
                        out_dir=bundle_dir,
                        execution_mode="mock",
                    )

        run_single_model.assert_not_called()
        self.assertEqual(summary.annotated_count, 0)
        self.assertEqual(summary.skipped_count, 1)

    def test_annotate_run_accepts_remote_annotator_model(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        bundle_dir = tmpdir / "annotation_bundle"
        source_record = {
            "record_id": "rec_00000001",
            "run_id": "run_001",
            "prompt_id": "p001",
            "prompt": "Describe this scenario",
            "artifact_type": "text",
            "artifact_path": "/tmp/source.txt",
            "prompt_metadata": {},
        }

        def fake_run_single_model(**kwargs):
            self.assertEqual(kwargs["model_name"], "OpenAI-Compatible-Chat")
            run_root = Path(kwargs["out_dir"])
            export_path = run_root / "exports" / "dataset.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path = run_root / "workdir" / "annreq_rec_00000001.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "summary": "Remote annotator output.",
                        "labels": ["ok"],
                        "confidence": 0.8,
                        "rationale": "Remote provider returned structured JSON.",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            export_path.write_text(
                json.dumps(
                    {
                        "prompt_id": "annreq_rec_00000001",
                        "artifact_path": str(artifact_path),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / "failures.json").write_text("[]", encoding="utf-8")
            return _build_summary(run_id="annotation_run_remote", output_dir=run_root, export_path=export_path)

        with patch("whitzard.annotation.service.load_run_manifest", return_value={"run_id": "run_001", "manifest_path": str(tmpdir / "run_manifest.json"), "export_path": str(tmpdir / "dataset.jsonl")}):
            with patch("whitzard.annotation.service.load_run_dataset_records", return_value=[source_record]):
                with patch("whitzard.annotation.service.run_single_model", side_effect=fake_run_single_model):
                    summary = annotate_run(
                        "run_001",
                        annotation_profile="default_review",
                        annotator_model="OpenAI-Compatible-Chat",
                        out_dir=bundle_dir,
                        execution_mode="mock",
                    )

        annotations = [
            json.loads(line)
            for line in Path(summary.annotations_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary.annotated_count, 1)
        self.assertEqual(annotations[0]["annotator_model"], "OpenAI-Compatible-Chat")
        self.assertEqual(annotations[0]["annotation"]["summary"], "Remote annotator output.")
