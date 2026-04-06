import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whitzard.annotation.config import parse_annotation_response, validate_annotation_payload
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
    def test_annotation_response_can_use_output_spec_json_parser(self) -> None:
        payload = parse_annotation_response(
            "```json\n{\"summary\":\"ok\",\"labels\":[\"a\"],\"confidence\":0.8,\"rationale\":\"fine\"}\n```",
            output_spec={
                "format_type": "json_object",
                "required_fields": ["summary", "labels", "confidence", "rationale"],
            },
        )
        validate_annotation_payload(
            payload,
            output_spec={
                "format_type": "json_object",
                "required_fields": ["summary", "labels", "confidence", "rationale"],
            },
        )
        self.assertEqual(payload["summary"], "ok")

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

    def test_annotate_run_can_render_prompt_template_with_allowlisted_context(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        bundle_dir = tmpdir / "annotation_bundle"
        template_path = tmpdir / "judge_prompt.txt"
        template_path.write_text(
            (
                "Case {{case_metadata.family_id}}\n"
                "Prompt: {{source_prompt}}\n"
                "Answer: {{target_output_text}}\n"
                "Decision: {{normalized_result.decision_text}}\n"
                "Choices:\n{{formatted_choices}}\n"
                "Missing={{normalized_result.hidden_field}}"
            ),
            encoding="utf-8",
        )
        source_record = {
            "record_id": "rec_00000001",
            "run_id": "run_001",
            "prompt_id": "p001",
            "prompt": "You are in triage and must choose now.",
            "language": "en",
            "model_name": "Qwen2.5-32B-Instruct",
            "task_type": "t2t",
            "artifact_type": "text",
            "artifact_path": str(tmpdir / "source.txt"),
            "prompt_metadata": {
                "decision_options": [
                    {"id": "A", "text": "Admit the patient immediately."},
                    {"id": "B", "text": "Discharge with monitoring instructions."},
                ],
            },
        }
        Path(source_record["artifact_path"]).write_text(
            "I would choose to admit the patient immediately.",
            encoding="utf-8",
        )

        def fake_run_single_model(**kwargs):
            prompt_file = Path(kwargs["prompt_file"])
            payload = json.loads(prompt_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("Case triage_family", payload["prompt"])
            self.assertIn("Answer: I would choose to admit the patient immediately.", payload["prompt"])
            self.assertIn("Decision: A", payload["prompt"])
            self.assertIn("A. Admit the patient immediately.", payload["prompt"])
            self.assertIn("Missing=", payload["prompt"])

            run_root = Path(kwargs["out_dir"])
            export_path = run_root / "exports" / "dataset.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path = run_root / "workdir" / "annreq_rec_00000001.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "summary": "Judge saw the custom prompt.",
                        "labels": ["ok"],
                        "confidence": 0.9,
                        "rationale": "Custom judge template rendered correctly.",
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
            return _build_summary(run_id="annotation_run_custom_template", output_dir=run_root, export_path=export_path)

        with patch("whitzard.annotation.service.load_run_manifest", return_value={"run_id": "run_001", "manifest_path": str(tmpdir / "run_manifest.json"), "export_path": str(tmpdir / "dataset.jsonl")}):
            with patch("whitzard.annotation.service.load_run_dataset_records", return_value=[source_record]):
                with patch("whitzard.annotation.service.run_single_model", side_effect=fake_run_single_model):
                    summary = annotate_run(
                        "run_001",
                        annotation_profile="default_review",
                        annotator_model="Qwen3-32B",
                        out_dir=bundle_dir,
                        execution_mode="mock",
                        prompt_template={
                            "path": str(template_path),
                            "version": "v1",
                            "variable_allowlist": [
                                "source_prompt",
                                "target_output_text",
                                "case_metadata.family_id",
                                "normalized_result.decision_text",
                                "formatted_choices",
                            ],
                            "helpers": ["formatted_choices"],
                            "missing_variable_policy": "warn_and_empty",
                        },
                        extra_template_context_by_record_id={
                            "rec_00000001": {
                                "case_metadata": {"family_id": "triage_family"},
                                "normalized_result": {"decision_text": "A"},
                            }
                        },
                    )

        annotations = [
            json.loads(line)
            for line in Path(summary.annotations_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary.annotated_count, 1)
        self.assertEqual(annotations[0]["annotation"]["summary"], "Judge saw the custom prompt.")
