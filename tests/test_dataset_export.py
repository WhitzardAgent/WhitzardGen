import json
import tempfile
import unittest
from pathlib import Path

from whitzard.exporters import (
    ExportBundleSource,
    build_dataset_records,
    export_dataset_bundle,
    export_dataset_bundle_for_runs,
    export_jsonl,
)
from whitzard.registry import load_registry
from whitzard.runtime.payloads import TaskPayload, TaskPrompt


class DatasetExportTests(unittest.TestCase):
    def test_jsonl_export_shape(self) -> None:
        registry = load_registry()
        model = registry.get_model("Z-Image")
        payload = TaskPayload(
            task_id="task_000001",
            model_name="Z-Image",
            execution_mode="mock",
            prompts=[
                TaskPrompt(
                    prompt_id="p001",
                    prompt="a futuristic city",
                    language="en",
                    negative_prompt="blurry",
                    metadata={"split": "train"},
                )
            ],
            params={"width": 1024, "height": 1024},
            workdir="/tmp/task_000001",
        )
        task_result = {
            "model_result": {
                "status": "success",
                "batch_items": [
                    {
                        "prompt_id": "p001",
                        "status": "success",
                        "metadata": {
                            "batch_id": "batch_001",
                            "batch_index": 0,
                            "execution_mode": "mock",
                            "mock": True,
                        },
                        "artifacts": [
                            {
                                "type": "image",
                                "path": "/tmp/task_000001/p001.png",
                                "metadata": {"width": 1024, "height": 1024, "format": "png"},
                            }
                        ],
                    }
                ],
            }
        }

        records = build_dataset_records(
            run_id="run_test",
            model=model,
            task_payload=payload,
            task_result=task_result,
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["run_id"], "run_test")
        self.assertEqual(record["task_id"], "task_000001")
        self.assertEqual(record["prompt_id"], "p001")
        self.assertEqual(record["model_name"], "Z-Image")
        self.assertEqual(record["artifact_type"], "image")
        self.assertEqual(record["execution_metadata"]["status"], "success")
        self.assertEqual(record["execution_metadata"]["batch_id"], "batch_001")
        self.assertEqual(record["execution_metadata"]["batch_index"], 0)
        self.assertEqual(record["execution_metadata"]["execution_mode"], "mock")
        self.assertTrue(record["execution_metadata"]["mock"])

        output_path = Path(tempfile.mkdtemp()) / "dataset.jsonl"
        export_jsonl(records, output_path)
        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["record_id"], "rec_00000001")

    def test_video_jsonl_export_shape(self) -> None:
        registry = load_registry()
        model = registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        payload = TaskPayload(
            task_id="task_000010",
            model_name="Wan2.2-T2V-A14B-Diffusers",
            execution_mode="mock",
            prompts=[
                TaskPrompt(
                    prompt_id="v001",
                    prompt="ocean waves crashing on black rocks",
                    language="en",
                    metadata={"split": "validation"},
                )
            ],
            params={"width": 1280, "height": 720, "fps": 16, "num_frames": 81},
            workdir="/tmp/task_000010",
            batch_id="wan_batch_001",
        )
        task_result = {
            "model_result": {
                "status": "success",
                "batch_items": [
                    {
                        "prompt_id": "v001",
                        "status": "success",
                        "metadata": {
                            "batch_id": "wan_batch_001",
                            "batch_index": 0,
                            "execution_mode": "mock",
                            "mock": True,
                        },
                        "artifacts": [
                            {
                                "type": "video",
                                "path": "/tmp/task_000010/v001.mp4",
                                "metadata": {
                                    "width": 1280,
                                    "height": 720,
                                    "fps": 16,
                                    "num_frames": 81,
                                    "format": "mp4",
                                },
                            }
                        ],
                    }
                ],
            }
        }

        records = build_dataset_records(
            run_id="run_video",
            model=model,
            task_payload=payload,
            task_result=task_result,
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["artifact_type"], "video")
        self.assertEqual(record["model_name"], "Wan2.2-T2V-A14B-Diffusers")
        self.assertEqual(record["execution_metadata"]["batch_id"], "wan_batch_001")
        self.assertEqual(record["execution_metadata"]["execution_mode"], "mock")
        self.assertTrue(record["execution_metadata"]["mock"])

    def test_export_dataset_bundle_link_mode_organizes_media_and_manifest(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        source_artifact = tmpdir / "source.png"
        source_artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
        source_dataset = tmpdir / "dataset.jsonl"
        source_record = {
            "record_id": "rec_00000001",
            "run_id": "run_test",
            "task_id": "task_000001",
            "prompt_id": "p001",
            "prompt": "a futuristic city",
            "negative_prompt": "blurry",
            "language": "en",
            "model_name": "Z-Image",
            "model_version": "1.0",
            "adapter_name": "ZImageAdapter",
            "modality": "image",
            "task_type": "t2i",
            "artifact_type": "image",
            "artifact_path": str(source_artifact),
            "artifact_metadata": {"width": 1024, "height": 1024, "format": "png"},
            "generation_params": {"width": 1024},
            "prompt_metadata": {"split": "train"},
            "execution_metadata": {"status": "success", "execution_mode": "mock"},
        }
        source_dataset.write_text(json.dumps(source_record, ensure_ascii=False) + "\n", encoding="utf-8")

        result = export_dataset_bundle(
            run_id="run_test",
            source_manifest={
                "run_id": "run_test",
                "status": "completed",
                "parent_run_id": "run_parent",
                "source_run_id": "run_source",
                "recovery_mode": "retry",
                "manifest_path": str(tmpdir / "run_manifest.json"),
            },
            source_dataset_path=source_dataset,
            bundle_root=tmpdir / "bundle",
            mode="link",
        )

        self.assertEqual(result.record_count, 1)
        self.assertEqual(result.export_mode, "link")
        self.assertEqual(result.counts_by_split, {"train": 1})
        dataset_lines = Path(result.dataset_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(dataset_lines), 1)
        exported_record = json.loads(dataset_lines[0])
        self.assertEqual(exported_record["artifact_path"], "media/train/Z-Image/image/run_test__rec_00000001.png")
        self.assertEqual(exported_record["split"], "train")
        self.assertEqual(exported_record["source_artifact_path"], str(source_artifact))
        self.assertEqual(exported_record["source_run_lineage"]["parent_run_id"], "run_parent")
        exported_artifact = Path(result.bundle_path) / exported_record["artifact_path"]
        self.assertTrue(exported_artifact.is_symlink())
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["record_count"], 1)
        self.assertEqual(manifest["export_mode"], "link")
        self.assertEqual(manifest["counts_by_artifact_type"], {"image": 1})
        self.assertEqual(manifest["counts_by_split"], {"train": 1})
        readme_text = Path(result.readme_path).read_text(encoding="utf-8")
        self.assertIn("# Export Bundle: bundle", readme_text)
        self.assertIn("## Dataset Card Summary", readme_text)
        self.assertIn("## Counts By Split", readme_text)

    def test_export_dataset_bundle_copy_mode_copies_artifact(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        source_artifact = tmpdir / "source.mp4"
        source_artifact.write_bytes(b"mp4")
        source_dataset = tmpdir / "dataset.jsonl"
        source_dataset.write_text(
            json.dumps(
                {
                    "record_id": "rec_00000002",
                    "run_id": "run_video",
                    "task_id": "task_000010",
                    "prompt_id": "v001",
                    "prompt": "ocean waves",
                    "language": "en",
                    "model_name": "Wan2.2-T2V-A14B-Diffusers",
                    "model_version": "1.0",
                    "adapter_name": "WanT2VDiffusersAdapter",
                    "modality": "video",
                    "task_type": "t2v",
                    "artifact_type": "video",
                    "artifact_path": str(source_artifact),
                    "artifact_metadata": {"format": "mp4"},
                    "generation_params": {"fps": 16},
                    "prompt_metadata": {},
                    "execution_metadata": {"status": "success", "execution_mode": "mock"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        result = export_dataset_bundle(
            run_id="run_video",
            source_manifest={"run_id": "run_video", "status": "completed"},
            source_dataset_path=source_dataset,
            bundle_root=tmpdir / "bundle_copy",
            mode="copy",
        )

        dataset_record = json.loads(Path(result.dataset_path).read_text(encoding="utf-8").strip())
        exported_artifact = Path(result.bundle_path) / dataset_record["artifact_path"]
        self.assertTrue(exported_artifact.exists())
        self.assertFalse(exported_artifact.is_symlink())
        self.assertEqual(exported_artifact.read_bytes(), b"mp4")
        self.assertEqual(dataset_record["split"], "unspecified")

    def test_export_dataset_bundle_skips_failed_or_missing_artifacts(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        source_artifact = tmpdir / "ok.png"
        source_artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
        source_dataset = tmpdir / "dataset.jsonl"
        records = [
            {
                "record_id": "rec_00000001",
                "run_id": "run_test",
                "task_id": "task_000001",
                "prompt_id": "p001",
                "prompt": "ok",
                "language": "en",
                "model_name": "Z-Image",
                "artifact_type": "image",
                "artifact_path": str(source_artifact),
                "generation_params": {},
                "prompt_metadata": {},
                "execution_metadata": {"status": "success"},
            },
            {
                "record_id": "rec_00000002",
                "run_id": "run_test",
                "task_id": "task_000002",
                "prompt_id": "p002",
                "prompt": "failed",
                "language": "en",
                "model_name": "Z-Image",
                "artifact_type": "image",
                "artifact_path": str(tmpdir / "missing.png"),
                "generation_params": {},
                "prompt_metadata": {},
                "execution_metadata": {"status": "failed"},
            },
            {
                "record_id": "rec_00000003",
                "run_id": "run_test",
                "task_id": "task_000003",
                "prompt_id": "p003",
                "prompt": "missing",
                "language": "en",
                "model_name": "Z-Image",
                "artifact_type": "image",
                "artifact_path": str(tmpdir / "also_missing.png"),
                "generation_params": {},
                "prompt_metadata": {},
                "execution_metadata": {"status": "success"},
            },
        ]
        source_dataset.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

        result = export_dataset_bundle(
            run_id="run_test",
            source_manifest={"run_id": "run_test", "status": "completed"},
            source_dataset_path=source_dataset,
            bundle_root=tmpdir / "bundle_filtered",
            mode="link",
        )

        dataset_lines = Path(result.dataset_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(dataset_lines), 1)
        self.assertEqual(result.skipped_count, 2)
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["record_count"], 1)
        self.assertEqual(manifest["skipped_record_count"], 2)

    def test_export_dataset_bundle_for_runs_merges_runs_filters_models_and_preserves_lineage(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        run1_artifact = tmpdir / "run1.png"
        run1_artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
        run2_artifact = tmpdir / "run2.mp4"
        run2_artifact.write_bytes(b"mp4")
        run1_dataset = tmpdir / "run1.jsonl"
        run2_dataset = tmpdir / "run2.jsonl"
        run1_dataset.write_text(
            json.dumps(
                {
                    "record_id": "rec_00000001",
                    "run_id": "run_001",
                    "task_id": "task_000001",
                    "prompt_id": "p001",
                    "prompt": "cat",
                    "language": "en",
                    "model_name": "Z-Image",
                    "model_version": "1.0",
                    "adapter_name": "ZImageAdapter",
                    "modality": "image",
                    "task_type": "t2i",
                    "artifact_type": "image",
                    "artifact_path": str(run1_artifact),
                    "artifact_metadata": {"format": "png"},
                    "generation_params": {},
                    "prompt_metadata": {"split": "train"},
                    "execution_metadata": {"status": "success", "execution_mode": "mock"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        run2_dataset.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "record_id": "rec_00000001",
                            "run_id": "run_002",
                            "task_id": "task_000010",
                            "prompt_id": "v001",
                            "prompt": "waves",
                            "language": "en",
                            "model_name": "Wan2.2-T2V-A14B-Diffusers",
                            "model_version": "1.0",
                            "adapter_name": "WanT2VDiffusersAdapter",
                            "modality": "video",
                            "task_type": "t2v",
                            "artifact_type": "video",
                            "artifact_path": str(run2_artifact),
                            "artifact_metadata": {"format": "mp4"},
                            "generation_params": {},
                            "prompt_metadata": {"split": "val"},
                            "execution_metadata": {"status": "success", "execution_mode": "mock"},
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "record_id": "rec_00000002",
                            "run_id": "run_002",
                            "task_id": "task_000011",
                            "prompt_id": "v002",
                            "prompt": "ignored",
                            "language": "en",
                            "model_name": "FLUX.1-dev",
                            "model_version": "1.0",
                            "adapter_name": "FluxAdapter",
                            "modality": "image",
                            "task_type": "t2i",
                            "artifact_type": "image",
                            "artifact_path": str(tmpdir / "missing.png"),
                            "artifact_metadata": {"format": "png"},
                            "generation_params": {},
                            "prompt_metadata": {"split": "val"},
                            "execution_metadata": {"status": "success", "execution_mode": "mock"},
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = export_dataset_bundle_for_runs(
            sources=[
                ExportBundleSource(
                    run_id="run_001",
                    source_manifest={"run_id": "run_001", "status": "completed"},
                    source_dataset_path=str(run1_dataset),
                ),
                ExportBundleSource(
                    run_id="run_002",
                    source_manifest={
                        "run_id": "run_002",
                        "status": "completed",
                        "parent_run_id": "run_parent",
                        "source_run_id": "run_source",
                        "recovery_mode": "resume",
                    },
                    source_dataset_path=str(run2_dataset),
                ),
            ],
            bundle_root=tmpdir / "merged_bundle",
            mode="link",
            selected_models=["Z-Image", "Wan2.2-T2V-A14B-Diffusers"],
        )

        self.assertEqual(result.source_run_ids, ["run_001", "run_002"])
        self.assertEqual(result.selected_models, ["Wan2.2-T2V-A14B-Diffusers", "Z-Image"])
        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.filtered_out_count, 1)
        self.assertEqual(result.counts_by_split, {"train": 1, "val": 1})
        self.assertEqual(result.counts_by_model, {"Wan2.2-T2V-A14B-Diffusers": 1, "Z-Image": 1})
        self.assertEqual(result.counts_by_run_id, {"run_001": 1, "run_002": 1})
        records = [
            json.loads(line)
            for line in Path(result.dataset_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual([record["record_id"] for record in records], ["rec_00000001", "rec_00000002"])
        self.assertEqual(records[0]["source_record_id"], "rec_00000001")
        self.assertIn(records[0]["split"], {"train", "val"})
        self.assertTrue(
            all(
                Path(result.bundle_path, record["artifact_path"]).exists()
                for record in records
            )
        )
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["source_runs"]), 2)
        self.assertEqual(manifest["selected_models"], ["Wan2.2-T2V-A14B-Diffusers", "Z-Image"])
        self.assertEqual(manifest["filtered_out_count"], 1)
        readme_text = Path(result.readme_path).read_text(encoding="utf-8")
        self.assertIn("Source runs: run_001, run_002", readme_text)
        self.assertIn("## Source Runs", readme_text)
        self.assertIn("## Counts By Run", readme_text)
        self.assertIn("## Counts By Model", readme_text)
