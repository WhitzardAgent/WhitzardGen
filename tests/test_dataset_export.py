import json
import tempfile
import unittest
from pathlib import Path

from aigc.exporters import build_dataset_records, export_jsonl
from aigc.registry import load_registry
from aigc.runtime.payloads import TaskPayload, TaskPrompt


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
