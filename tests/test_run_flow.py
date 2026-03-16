import json
import tempfile
import unittest
from pathlib import Path

from aigc.run_flow import run_models, run_single_model
from aigc.runtime.payloads import TaskPayload
from aigc.runtime.worker import execute_task_payload


class FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class FakeEnvManager:
    def ensure_environment(self, model_name: str):
        return FakeEnvRecord()

    def inspect_model_environment(self, model_name: str):
        return FakeEnvRecord()


class RunFlowTests(unittest.TestCase):
    def test_minimal_run_wiring_creates_run_dir_and_export(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a futuristic city at night\n", encoding="utf-8")

        def fake_worker_runner(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            artifact_path = workdir / f"{task_payload['prompts'][0]['prompt_id']}.png"
            artifact_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x08\x02\x00\x00\x00")
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "plan": {"mode": "in_process"},
                "execution_result": {"exit_code": 0, "logs": "ok", "outputs": {}},
                "model_result": {
                    "status": "success",
                    "batch_items": [
                        {
                            "prompt_id": task_payload["prompts"][0]["prompt_id"],
                            "status": "success",
                            "artifacts": [
                                {
                                    "type": "image",
                                    "path": str(artifact_path),
                                    "metadata": {"width": 1, "height": 1, "format": "png"},
                                }
                            ],
                        }
                    ],
                    "logs": "ok",
                    "metadata": {},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 0, "ok"

        summary = run_single_model(
            model_name="Z-Image",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "demo",
            execution_mode="real",
            env_manager=FakeEnvManager(),
            worker_runner=fake_worker_runner,
        )

        self.assertTrue(Path(summary.output_dir).exists())
        self.assertEqual(summary.tasks_scheduled, 1)
        self.assertEqual(summary.records_exported, 1)
        self.assertEqual(summary.execution_mode, "real")
        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["execution_mode"], "real")
        self.assertEqual(manifest["status"], "completed")
        export_path = Path(summary.export_path)
        self.assertTrue(export_path.exists())
        record = json.loads(export_path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["model_name"], "Z-Image")
        self.assertEqual(record["execution_metadata"]["status"], "success")

    def test_multi_model_mock_run_batches_and_exports_records(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text(
            "\n".join(
                [
                    "a futuristic city at night",
                    "a cat sitting on a chair",
                    "一只可爱的猫",
                    "a watercolor mountain lake",
                    "a robot reading a book",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        def inprocess_worker_runner(_env_record, task_file: Path, result_file: Path):
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            result = execute_task_payload(payload)
            result_file.write_text(json.dumps(result), encoding="utf-8")
            return 0, "ok"

        summary = run_models(
            model_names=["Z-Image", "FLUX.1-dev"],
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "multi",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )

        self.assertEqual(summary.tasks_scheduled, 5)
        self.assertEqual(summary.records_exported, 10)
        export_lines = Path(summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(export_lines), 10)
        records = [json.loads(line) for line in export_lines]
        self.assertEqual(
            {record["model_name"] for record in records},
            {"Z-Image", "FLUX.1-dev"},
        )
        self.assertTrue(all(record["execution_metadata"]["execution_mode"] == "mock" for record in records))
        self.assertTrue(all(record["execution_metadata"]["mock"] for record in records))
        self.assertEqual(
            sorted(record["prompt_id"] for record in records if record["model_name"] == "Z-Image"),
            [f"prompt_{index:06d}" for index in range(1, 6)],
        )

        zimage_task_dir = Path(summary.output_dir) / "tasks" / "z-image"
        task_files = sorted(
            path for path in zimage_task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        first_task = json.loads(task_files[0].read_text(encoding="utf-8"))
        second_task = json.loads(task_files[1].read_text(encoding="utf-8"))
        self.assertEqual(len(first_task["prompts"]), 4)
        self.assertEqual(len(second_task["prompts"]), 1)
        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["execution_mode"], "mock")
        self.assertEqual(manifest["task_count"], 5)

    def test_run_models_supports_csv_and_jsonl_prompt_sources_in_mock_mode(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "example.csv"
        csv_path.write_text(
            "prompt_id,prompt,language\np001,a futuristic city,en\np002,一只可爱的猫,zh\n",
            encoding="utf-8",
        )
        jsonl_path = tmpdir / "example.jsonl"
        jsonl_path.write_text(
            "\n".join(
                [
                    json.dumps({"prompt_id": "j001", "prompt": "a red fox", "language": "en"}),
                    json.dumps({"prompt_id": "j002", "prompt": "一片雪山", "language": "zh"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        def inprocess_worker_runner(_env_record, task_file: Path, result_file: Path):
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            result = execute_task_payload(payload)
            result_file.write_text(json.dumps(result), encoding="utf-8")
            return 0, "ok"

        csv_summary = run_models(
            model_names=["Z-Image-Turbo"],
            prompt_file=csv_path,
            out_dir=tmpdir / "runs" / "csv",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )
        jsonl_summary = run_models(
            model_names=["Qwen-Image-2512"],
            prompt_file=jsonl_path,
            out_dir=tmpdir / "runs" / "jsonl",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )

        csv_records = [
            json.loads(line)
            for line in Path(csv_summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        jsonl_records = [
            json.loads(line)
            for line in Path(jsonl_summary.export_path)
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        self.assertEqual([record["prompt_id"] for record in csv_records], ["p001", "p002"])
        self.assertEqual([record["prompt_id"] for record in jsonl_records], ["j001", "j002"])
        self.assertTrue(all(record["execution_metadata"]["mock"] for record in csv_records))
        self.assertTrue(all(record["execution_metadata"]["mock"] for record in jsonl_records))

    def test_video_multi_model_mock_run_exports_video_records(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "video.txt"
        prompts_path.write_text(
            "a drone shot over snowy cliffs\nan astronaut walks across a red desert\n",
            encoding="utf-8",
        )

        def inprocess_worker_runner(_env_record, task_file: Path, result_file: Path):
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            result = execute_task_payload(payload)
            result_file.write_text(json.dumps(result), encoding="utf-8")
            return 0, "ok"

        summary = run_models(
            model_names=["Wan2.2-T2V-A14B-Diffusers", "LongCat-Video"],
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "video",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )

        self.assertEqual(summary.tasks_scheduled, 4)
        self.assertEqual(summary.records_exported, 4)
        records = [
            json.loads(line)
            for line in Path(summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual(
            {record["model_name"] for record in records},
            {"Wan2.2-T2V-A14B-Diffusers", "LongCat-Video"},
        )
        self.assertTrue(all(record["artifact_type"] == "video" for record in records))
        self.assertTrue(all(record["execution_metadata"]["execution_mode"] == "mock" for record in records))

    def test_video_run_supports_csv_and_jsonl_prompt_sources_in_mock_mode(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "video.csv"
        csv_path.write_text(
            "prompt_id,prompt,language\nv001,a surfer races toward a giant wave,en\nv002,一艘小船穿过晨雾,zh\n",
            encoding="utf-8",
        )
        jsonl_path = tmpdir / "video.jsonl"
        jsonl_path.write_text(
            "\n".join(
                [
                    json.dumps({"prompt_id": "w001", "prompt": "city lights reflected in rain", "language": "en"}),
                    json.dumps({"prompt_id": "w002", "prompt": "山谷上空缓慢移动的云层", "language": "zh"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        def inprocess_worker_runner(_env_record, task_file: Path, result_file: Path):
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            result = execute_task_payload(payload)
            result_file.write_text(json.dumps(result), encoding="utf-8")
            return 0, "ok"

        csv_summary = run_models(
            model_names=["Wan2.2-T2V-A14B-Diffusers"],
            prompt_file=csv_path,
            out_dir=tmpdir / "runs" / "video_csv",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )
        jsonl_summary = run_models(
            model_names=["MOVA-720p"],
            prompt_file=jsonl_path,
            out_dir=tmpdir / "runs" / "video_jsonl",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=inprocess_worker_runner,
        )

        csv_records = [
            json.loads(line)
            for line in Path(csv_summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        jsonl_records = [
            json.loads(line)
            for line in Path(jsonl_summary.export_path)
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        self.assertEqual([record["prompt_id"] for record in csv_records], ["v001", "v002"])
        self.assertEqual([record["prompt_id"] for record in jsonl_records], ["w001", "w002"])
        self.assertTrue(all(record["artifact_type"] == "video" for record in csv_records))
        self.assertTrue(all(record["artifact_type"] == "video" for record in jsonl_records))
