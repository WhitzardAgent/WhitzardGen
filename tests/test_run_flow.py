import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from aigc.run_flow import (
    RunFlowError,
    _assign_gpus_to_replicas,
    _calculate_replica_count,
    _limit_available_gpus_for_model,
    _shard_tasks_across_replicas,
    run_models,
    run_single_model,
)
from aigc.registry import load_registry
from aigc.runtime.payloads import TaskPayload
from aigc.runtime.worker import execute_task_payload
from aigc.utils.progress import TextRunProgress


class FakeEnvRecord:
    env_id = "env_test"
    state = "ready"


class FakeEnvManager:
    def ensure_ready(self, model_name: str, foreground: bool = True, progress=None):
        return FakeEnvRecord()

    def ensure_environment(self, model_name: str):
        return FakeEnvRecord()

    def inspect_model_environment(self, model_name: str):
        return FakeEnvRecord()


class RunFlowTests(unittest.TestCase):
    def test_replica_count_calculation_uses_gpu_count_and_gpus_per_replica(self) -> None:
        self.assertEqual(
            _calculate_replica_count(
                total_available_gpus=8,
                gpus_per_replica=2,
                supports_multi_replica=True,
                execution_mode="real",
                task_count=10,
            ),
            4,
        )
        self.assertEqual(
            _calculate_replica_count(
                total_available_gpus=1,
                gpus_per_replica=2,
                supports_multi_replica=True,
                execution_mode="real",
                task_count=10,
            ),
            1,
        )

    def test_gpu_assignment_and_task_sharding_are_deterministic(self) -> None:
        self.assertEqual(
            _assign_gpus_to_replicas(
                available_gpus=[0, 1, 2, 3, 4, 5, 6, 7],
                gpus_per_replica=2,
                replica_count=4,
            ),
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )

        prepared = [f"task_{index}" for index in range(1, 7)]
        shards = _shard_tasks_across_replicas(
            prepared_tasks=prepared,  # type: ignore[arg-type]
            replica_count=3,
        )
        self.assertEqual(shards, [["task_1", "task_4"], ["task_2", "task_5"], ["task_3", "task_6"]])

    def test_available_gpu_list_is_capped_by_model_max_gpus(self) -> None:
        registry = load_registry()
        model = registry.get_model("Wan2.2-T2V-A14B-Diffusers")

        self.assertEqual(
            _limit_available_gpus_for_model([0, 1, 2, 3, 4, 5, 6, 7], model),
            [0, 1, 2, 3, 4, 5, 6, 7],
        )

        capped_model = type(
            "ModelStub",
            (),
            {"max_gpus": 4},
        )()
        self.assertEqual(
            _limit_available_gpus_for_model([0, 1, 2, 3, 4, 5, 6, 7], capped_model),  # type: ignore[arg-type]
            [0, 1, 2, 3],
        )

    def test_selected_model_uses_persistent_worker_strategy_in_mock_mode(self) -> None:
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

        summary = run_single_model(
            model_name="Z-Image",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "persistent_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
        )

        task_dir = Path(summary.output_dir) / "tasks" / "z-image"
        task_files = sorted(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        self.assertEqual(len(task_files), 2)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in task_files]
        self.assertTrue(all(payload["worker_strategy"] == "persistent_worker" for payload in payloads))

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["per_model_summary"]["Z-Image"]["worker_strategy"],
            "persistent_worker",
        )
        running_log = Path(summary.output_dir) / "running.log"
        self.assertTrue(running_log.exists())
        running_log_text = running_log.read_text(encoding="utf-8")
        self.assertRegex(running_log_text, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[run\] Starting run")
        self.assertIn("[worker][Z-Image][replica=0] GPUs=[] starting persistent worker", running_log_text)

    def test_selected_video_model_uses_persistent_worker_strategy_in_mock_mode(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "video.txt"
        prompts_path.write_text("a camera glides through a neon tunnel\n", encoding="utf-8")

        summary = run_single_model(
            model_name="CogVideoX-5B",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "persistent_video_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
        )

        task_dir = Path(summary.output_dir) / "tasks" / "cogvideox-5b"
        task_file = next(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        payload = json.loads(task_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["worker_strategy"], "persistent_worker")

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["per_model_summary"]["CogVideoX-5B"]["worker_strategy"],
            "persistent_worker",
        )

    def test_multi_replica_mock_run_shards_image_tasks_and_exports_replica_metadata(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "replica_image.txt"
        prompts_path.write_text(
            "\n".join([f"image test prompt {index:03d}" for index in range(1, 10)]) + "\n",
            encoding="utf-8",
        )
        progress_stream = StringIO()

        with patch.dict(os.environ, {"AIGC_AVAILABLE_GPUS": "0,1"}, clear=False):
            summary = run_single_model(
                model_name="Z-Image",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "replica_image",
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                batch_limit=1,
                progress=TextRunProgress(stream=progress_stream),
            )

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        per_model = manifest["per_model_summary"]["Z-Image"]
        self.assertEqual(per_model["replica_count"], 2)
        self.assertEqual(
            [replica["gpu_assignment"] for replica in per_model["replicas"]],
            [[0], [1]],
        )
        self.assertEqual(
            [replica["task_count"] for replica in per_model["replicas"]],
            [5, 4],
        )

        records = [
            json.loads(line)
            for line in Path(summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual({record["execution_metadata"]["replica_id"] for record in records}, {0, 1})
        self.assertEqual(
            {tuple(record["execution_metadata"]["gpu_assignment"]) for record in records},
            {(0,), (1,)},
        )

        progress_text = progress_stream.getvalue()
        self.assertIn("[run][Z-Image] available_gpus=[0, 1]", progress_text)
        self.assertIn("[run][Z-Image] starting 2 replicas", progress_text)
        self.assertIn("[run][Z-Image] replica=0 assigned 5 tasks GPUs=[0]", progress_text)
        self.assertIn("[run][Z-Image] replica=1 assigned 4 tasks GPUs=[1]", progress_text)

    def test_multi_replica_mock_run_shards_video_tasks_and_uses_gpu_groups(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "replica_video.txt"
        prompts_path.write_text(
            "\n".join([f"video test prompt {index:03d}" for index in range(1, 5)]) + "\n",
            encoding="utf-8",
        )
        progress_stream = StringIO()

        with patch.dict(os.environ, {"AIGC_AVAILABLE_GPUS": "0,1,2,3"}, clear=False):
            summary = run_single_model(
                model_name="CogVideoX-5B",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "replica_video",
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                progress=TextRunProgress(stream=progress_stream),
            )

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        per_model = manifest["per_model_summary"]["CogVideoX-5B"]
        self.assertEqual(per_model["replica_count"], 4)
        self.assertEqual(
            [replica["gpu_assignment"] for replica in per_model["replicas"]],
            [[0], [1], [2], [3]],
        )
        self.assertEqual(
            [replica["task_count"] for replica in per_model["replicas"]],
            [1, 1, 1, 1],
        )

        records = [
            json.loads(line)
            for line in Path(summary.export_path).read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual({record["execution_metadata"]["replica_id"] for record in records}, {0, 1, 2, 3})
        self.assertEqual(
            {tuple(record["execution_metadata"]["gpu_assignment"]) for record in records},
            {(0,), (1,), (2,), (3,)},
        )

        progress_text = progress_stream.getvalue()
        self.assertIn("[run][CogVideoX-5B] available_gpus=[0, 1, 2, 3]", progress_text)
        self.assertIn("[run][CogVideoX-5B] gpus_per_replica=1", progress_text)
        self.assertIn("[run][CogVideoX-5B] starting 4 replicas", progress_text)

    def test_default_run_root_uses_runtime_settings(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a calm lake at sunrise\n", encoding="utf-8")
        runtime_config = tmpdir / "local_runtime.yaml"
        configured_root = tmpdir / "shared_runs"
        runtime_config.write_text(
            f"paths:\n  runs_root: {configured_root}\n",
            encoding="utf-8",
        )

        def fake_worker_runner(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            artifact_path = workdir / f"{task_payload['prompts'][0]['prompt_id']}.png"
            artifact_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + b"\x00\x00\x00\x01"
                + b"\x00\x00\x00\x01"
                + b"\x08\x02\x00\x00\x00"
            )
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
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

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(runtime_config)}, clear=False):
            summary = run_single_model(
                model_name="Z-Image-Turbo",
                prompt_file=prompts_path,
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                worker_runner=fake_worker_runner,
            )

        self.assertTrue(summary.output_dir.startswith(str(configured_root)))
        self.assertTrue(Path(summary.output_dir).exists())

    def test_real_run_prefers_worker_result_logs_over_generic_wrapper_error(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a futuristic city at night\n", encoding="utf-8")

        def failing_worker_runner(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
                "plan": None,
                "execution_result": {
                    "exit_code": 1,
                    "logs": "Traceback (most recent call last):\nRuntimeError: real root cause",
                    "outputs": {},
                },
                "model_result": {
                    "status": "failed",
                    "batch_items": [],
                    "logs": "Traceback (most recent call last):\nRuntimeError: real root cause",
                    "metadata": {},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 1, "ERROR conda.cli.main_run:execute(125): generic wrapper failure"

        with self.assertRaises(RunFlowError) as context:
            run_single_model(
                model_name="Z-Image-Turbo",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "worker_failure",
                execution_mode="real",
                env_manager=FakeEnvManager(),
                worker_runner=failing_worker_runner,
            )

        message = str(context.exception)
        self.assertIn("real root cause", message)
        self.assertIn("generic wrapper failure", message)

    def test_real_run_uses_foreground_ensure_ready_when_env_is_needed(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a futuristic city at night\n", encoding="utf-8")

        class TrackingEnvManager(FakeEnvManager):
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def ensure_ready(self, model_name: str, foreground: bool = True, progress=None):
                self.calls.append((model_name, foreground))
                return FakeEnvRecord()

        manager = TrackingEnvManager()

        def fake_worker_runner(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            artifact_path = workdir / f"{task_payload['prompts'][0]['prompt_id']}.png"
            artifact_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + b"\x00\x00\x00\x01"
                + b"\x00\x00\x00\x01"
                + b"\x08\x02\x00\x00\x00"
            )
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
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

        run_single_model(
            model_name="Z-Image-Turbo",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "ensure_ready",
            execution_mode="real",
            env_manager=manager,
            worker_runner=fake_worker_runner,
        )

        self.assertEqual(manager.calls, [("Z-Image-Turbo", True)])

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
        running_log = Path(summary.output_dir) / "running.log"
        self.assertTrue(running_log.exists())
        running_log_text = running_log.read_text(encoding="utf-8")
        self.assertRegex(running_log_text, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[1/9\] Loading prompts\.\.\.")
        self.assertIn("Run complete", running_log_text)
        self.assertIn("running_log:", running_log_text)
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
