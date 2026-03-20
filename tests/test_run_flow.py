import json
import os
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from aigc.env.manager import EnvManagerError
from aigc.run_ledger import RunLedgerWriter
from aigc.run_flow import (
    FailureBudget,
    FailurePolicy,
    PreparedTask,
    ReplicaPlan,
    RunFlowError,
    TaskExecutionOutcome,
    _handle_runtime_event,
    _run_persistent_worker_replicas,
    _default_generation_params,
    _assign_gpus_to_replicas,
    _calculate_replica_count,
    _limit_available_gpus_for_model,
    _shard_tasks_across_replicas,
    run_models,
    run_single_model,
)
from aigc.registry import load_registry
from aigc.runtime.payloads import TaskPayload, TaskPrompt
from aigc.runtime_telemetry import RunTelemetry
from aigc.runtime.worker import execute_task_payload
from aigc.utils.progress import TextRunProgress
from aigc.utils.runtime_logging import RunLogger


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
    @staticmethod
    def _inprocess_worker_runner(_env_record, task_file: Path, result_file: Path):
        payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
        result = execute_task_payload(payload)
        result_file.write_text(json.dumps(result), encoding="utf-8")
        return 0, "ok"

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

    def test_multi_model_manifest_includes_profile_and_effective_model_summary(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "image_multi.txt"
        prompts_path.write_text("a futuristic city at night\na cat sitting on a chair\n", encoding="utf-8")

        summary = run_models(
            model_names=["Z-Image", "FLUX.1-dev"],
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "multi_model_manifest",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=self._inprocess_worker_runner,
            profile_name="image_mock",
            profile_path="configs/run_profiles/image_mock.yaml",
            profile_generation_defaults={"width": 1024, "num_inference_steps": 40},
            profile_runtime={"available_gpus": [0, 1]},
        )

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["profile"],
            {
                "name": "image_mock",
                "path": "configs/run_profiles/image_mock.yaml",
                "generation_defaults": {"width": 1024, "num_inference_steps": 40},
                "runtime": {"available_gpus": [0, 1]},
            },
        )
        self.assertEqual(manifest["models"], ["Z-Image", "FLUX.1-dev"])
        self.assertEqual(manifest["per_model_summary"]["Z-Image"]["conda_env_name"], "zimage")
        self.assertIn(
            "supports_batch_prompts",
            manifest["per_model_summary"]["FLUX.1-dev"],
        )
        self.assertIn("local_paths", manifest["per_model_summary"]["FLUX.1-dev"])

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

    def test_selected_wan_video_model_uses_persistent_worker_strategy_in_mock_mode(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "wan_video.txt"
        prompts_path.write_text("two cats spar under a spotlight\n", encoding="utf-8")

        summary = run_single_model(
            model_name="Wan2.2-T2V-A14B-Diffusers",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "persistent_wan_video_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
        )

        task_dir = Path(summary.output_dir) / "tasks" / "wan2_2-t2v-a14b-diffusers"
        task_file = next(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        payload = json.loads(task_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["worker_strategy"], "persistent_worker")

        manifest = json.loads((Path(summary.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["per_model_summary"]["Wan2.2-T2V-A14B-Diffusers"]["worker_strategy"],
            "persistent_worker",
        )

    def test_wan_mock_run_batches_two_prompts_into_one_task(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "wan_batch.txt"
        prompts_path.write_text(
            "two cats spar under a spotlight\ncamera pans across a misty harbor\n",
            encoding="utf-8",
        )

        summary = run_single_model(
            model_name="Wan2.2-T2V-A14B-Diffusers",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "wan_batch_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
        )

        task_dir = Path(summary.output_dir) / "tasks" / "wan2_2-t2v-a14b-diffusers"
        task_files = sorted(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        self.assertEqual(len(task_files), 1)
        payload = json.loads(task_files[0].read_text(encoding="utf-8"))
        self.assertEqual(len(payload["prompts"]), 2)

    def test_cogvideox_mock_run_batches_two_prompts_into_one_task(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "cog_batch.txt"
        prompts_path.write_text(
            "neon tunnel flythrough\nslow orbit around a crystal tower\n",
            encoding="utf-8",
        )

        summary = run_single_model(
            model_name="CogVideoX-5B",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "cog_batch_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
        )

        task_dir = Path(summary.output_dir) / "tasks" / "cogvideox-5b"
        task_files = sorted(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        self.assertEqual(len(task_files), 1)
        payload = json.loads(task_files[0].read_text(encoding="utf-8"))
        self.assertEqual(len(payload["prompts"]), 2)

    def test_longcat_mock_run_batches_two_prompts_into_one_task(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "longcat_batch.txt"
        prompts_path.write_text(
            "a train crosses a snowy bridge at dawn\nslow dolly through a lantern-lit alley\n",
            encoding="utf-8",
        )

        summary = run_single_model(
            model_name="LongCat-Video",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "longcat_batch_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=self._inprocess_worker_runner,
        )

        task_dir = Path(summary.output_dir) / "tasks" / "longcat-video"
        task_files = sorted(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        self.assertEqual(len(task_files), 1)
        payload = json.loads(task_files[0].read_text(encoding="utf-8"))
        self.assertEqual(len(payload["prompts"]), 2)
        self.assertEqual(payload["params"]["guidance_scale"], 4.0)

    def test_default_generation_params_omit_seed_without_global_default(self) -> None:
        registry = load_registry()
        model = registry.get_model("Z-Image")
        prompt = type(
            "PromptStub",
            (),
            {"prompt": "a city at night", "language": "en", "negative_prompt": None, "parameters": {}, "metadata": {}},
        )()

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(Path(tempfile.mkdtemp()) / "missing.yaml")}, clear=False):
            params = _default_generation_params(model, [prompt])  # type: ignore[list-item]

        self.assertNotIn("seed", params)

    def test_default_generation_params_honor_global_default_seed(self) -> None:
        registry = load_registry()
        model = registry.get_model("Z-Image")
        prompt = type(
            "PromptStub",
            (),
            {"prompt": "a city at night", "language": "en", "negative_prompt": None, "parameters": {}, "metadata": {}},
        )()
        tmpdir = Path(tempfile.mkdtemp())
        runtime_config = tmpdir / "local_runtime.yaml"
        runtime_config.write_text("generation:\n  default_seed: 123\n", encoding="utf-8")

        with patch.dict(os.environ, {"AIGC_LOCAL_RUNTIME_FILE": str(runtime_config)}, clear=False):
            params = _default_generation_params(model, [prompt])  # type: ignore[list-item]

        self.assertEqual(params["seed"], 123)

    def test_model_generation_defaults_come_from_registry_config(self) -> None:
        registry = load_registry()
        model = registry.get_model("LongCat-Video")
        prompt = type(
            "PromptStub",
            (),
            {"prompt": "a city at night", "language": "en", "negative_prompt": None, "parameters": {}, "metadata": {}},
        )()

        params = _default_generation_params(model, [prompt])  # type: ignore[list-item]

        self.assertEqual(params["width"], 1280)
        self.assertEqual(params["height"], 720)
        self.assertEqual(params["fps"], 30)
        self.assertEqual(params["num_frames"], 121)
        self.assertEqual(params["num_inference_steps"], 50)
        self.assertEqual(params["guidance_scale"], 4.0)
        self.assertIn("checkpoint_dir", params)

    def test_generation_params_apply_profile_defaults_then_prompt_overrides(self) -> None:
        registry = load_registry()
        model = registry.get_model("Z-Image")
        prompt = type(
            "PromptStub",
            (),
            {
                "prompt": "a city at night",
                "language": "en",
                "negative_prompt": "blurry",
                "parameters": {"width": 1280, "guidance_scale": 5.5},
                "metadata": {},
            },
        )()

        params = _default_generation_params(
            model,
            [prompt],  # type: ignore[list-item]
            generation_defaults={
                "width": 1024,
                "height": 1024,
                "guidance_scale": 4.0,
                "num_inference_steps": 40,
            },
        )

        self.assertEqual(params["width"], 1280)
        self.assertEqual(params["height"], 1024)
        self.assertEqual(params["guidance_scale"], 5.5)
        self.assertEqual(params["num_inference_steps"], 40)
        self.assertEqual(params["negative_prompts"], ["blurry"])

    def test_batching_splits_when_effective_generation_params_differ(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "image_params.jsonl"
        prompts_path.write_text(
            "\n".join(
                [
                    '{"prompt_id":"p001","prompt":"a city skyline","language":"en"}',
                    '{"prompt_id":"p002","prompt":"a forest cabin","language":"en","parameters":{"width":1280}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = run_single_model(
            model_name="Z-Image",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "image_params_batching",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=self._inprocess_worker_runner,
            profile_generation_defaults={"width": 1024, "height": 1024},
        )

        task_dir = Path(summary.output_dir) / "tasks" / "z-image"
        task_files = sorted(
            path for path in task_dir.iterdir() if path.suffix == ".json" and ".result." not in path.name
        )
        self.assertEqual(len(task_files), 2)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in task_files]
        self.assertEqual([payload["params"]["width"] for payload in payloads], [1024, 1280])

    def test_unknown_prompt_parameter_warning_flows_through_run_progress(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "image_warn.jsonl"
        prompts_path.write_text(
            '{"prompt_id":"p001","prompt":"a city skyline","language":"en","parameters":{"style":"cinematic"}}\n',
            encoding="utf-8",
        )
        progress_stream = StringIO()

        run_single_model(
            model_name="Z-Image",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "image_warn",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=self._inprocess_worker_runner,
            progress=TextRunProgress(stream=progress_stream),
        )

        progress_text = progress_stream.getvalue()
        self.assertIn("Unknown generation parameter key", progress_text)
        self.assertIn("prompt_id=p001", progress_text)

    def test_multi_replica_mock_run_shards_image_tasks_and_exports_replica_metadata(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "replica_image.txt"
        prompts_path.write_text(
            "\n".join([f"image test prompt {index:03d}" for index in range(1, 10)]) + "\n",
            encoding="utf-8",
        )
        progress_stream = StringIO()

        class FakeSession:
            def __init__(
                self,
                *,
                model,
                env_record,
                execution_mode,
                replica_id=0,
                gpu_assignment=None,
                replica_log_path=None,
                log_callback=None,
            ) -> None:
                self.replica_id = replica_id
                self.gpu_assignment = list(gpu_assignment or [])

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def run_task(self, prepared_task):
                workdir = Path(prepared_task.payload.workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                prompt_id = prepared_task.payload.prompts[0].prompt_id
                artifact_path = workdir / f"{prompt_id}.png"
                artifact_path.write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                    + b"\x00\x00\x00\rIHDR"
                    + b"\x00\x00\x00\x01"
                    + b"\x00\x00\x00\x01"
                    + b"\x08\x02\x00\x00\x00"
                )
                prepared_task.result_file.write_text(
                    json.dumps(
                        {
                            "task_id": prepared_task.payload.task_id,
                            "model_name": prepared_task.payload.model_name,
                            "execution_mode": prepared_task.payload.execution_mode,
                            "plan": {"mode": "in_process"},
                            "execution_result": {"exit_code": 0, "logs": "", "outputs": {}},
                            "model_result": {
                                "status": "success",
                                "batch_items": [
                                    {
                                        "prompt_id": prompt_id,
                                        "status": "success",
                                        "metadata": {},
                                        "artifacts": [
                                            {
                                                "type": "image",
                                                "path": str(artifact_path),
                                                "metadata": {"width": 1, "height": 1, "format": "png"},
                                            }
                                        ],
                                    }
                                ],
                                "logs": "",
                                "metadata": {},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return 0, ""

        with patch.dict(os.environ, {"AIGC_AVAILABLE_GPUS": "0,1"}, clear=False):
            with patch("aigc.run_flow._PersistentWorkerSession", FakeSession):
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
        self.assertEqual(sum(replica["task_count"] for replica in per_model["replicas"]), 9)
        self.assertEqual(per_model["replica_count_requested"], 2)
        self.assertGreaterEqual(per_model["replica_count_started"], 1)
        self.assertGreaterEqual(per_model["replica_count_active_final"], 1)

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
        self.assertIn("[SCHED] model=Z-Image available_gpus=[0, 1]", progress_text)
        self.assertIn("[SCHED] model=Z-Image starting 2 replicas", progress_text)
        self.assertIn("[SCHED] model=Z-Image replica=0 assigned 5 tasks GPUs=[0]", progress_text)
        self.assertIn("[SCHED] model=Z-Image replica=1 assigned 4 tasks GPUs=[1]", progress_text)
        self.assertIn("[REPLICA] model=Z-Image", progress_text)

    def test_multi_replica_mock_run_shards_video_tasks_and_uses_gpu_groups(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "replica_video.txt"
        prompts_path.write_text(
            "\n".join([f"video test prompt {index:03d}" for index in range(1, 5)]) + "\n",
            encoding="utf-8",
        )
        progress_stream = StringIO()

        class FakeSession:
            def __init__(
                self,
                *,
                model,
                env_record,
                execution_mode,
                replica_id=0,
                gpu_assignment=None,
                replica_log_path=None,
                log_callback=None,
            ) -> None:
                self.replica_id = replica_id
                self.gpu_assignment = list(gpu_assignment or [])

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def run_task(self, prepared_task):
                workdir = Path(prepared_task.payload.workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                prompt_id = prepared_task.payload.prompts[0].prompt_id
                artifact_path = workdir / f"{prompt_id}.mp4"
                artifact_path.write_bytes(b"FAKE-MP4")
                prepared_task.result_file.write_text(
                    json.dumps(
                        {
                            "task_id": prepared_task.payload.task_id,
                            "model_name": prepared_task.payload.model_name,
                            "execution_mode": prepared_task.payload.execution_mode,
                            "plan": {"mode": "in_process"},
                            "execution_result": {"exit_code": 0, "logs": "", "outputs": {}},
                            "model_result": {
                                "status": "success",
                                "batch_items": [
                                    {
                                        "prompt_id": prompt_id,
                                        "status": "success",
                                        "metadata": {},
                                        "artifacts": [
                                            {
                                                "type": "video",
                                                "path": str(artifact_path),
                                                "metadata": {"fps": 16, "num_frames": 81, "format": "mp4"},
                                            }
                                        ],
                                    }
                                ],
                                "logs": "",
                                "metadata": {},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return 0, ""

        with patch.dict(os.environ, {"AIGC_AVAILABLE_GPUS": "0,1,2,3"}, clear=False):
            with patch("aigc.run_flow._PersistentWorkerSession", FakeSession):
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
        self.assertEqual(per_model["replica_count"], 2)
        self.assertEqual(
            [replica["gpu_assignment"] for replica in per_model["replicas"]],
            [[0], [1]],
        )
        self.assertEqual(sum(replica["task_count"] for replica in per_model["replicas"]), 2)
        self.assertEqual(per_model["replica_count_requested"], 2)

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
        self.assertIn("[SCHED] model=CogVideoX-5B available_gpus=[0, 1, 2, 3]", progress_text)
        self.assertIn("[SCHED] model=CogVideoX-5B gpus_per_replica=1", progress_text)
        self.assertIn("[SCHED] model=CogVideoX-5B starting 2 replicas", progress_text)

    def test_primary_ready_starts_early_dispatch_before_secondary_ready(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        progress_stream = StringIO()
        progress = TextRunProgress(stream=progress_stream)
        registry = load_registry()
        model = registry.get_model("CogVideoX-5B")
        events: list[str] = []

        class FakeSession:
            def __init__(
                self,
                *,
                model,
                env_record,
                execution_mode,
                replica_id=0,
                gpu_assignment=None,
                replica_log_path=None,
                log_callback=None,
            ) -> None:
                self.replica_id = replica_id

            def __enter__(self):
                events.append(f"enter:{self.replica_id}")
                if self.replica_id == 1:
                    events.append("loading:1")
                    time.sleep(0.1)
                    events.append("ready:1")
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                events.append(f"exit:{self.replica_id}")

            def run_task(self, prepared_task):
                events.append(f"run:{self.replica_id}:{prepared_task.payload.task_id}")
                return 0, ""

        def fake_execute_prepared_task(
            *,
            prepared_task,
            runner,
            run_id,
            run_root,
            failures,
            progress,
            ledger_writer,
            telemetry=None,
            state_lock=None,
        ):
            events.append(f"dispatch:{prepared_task.payload.task_id}")
            runner(prepared_task)
            return TaskExecutionOutcome(
                result_payload={
                    "task_id": prepared_task.payload.task_id,
                    "model_result": {"status": "success", "batch_items": []},
                    "execution_result": {"exit_code": 0, "logs": "", "outputs": {}},
                },
                task_failed=False,
                failed_prompt_count=0,
                successful_prompt_count=1,
                duration_sec=0.5,
            )

        def make_prepared(task_id: str, prompt_id: str) -> PreparedTask:
            payload = TaskPayload(
                task_id=task_id,
                model_name=model.name,
                execution_mode="mock",
                prompts=[TaskPrompt(prompt_id=prompt_id, prompt=f"prompt {prompt_id}", language="en")],
                params={},
                workdir=str(tmpdir / task_id),
                worker_strategy="persistent_worker",
            )
            return PreparedTask(
                model=model,
                payload=payload,
                task_file=tmpdir / f"{task_id}.json",
                result_file=tmpdir / f"{task_id}.result.json",
                batch_number=1,
                total_tasks_for_model=2,
            )

        replica_plans = [
            ReplicaPlan(replica_id=0, gpu_assignment=[0], tasks=[make_prepared("task_001", "p001")]),
            ReplicaPlan(replica_id=1, gpu_assignment=[1], tasks=[make_prepared("task_002", "p002")]),
        ]
        ledger_writer = RunLedgerWriter(tmpdir / "samples.jsonl")
        try:
            with patch("aigc.run_flow._PersistentWorkerSession", FakeSession), patch(
                "aigc.run_flow._execute_prepared_task",
                fake_execute_prepared_task,
            ):
                _run_persistent_worker_replicas(
                    model=model,
                    env_record=FakeEnvRecord(),
                    replica_plans=replica_plans,
                    execution_mode="mock",
                    run_id="run_test",
                    run_root=tmpdir,
                    workers_root=tmpdir / "workers",
                    failures=[],
                    progress=progress,
                    ledger_writer=ledger_writer,
                    telemetry=None,
                    failure_policy=FailurePolicy(),
                    failure_budget=FailureBudget(total_planned_outputs=2),
                )
        finally:
            ledger_writer.close()

        self.assertLess(events.index("enter:0"), events.index("enter:1"))
        self.assertLess(events.index("loading:1"), events.index("dispatch:task_001"))
        self.assertLess(events.index("dispatch:task_001"), events.index("ready:1"))
        progress_text = progress_stream.getvalue()
        self.assertIn("[SCHED] model=CogVideoX-5B bootstrapping primary replica=0 GPUs=[0]", progress_text)
        self.assertIn(
            "[SCHED] model=CogVideoX-5B primary replica ready, starting early dispatch replica=0 GPUs=[0]",
            progress_text,
        )
        self.assertIn("[SCHED] model=CogVideoX-5B warming secondary replicas count=1", progress_text)
        self.assertIn(
            "[SCHED] model=CogVideoX-5B secondary replica=1 ready, joined active pool GPUs=[1] replicas_active=1/2",
            progress_text,
        )

    def test_secondary_startup_failure_retries_once_then_degrades(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        progress_stream = StringIO()
        progress = TextRunProgress(stream=progress_stream)
        registry = load_registry()
        model = registry.get_model("CogVideoX-5B")
        attempts: dict[int, int] = {}
        failures: list[dict[str, object]] = []
        telemetry_lines: list[str] = []
        telemetry = RunTelemetry(
            run_id="run_test",
            execution_mode="mock",
            emit_callback=telemetry_lines.append,
            status_path=tmpdir / "runtime_status.json",
            emit_prompt_interval=1,
            emit_sec_interval=1.0,
        )

        class FakeSession:
            def __init__(
                self,
                *,
                model,
                env_record,
                execution_mode,
                replica_id=0,
                gpu_assignment=None,
                replica_log_path=None,
                log_callback=None,
            ) -> None:
                self.replica_id = replica_id

            def __enter__(self):
                attempts[self.replica_id] = attempts.get(self.replica_id, 0) + 1
                if self.replica_id == 1:
                    raise RunFlowError("simulated secondary startup failure")
                telemetry.record_runtime_event(
                    f"[worker][{model.name}][replica={self.replica_id}] GPUs=[{self.replica_id}] ready"
                )
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def run_task(self, prepared_task):
                return 0, ""

        def fake_execute_prepared_task(
            *,
            prepared_task,
            runner,
            run_id,
            run_root,
            failures,
            progress,
            ledger_writer,
            telemetry=None,
            state_lock=None,
        ):
            runner(prepared_task)
            return TaskExecutionOutcome(
                result_payload={
                    "task_id": prepared_task.payload.task_id,
                    "model_result": {"status": "success", "batch_items": []},
                    "execution_result": {"exit_code": 0, "logs": "", "outputs": {}},
                },
                task_failed=False,
                failed_prompt_count=0,
                successful_prompt_count=1,
                duration_sec=0.5,
            )

        def make_prepared(task_id: str, prompt_id: str) -> PreparedTask:
            payload = TaskPayload(
                task_id=task_id,
                model_name=model.name,
                execution_mode="mock",
                prompts=[TaskPrompt(prompt_id=prompt_id, prompt=f"prompt {prompt_id}", language="en")],
                params={},
                workdir=str(tmpdir / task_id),
                worker_strategy="persistent_worker",
            )
            return PreparedTask(
                model=model,
                payload=payload,
                task_file=tmpdir / f"{task_id}.json",
                result_file=tmpdir / f"{task_id}.result.json",
                batch_number=int(task_id.rsplit("_", 1)[1]),
                total_tasks_for_model=3,
            )

        replica_plans = [
            ReplicaPlan(replica_id=0, gpu_assignment=[0], tasks=[make_prepared("task_001", "p001"), make_prepared("task_002", "p002"), make_prepared("task_003", "p003")]),
            ReplicaPlan(replica_id=1, gpu_assignment=[1], tasks=[]),
        ]
        ledger_writer = RunLedgerWriter(tmpdir / "samples.jsonl")
        try:
            telemetry.set_plan(prepared_tasks_by_model={model.name: replica_plans[0].tasks})
            telemetry.register_replica_assignments(model_name=model.name, replica_plans=replica_plans)
            with patch("aigc.run_flow._PersistentWorkerSession", FakeSession), patch(
                "aigc.run_flow._execute_prepared_task",
                fake_execute_prepared_task,
            ):
                results = _run_persistent_worker_replicas(
                    model=model,
                    env_record=FakeEnvRecord(),
                    replica_plans=replica_plans,
                    execution_mode="mock",
                    run_id="run_test",
                    run_root=tmpdir,
                    workers_root=tmpdir / "workers",
                    failures=failures,
                    progress=progress,
                    ledger_writer=ledger_writer,
                    telemetry=telemetry,
                    failure_policy=FailurePolicy(),
                    failure_budget=FailureBudget(total_planned_outputs=3),
                )
        finally:
            ledger_writer.close()

        snapshot = telemetry.finalize(status="completed")
        self.assertEqual(len(results), 3)
        self.assertEqual(attempts.get(1), 2)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0]["non_fatal"])
        self.assertTrue(failures[0]["unavailable"])
        self.assertEqual(failures[0]["category"], "worker_startup_error")
        self.assertEqual(snapshot["models"][model.name]["replica_startup_failures"], 2)
        self.assertEqual(snapshot["models"][model.name]["active_replicas"], 1)
        self.assertTrue(snapshot["replicas"][model.name]["r1"]["unavailable"])
        progress_text = progress_stream.getvalue()
        self.assertIn("[WARN] model=CogVideoX-5B secondary replica=1 startup failed, retrying GPUs=[1]", progress_text)
        self.assertIn(
            "[WARN] model=CogVideoX-5B secondary replica=1 unavailable after retry, continuing with 1/2 active replicas GPUs=[1]",
            progress_text,
        )

    def test_runtime_telemetry_is_logged_and_persisted_during_run(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "telemetry_prompts.txt"
        prompts_path.write_text(
            "\n".join([f"telemetry prompt {index:03d}" for index in range(1, 11)]) + "\n",
            encoding="utf-8",
        )

        summary = run_single_model(
            model_name="Z-Image",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "telemetry_mock",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=self._inprocess_worker_runner,
        )

        run_root = Path(summary.output_dir)
        running_log = (run_root / "running.log").read_text(encoding="utf-8")
        self.assertIn("[THROUGHPUT] overall prompts=10/10", running_log)
        self.assertIn("[THROUGHPUT] model=Z-Image prompts=10/10", running_log)

        runtime_status = json.loads((run_root / "runtime_status.json").read_text(encoding="utf-8"))
        self.assertEqual(runtime_status["processed_prompts"], 10)
        self.assertEqual(runtime_status["status"], "completed")
        self.assertIn("Z-Image", runtime_status["models"])

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runtime_status_path"], str(run_root / "runtime_status.json"))
        self.assertEqual(manifest["runtime_metrics"]["processed_prompts"], 10)
        self.assertIn("runtime_status_json", manifest["export_paths"])

    def test_progress_events_are_logged_and_persisted(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        progress_stream = StringIO()
        logger = RunLogger(log_path=tmpdir / "running.log")
        telemetry = RunTelemetry(
            run_id="run_progress",
            execution_mode="real",
            emit_callback=lambda _line: None,
            status_path=tmpdir / "runtime_status.json",
            emit_prompt_interval=1,
            emit_sec_interval=999.0,
        )
        prepared_task = type("Prepared", (), {"payload": type("Payload", (), {"prompts": [object()]})()})
        telemetry.set_plan(prepared_tasks_by_model={"Z-Image": [prepared_task]})
        try:
            _handle_runtime_event(
                logger=logger,
                terminal_progress=TextRunProgress(stream=progress_stream),
                telemetry=telemetry,
                message="[progress] model=Z-Image replica=0 task=task_001 batch=1 phase=generating step=7/40 true_progress=yes",
            )
        finally:
            logger.close()

        running_log = (tmpdir / "running.log").read_text(encoding="utf-8")
        self.assertIn("[progress] model=Z-Image replica=0 task=task_001 batch=1 phase=generating step=7/40 true_progress=yes", running_log)
        runtime_status = json.loads((tmpdir / "runtime_status.json").read_text(encoding="utf-8"))
        replica = runtime_status["replicas"]["Z-Image"]["r0"]
        self.assertEqual(replica["current_task_id"], "task_001")
        self.assertEqual(replica["current_step"], 7)
        self.assertEqual(replica["total_steps"], 40)

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
        ledger_path = tmpdir / "runs" / "worker_failure" / "samples.jsonl"
        self.assertTrue(ledger_path.exists())
        ledger_records = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual(len(ledger_records), 1)
        self.assertEqual(ledger_records[0]["status"], "failed")
        self.assertIn("real root cause", ledger_records[0]["error_message"])

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

    def test_run_can_continue_after_task_failures_when_policy_allows(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "continue.txt"
        prompts_path.write_text("prompt one\nprompt two\n", encoding="utf-8")
        call_count = 0

        def mixed_worker_runner(_env_record, task_file: Path, result_file: Path):
            nonlocal call_count
            call_count += 1
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            prompt = task_payload["prompts"][0]
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            if call_count == 1:
                result_payload = {
                    "task_id": task_payload["task_id"],
                    "model_name": task_payload["model_name"],
                    "execution_mode": task_payload["execution_mode"],
                    "plan": {"mode": "in_process"},
                    "execution_result": {"exit_code": 1, "logs": "RuntimeError: boom", "outputs": {}},
                    "model_result": {
                        "status": "failed",
                        "batch_items": [],
                        "logs": "RuntimeError: boom",
                        "metadata": {"error_type": "RuntimeError"},
                    },
                }
                result_file.write_text(json.dumps(result_payload), encoding="utf-8")
                return 1, "RuntimeError: boom"

            artifact_path = workdir / f"{prompt['prompt_id']}.png"
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
                            "prompt_id": prompt["prompt_id"],
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
            model_name="Z-Image-Turbo",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "continue_on_error",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=mixed_worker_runner,
            batch_limit=1,
            continue_on_error=True,
        )

        self.assertEqual(summary.status, "completed_with_failures")
        failures = json.loads((Path(summary.output_dir) / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["category"], "task_execution_error")
        ledger_records = [
            json.loads(line)
            for line in (Path(summary.output_dir) / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual([record["status"] for record in ledger_records], ["failed", "success"])
        self.assertEqual(ledger_records[0]["failure_category"], "task_execution_error")
        exported_records = [
            json.loads(line)
            for line in Path(summary.export_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(exported_records), 1)

    def test_run_stops_when_max_failures_threshold_is_exceeded(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "threshold.txt"
        prompts_path.write_text("p1\np2\np3\n", encoding="utf-8")
        call_count = 0

        def always_fail_worker(_env_record, task_file: Path, result_file: Path):
            nonlocal call_count
            call_count += 1
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
                "plan": {"mode": "in_process"},
                "execution_result": {"exit_code": 1, "logs": "RuntimeError: threshold", "outputs": {}},
                "model_result": {
                    "status": "failed",
                    "batch_items": [],
                    "logs": "RuntimeError: threshold",
                    "metadata": {"error_type": "RuntimeError"},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 1, "RuntimeError: threshold"

        with self.assertRaises(RunFlowError) as context:
            run_single_model(
                model_name="Z-Image-Turbo",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "max_failures",
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                worker_runner=always_fail_worker,
                batch_limit=1,
                continue_on_error=True,
                max_failures=1,
            )

        self.assertIn("Failure policy threshold exceeded", str(context.exception))
        self.assertEqual(call_count, 2)
        manifest = json.loads((tmpdir / "runs" / "max_failures" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("Failure policy threshold exceeded", manifest["stop_reason"])
        failures = json.loads((tmpdir / "runs" / "max_failures" / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(len(failures), 2)

    def test_run_stops_when_failure_rate_threshold_is_exceeded(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "failure_rate.txt"
        prompts_path.write_text("p1\np2\np3\np4\n", encoding="utf-8")
        call_count = 0

        def rate_worker(_env_record, task_file: Path, result_file: Path):
            nonlocal call_count
            call_count += 1
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            prompt = task_payload["prompts"][0]
            workdir = Path(task_payload["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            if call_count <= 2:
                result_payload = {
                    "task_id": task_payload["task_id"],
                    "model_name": task_payload["model_name"],
                    "execution_mode": task_payload["execution_mode"],
                    "plan": {"mode": "in_process"},
                    "execution_result": {"exit_code": 1, "logs": "RuntimeError: rate", "outputs": {}},
                    "model_result": {
                        "status": "failed",
                        "batch_items": [],
                        "logs": "RuntimeError: rate",
                        "metadata": {"error_type": "RuntimeError"},
                    },
                }
                result_file.write_text(json.dumps(result_payload), encoding="utf-8")
                return 1, "RuntimeError: rate"

            artifact_path = workdir / f"{prompt['prompt_id']}.png"
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
                            "prompt_id": prompt["prompt_id"],
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

        with self.assertRaises(RunFlowError) as context:
            run_single_model(
                model_name="Z-Image-Turbo",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "failure_rate",
                execution_mode="mock",
                env_manager=FakeEnvManager(),
                worker_runner=rate_worker,
                batch_limit=1,
                continue_on_error=True,
                max_failure_rate=0.25,
            )

        self.assertIn("Failure policy threshold exceeded", str(context.exception))
        self.assertEqual(call_count, 2)

    def test_failure_category_is_recorded_for_model_load_error(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "model_load.txt"
        prompts_path.write_text("prompt one\n", encoding="utf-8")

        def failing_worker(_env_record, task_file: Path, result_file: Path):
            task_payload = json.loads(task_file.read_text(encoding="utf-8"))
            logs = "Persistent worker failed during startup.\nfrom_pretrained exploded."
            result_payload = {
                "task_id": task_payload["task_id"],
                "model_name": task_payload["model_name"],
                "execution_mode": task_payload["execution_mode"],
                "plan": None,
                "execution_result": {"exit_code": 1, "logs": logs, "outputs": {}},
                "model_result": {
                    "status": "failed",
                    "batch_items": [],
                    "logs": logs,
                    "metadata": {"error_type": "RuntimeError"},
                },
            }
            result_file.write_text(json.dumps(result_payload), encoding="utf-8")
            return 1, logs

        summary = run_single_model(
            model_name="Wan2.2-T2V-A14B-Diffusers",
            prompt_file=prompts_path,
            out_dir=tmpdir / "runs" / "model_load_category",
            execution_mode="mock",
            env_manager=FakeEnvManager(),
            worker_runner=failing_worker,
            continue_on_error=True,
        )

        self.assertEqual(summary.status, "completed_with_failures")
        failures = json.loads((Path(summary.output_dir) / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(failures[0]["category"], "model_load_error")
        ledger_record = json.loads((Path(summary.output_dir) / "samples.jsonl").read_text(encoding="utf-8").strip())
        self.assertEqual(ledger_record["failure_category"], "model_load_error")

    def test_real_run_fails_clearly_when_required_conda_env_is_missing(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        prompts_path = tmpdir / "example.txt"
        prompts_path.write_text("a futuristic city at night\n", encoding="utf-8")

        class MissingEnvManager(FakeEnvManager):
            def ensure_ready(self, model_name: str, foreground: bool = True, progress=None):
                del foreground, progress
                raise EnvManagerError(
                    f"Environment for {model_name} is not available.\n"
                    "Required conda env: zimage\n"
                    "Please create this environment manually before running."
                )

        with self.assertRaises(EnvManagerError) as context:
            run_single_model(
                model_name="Z-Image-Turbo",
                prompt_file=prompts_path,
                out_dir=tmpdir / "runs" / "missing_env",
                execution_mode="real",
                env_manager=MissingEnvManager(),
            )

        self.assertIn("Required conda env: zimage", str(context.exception))
        self.assertIn("Please create this environment manually before running.", str(context.exception))

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
        self.assertRegex(running_log_text, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[STAGE 1/9\] Loading prompts\.\.\.")
        self.assertIn("[summary] Run complete", running_log_text)
        self.assertIn("[summary] running_log:", running_log_text)
        export_path = Path(summary.export_path)
        self.assertTrue(export_path.exists())
        record = json.loads(export_path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["model_name"], "Z-Image")
        self.assertEqual(record["execution_metadata"]["status"], "success")
        ledger_path = Path(summary.output_dir) / "samples.jsonl"
        self.assertTrue(ledger_path.exists())
        ledger_records = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual(len(ledger_records), 1)
        self.assertEqual(ledger_records[0]["prompt"], "a futuristic city at night")
        self.assertEqual(ledger_records[0]["status"], "success")
        self.assertEqual(ledger_records[0]["artifact_type"], "image")

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

        self.assertEqual(summary.tasks_scheduled, 2)
        self.assertEqual(summary.records_exported, 4)
        ledger_records = [
            json.loads(line)
            for line in (Path(summary.output_dir) / "samples.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        self.assertEqual(len(ledger_records), 4)
        self.assertEqual(
            sorted(record["prompt_id"] for record in ledger_records),
            ["prompt_000001", "prompt_000001", "prompt_000002", "prompt_000002"],
        )
        self.assertTrue(all(record["artifact_type"] == "video" for record in ledger_records))
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
