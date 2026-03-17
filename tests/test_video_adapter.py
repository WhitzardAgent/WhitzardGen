import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aigc.adapters.video_family import (
    CogVideoX5BAdapter,
    WanT2VDiffusersAdapter,
    extract_video_metadata,
    metadata_sidecar_path,
    resolve_video_model_reference,
)
from aigc.registry import load_registry


class VideoAdapterTests(unittest.TestCase):
    def test_mock_video_execution_generates_artifact_and_metadata(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a cinematic drone shot over a coastline"],
            prompt_ids=["p001"],
            params={
                "width": 640,
                "height": 360,
                "fps": 12,
                "num_frames": 25,
                "num_inference_steps": 8,
                "guidance_scale": 4.0,
                "negative_prompts": ["low quality"],
                "seed": 11,
                "_runtime_config": {"execution_mode": "mock"},
            },
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_001"

        result = adapter.execute(
            plan=plan,
            prompts=["a cinematic drone shot over a coastline"],
            params={},
            workdir=str(tmpdir),
        )
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["a cinematic drone shot over a coastline"],
            prompt_ids=["p001"],
            workdir=str(tmpdir),
        )

        self.assertEqual(collected.status, "success")
        item = collected.batch_items[0]
        self.assertEqual(item.prompt_id, "p001")
        self.assertEqual(item.artifacts[0].type, "video")
        self.assertTrue(Path(item.artifacts[0].path).exists())
        self.assertTrue(metadata_sidecar_path(item.artifacts[0].path).exists())
        self.assertEqual(item.metadata["batch_id"], "batch_001")
        self.assertEqual(item.metadata["batch_index"], 0)
        self.assertTrue(item.metadata["mock"])
        self.assertEqual(collected.metadata["execution_mode"], "mock")
        self.assertEqual(item.artifacts[0].metadata["fps"], 12)
        self.assertEqual(item.artifacts[0].metadata["num_frames"], 25)
        self.assertEqual(item.artifacts[0].metadata["format"], "mp4")

    def test_extract_video_metadata_reads_sidecar(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a quiet forest with drifting fog"],
            prompt_ids=["p002"],
            params={"_runtime_config": {"execution_mode": "mock"}},
            workdir=str(tmpdir),
        )
        result = adapter.execute(
            plan=plan,
            prompts=["a quiet forest with drifting fog"],
            params={},
            workdir=str(tmpdir),
        )
        path = result.outputs["p002"]["path"]
        metadata = extract_video_metadata(path)
        self.assertEqual(metadata["format"], "mp4")
        self.assertEqual(metadata["width"], 1280)
        self.assertEqual(metadata["height"], 720)

    def test_wan_real_command_uses_torchrun_when_max_gpus_exceeds_one(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        repo_dir = tmpdir / "Wan2.2"
        ckpt_dir = tmpdir / "Wan2.2-T2V-A14B"
        repo_dir.mkdir()
        ckpt_dir.mkdir()
        (repo_dir / "generate.py").write_text("print('stub')\n", encoding="utf-8")
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "repo_path": str(repo_dir),
                "weights_path": str(ckpt_dir),
            },
            runtime={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").runtime,
                "max_gpus": 8,
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)

        command = adapter.build_real_command(
            prompts=["Two anthropomorphic cats fight on stage."],
            prompt_ids=["p001"],
            params={"max_gpus": 8},
            workdir=str(tmpdir),
            inputs={"width": 1280, "height": 720},
        )

        self.assertEqual(
            command,
            [
                "torchrun",
                "--nproc_per_node=8",
                str(repo_dir / "generate.py"),
                "--task",
                "t2v-A14B",
                "--size",
                "1280*720",
                "--ckpt_dir",
                str(ckpt_dir),
                "--dit_fsdp",
                "--t5_fsdp",
                "--ulysses_size",
                "8",
                "--prompt",
                "Two anthropomorphic cats fight on stage.",
            ],
        )

    def test_wan_real_command_uses_single_gpu_python_fallback(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        repo_dir = tmpdir / "Wan2.2"
        ckpt_dir = tmpdir / "Wan2.2-T2V-A14B"
        repo_dir.mkdir()
        ckpt_dir.mkdir()
        (repo_dir / "generate.py").write_text("print('stub')\n", encoding="utf-8")
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "repo_path": str(repo_dir),
                "weights_path": str(ckpt_dir),
            },
            runtime={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").runtime,
                "max_gpus": 1,
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)

        command = adapter.build_real_command(
            prompts=["Two anthropomorphic cats fight on stage."],
            prompt_ids=["p001"],
            params={"max_gpus": 1},
            workdir=str(tmpdir),
            inputs={"width": 1280, "height": 720},
        )

        self.assertEqual(
            command,
            [
                "python",
                str(repo_dir / "generate.py"),
                "--task",
                "t2v-A14B",
                "--size",
                "1280*720",
                "--ckpt_dir",
                str(ckpt_dir),
                "--offload_model",
                "True",
                "--convert_model_dtype",
                "--prompt",
                "Two anthropomorphic cats fight on stage.",
            ],
        )

    def test_wan_real_command_reports_repo_and_weights_expectations(self) -> None:
        registry = load_registry()
        base_model = registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        model = replace(
            base_model,
            weights={
                **base_model.weights,
                "repo_path": "/deps/Wan2.2",
                "weights_path": tempfile.mkdtemp(),
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)

        with self.assertRaisesRegex(
            RuntimeError,
            "repo_path does not exist",
        ):
            adapter.build_real_command(
                prompts=["Two anthropomorphic cats fight on stage."],
                prompt_ids=["p001"],
                params={},
                workdir=tempfile.mkdtemp(),
                inputs={"width": 1280, "height": 720},
            )

    def test_video_model_reference_prefers_weights_path(self) -> None:
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "local_path": "/models/raw-wan",
                "weights_path": "/models/Wan2.2-T2V-A14B-Diffusers",
            },
        )

        self.assertEqual(
            resolve_video_model_reference(model),
            "/models/Wan2.2-T2V-A14B-Diffusers",
        )

    def test_wan_collect_recovers_generated_mp4_from_workdir(self) -> None:
        registry = load_registry()
        tmpdir = Path(tempfile.mkdtemp())
        repo_dir = tmpdir / "Wan2.2"
        ckpt_dir = tmpdir / "Wan2.2-T2V-A14B"
        repo_dir.mkdir()
        ckpt_dir.mkdir()
        (repo_dir / "generate.py").write_text("print('stub')\n", encoding="utf-8")
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "repo_path": str(repo_dir),
                "weights_path": str(ckpt_dir),
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)
        generated_path = tmpdir / "t2v_out.mp4"
        generated_path.write_bytes(b"mock-video")
        plan = adapter.prepare(
            prompts=["a cinematic duel in heavy rain"],
            prompt_ids=["p003"],
            params={
                "_runtime_config": {"execution_mode": "real"},
                "repo_dir": str(repo_dir),
                "local_model_path": str(ckpt_dir),
            },
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_recover_001"

        collected = adapter.collect(
            plan=plan,
            exec_result=type("ExecResult", (), {"logs": "ok", "outputs": {}})(),
            prompts=["a cinematic duel in heavy rain"],
            prompt_ids=["p003"],
            workdir=str(tmpdir),
        )

        item = collected.batch_items[0]
        self.assertEqual(item.status, "success")
        self.assertTrue(Path(item.artifacts[0].path).exists())
        self.assertEqual(Path(item.artifacts[0].path).name, "p003.mp4")

    def test_cogvideox_mock_video_defaults_match_reference_shape(self) -> None:
        registry = load_registry()
        adapter = CogVideoX5BAdapter(model_config=registry.get_model("CogVideoX-5B"))
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["A panda plays guitar in a bamboo forest."],
            prompt_ids=["c001"],
            params={"_runtime_config": {"execution_mode": "mock"}},
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_cog_001"

        result = adapter.execute(
            plan=plan,
            prompts=["A panda plays guitar in a bamboo forest."],
            params={},
            workdir=str(tmpdir),
        )
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["A panda plays guitar in a bamboo forest."],
            prompt_ids=["c001"],
            workdir=str(tmpdir),
        )

        self.assertEqual(collected.status, "success")
        item = collected.batch_items[0]
        self.assertEqual(item.artifacts[0].metadata["width"], 720)
        self.assertEqual(item.artifacts[0].metadata["height"], 480)
        self.assertEqual(item.artifacts[0].metadata["fps"], 8)
        self.assertEqual(item.artifacts[0].metadata["num_frames"], 49)
        self.assertEqual(item.artifacts[0].metadata["guidance_scale"], 6.0)
