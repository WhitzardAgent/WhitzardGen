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

    def test_wan_capabilities_enable_persistent_worker(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )

        self.assertTrue(adapter.capabilities.supports_persistent_worker)
        self.assertEqual(adapter.capabilities.preferred_worker_strategy, "persistent_worker")
        self.assertEqual(adapter.real_execution_mode, "in_process")

    def test_wan_validate_model_reference_requires_diffusers_layout(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        weights_dir = tmpdir / "Wan2.2-T2V-A14B-Diffusers"
        weights_dir.mkdir()
        (weights_dir / "model_index.json").write_text("{}", encoding="utf-8")
        registry = load_registry()
        model = replace(
            registry.get_model("Wan2.2-T2V-A14B-Diffusers"),
            weights={
                **registry.get_model("Wan2.2-T2V-A14B-Diffusers").weights,
                "weights_path": str(weights_dir),
            },
        )
        adapter = WanT2VDiffusersAdapter(model_config=model)

        with self.assertRaisesRegex(
            RuntimeError,
            "Missing required files: vae/config.json",
        ):
            adapter.validate_model_reference(str(weights_dir))

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

    def test_wan_generate_frames_passes_guidance_scale_2(self) -> None:
        registry = load_registry()
        adapter = WanT2VDiffusersAdapter(
            model_config=registry.get_model("Wan2.2-T2V-A14B-Diffusers")
        )
        plan = adapter.prepare(
            prompts=["a cinematic duel in heavy rain"],
            prompt_ids=["p003"],
            params={
                "_runtime_config": {"execution_mode": "real"},
                "guidance_scale_2": 3.5,
            },
            workdir=tempfile.mkdtemp(),
        )

        class _FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class _FakeTorch:
            class Generator(_FakeGenerator):
                pass

        class _FakePipe:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return type("Output", (), {"frames": [[b"frame-1", b"frame-2"]]})()

        pipe = _FakePipe()
        frames = adapter.generate_frames(
            pipe=pipe,
            plan=plan,
            prompt="a cinematic duel in heavy rain",
            negative_prompt="low quality",
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=40,
            guidance_scale=4.0,
            seed=42,
            torch=_FakeTorch,
            device="cuda",
        )

        self.assertEqual(frames, [b"frame-1", b"frame-2"])
        self.assertEqual(pipe.calls[0]["guidance_scale_2"], 3.5)
        self.assertEqual(pipe.calls[0]["negative_prompt"], "low quality")
        self.assertEqual(pipe.calls[0]["num_frames"], 81)

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
