import tempfile
import unittest
from pathlib import Path

from aigc.adapters.video_family import (
    WanT2VDiffusersAdapter,
    extract_video_metadata,
    metadata_sidecar_path,
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
