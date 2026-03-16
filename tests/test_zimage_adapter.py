import base64
import tempfile
import unittest
from pathlib import Path

from aigc.adapters.base import ExecutionPlan, ExecutionResult
from aigc.adapters.zimage import ZImageAdapter
from aigc.registry import load_registry


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nX1EAAAAASUVORK5CYII="
)


class ZImageAdapterTests(unittest.TestCase):
    def test_collect_normalizes_png_artifacts(self) -> None:
        registry = load_registry()
        adapter = ZImageAdapter(model_config=registry.get_model("Z-Image"))
        tmpdir = Path(tempfile.mkdtemp())
        image_path = tmpdir / "p001.png"
        image_path.write_bytes(TINY_PNG)

        plan = ExecutionPlan(mode="in_process", inputs={})
        exec_result = ExecutionResult(
            exit_code=0,
            logs="ok",
            outputs={
                "p001": {
                    "path": str(image_path),
                    "seed": 42,
                    "guidance_scale": 4.0,
                    "num_inference_steps": 50,
                }
            },
        )
        result = adapter.collect(
            plan=plan,
            exec_result=exec_result,
            prompts=["a futuristic city"],
            prompt_ids=["p001"],
            workdir=str(tmpdir),
        )
        self.assertEqual(result.status, "success")
        item = result.batch_items[0]
        self.assertEqual(item.prompt_id, "p001")
        self.assertEqual(item.artifacts[0].type, "image")
        self.assertEqual(item.artifacts[0].metadata["width"], 1)
        self.assertEqual(item.artifacts[0].metadata["format"], "png")

    def test_mock_execute_generates_batch_pngs(self) -> None:
        registry = load_registry()
        adapter = ZImageAdapter(model_config=registry.get_model("Z-Image"))
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a futuristic city", "一只可爱的猫"],
            prompt_ids=["p001", "p002"],
            params={
                "width": 32,
                "height": 16,
                "guidance_scale": 4.0,
                "num_inference_steps": 8,
                "seed": 7,
                "negative_prompts": ["", "blurry"],
            },
            workdir=str(tmpdir),
        )
        plan.inputs["batch_id"] = "batch_001"
        plan.inputs["runtime"] = {"execution_mode": "mock"}

        result = adapter.execute(
            plan=plan,
            prompts=["a futuristic city", "一只可爱的猫"],
            params={},
            workdir=str(tmpdir),
        )
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["a futuristic city", "一只可爱的猫"],
            prompt_ids=["p001", "p002"],
            workdir=str(tmpdir),
        )

        self.assertEqual(collected.status, "success")
        self.assertEqual(len(collected.batch_items), 2)
        first_item = collected.batch_items[0]
        second_item = collected.batch_items[1]
        self.assertTrue(Path(first_item.artifacts[0].path).exists())
        self.assertEqual(first_item.metadata["batch_id"], "batch_001")
        self.assertEqual(first_item.metadata["batch_index"], 0)
        self.assertTrue(first_item.metadata["mock"])
        self.assertEqual(collected.metadata["execution_mode"], "mock")
        self.assertEqual(second_item.metadata["batch_index"], 1)
        self.assertEqual(first_item.artifacts[0].metadata["width"], 32)
        self.assertEqual(first_item.artifacts[0].metadata["height"], 16)
