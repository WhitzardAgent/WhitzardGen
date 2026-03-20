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

    def test_real_execute_emits_true_progress_steps_when_pipeline_supports_callback(self) -> None:
        registry = load_registry()
        adapter = ZImageAdapter(model_config=registry.get_model("Z-Image"))
        tmpdir = Path(tempfile.mkdtemp())
        plan = adapter.prepare(
            prompts=["a futuristic city"],
            prompt_ids=["p001"],
            params={"width": 32, "height": 32, "num_inference_steps": 4},
            workdir=str(tmpdir),
        )
        events: list[dict[str, object]] = []

        class _FakeImage:
            def save(self, path: Path) -> None:
                path.write_bytes(TINY_PNG)

        class _FakePipe:
            def __call__(
                self,
                *,
                prompt,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
                **kwargs,
            ):
                del prompt, height, width, guidance_scale, callback_on_step_end_tensor_inputs, kwargs
                for step_index in range(num_inference_steps):
                    if callback_on_step_end is not None:
                        callback_on_step_end(self, step_index, 0, {})
                return type("Output", (), {"images": [_FakeImage()]})()

        class _FakeTorch:
            class Generator:
                def __init__(self, device: str) -> None:
                    self.device = device

                def manual_seed(self, seed: int):
                    return self

        adapter._loaded_pipeline = _FakePipe()
        adapter._loaded_torch = _FakeTorch
        adapter._loaded_device = "cuda"

        result = adapter.execute(
            plan=plan,
            prompts=["a futuristic city"],
            params={},
            workdir=str(tmpdir),
            progress_callback=events.append,
        )

        self.assertIn("p001", result.outputs)
        self.assertEqual([event["current_step"] for event in events], [1, 2, 3, 4])
        self.assertTrue(all(event["supports_true_progress"] for event in events))
