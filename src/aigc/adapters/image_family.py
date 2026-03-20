from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any
import inspect

from aigc.adapters.base import (
    AdapterCapabilities,
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
    ProgressCallback,
)


class MockCapableImageAdapter(BaseAdapter):
    default_width = 1024
    default_height = 1024
    default_guidance_scale = 4.0
    default_num_inference_steps = 50
    guidance_argument_name = "guidance_scale"
    include_cfg_normalization = False

    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionPlan:
        width, height = resolve_image_dimensions(params)
        negative_prompts = resolve_negative_prompts(
            prompts=prompts,
            params=params,
            supports_negative_prompt=self.capabilities.supports_negative_prompt,
        )
        inputs = {
            "prompt_ids": list(prompt_ids),
            "width": width,
            "height": height,
            "negative_prompts": negative_prompts,
            "num_inference_steps": int(
                params.get("num_inference_steps", self.default_num_inference_steps)
            ),
            "guidance_scale": float(params.get("guidance_scale", self.default_guidance_scale)),
        }
        if params.get("seed") not in (None, ""):
            inputs["seed"] = int(params["seed"])
        if self.include_cfg_normalization:
            inputs["cfg_normalization"] = bool(params.get("cfg_normalization", False))
        inputs.update(self.extra_prepare_inputs(params))
        return ExecutionPlan(mode="in_process", inputs=inputs)

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, Any],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        runtime = dict(plan.inputs.get("runtime", {}))
        if runtime.get("execution_mode") == "mock":
            outputs = self._execute_mock(plan=plan, prompts=prompts, workdir=workdir)
            return ExecutionResult(
                exit_code=0,
                logs=f"{self.model_config.name} mock image generation completed.",
                outputs=outputs,
            )

        outputs = self._execute_real(
            plan=plan,
            prompts=prompts,
            workdir=workdir,
            progress_callback=progress_callback,
        )
        return ExecutionResult(
            exit_code=0,
            logs=f"{self.model_config.name} image generation completed.",
            outputs=outputs,
        )

    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        batch_id = plan.inputs.get("batch_id")
        runtime = dict(plan.inputs.get("runtime", {}))
        items: list[BatchItemResult] = []
        success_count = 0

        for fallback_index, prompt_id in enumerate(prompt_ids):
            output = dict(exec_result.outputs.get(prompt_id, {}))
            output_path = Path(str(output.get("path", "")))
            if not output_path.exists():
                items.append(
                    BatchItemResult(
                        prompt_id=prompt_id,
                        artifacts=[],
                        status="failed",
                        metadata={
                            "batch_id": output.get("batch_id", batch_id),
                            "batch_index": output.get("batch_index", fallback_index),
                        },
                        error_message=f"Expected artifact missing for {prompt_id}",
                    )
                )
                continue

            artifact_metadata = extract_png_metadata(output_path)
            artifact_metadata.update(
                {
                    "seed": output.get("seed"),
                    "guidance_scale": output.get("guidance_scale"),
                    "num_inference_steps": output.get("num_inference_steps"),
                    "mock": bool(output.get("mock", False)),
                }
            )
            item_metadata = {
                "seed": output.get("seed"),
                "guidance_scale": output.get("guidance_scale"),
                "num_inference_steps": output.get("num_inference_steps"),
                "batch_id": output.get("batch_id", batch_id),
                "batch_index": output.get("batch_index", fallback_index),
                "mock": bool(output.get("mock", False)),
                "replica_id": runtime.get("replica_id"),
                "gpu_assignment": list(runtime.get("gpu_assignment", [])),
            }
            items.append(
                BatchItemResult(
                    prompt_id=prompt_id,
                    artifacts=[
                        ArtifactRecord(
                            type="image",
                            path=str(output_path),
                            metadata=artifact_metadata,
                        )
                    ],
                    status="success",
                    metadata=item_metadata,
                )
            )
            success_count += 1

        if success_count == len(prompt_ids):
            status = "success"
        elif success_count == 0:
            status = "failed"
        else:
            status = "partial_success"

        return ModelResult(
            status=status,
            batch_items=items,
            logs=exec_result.logs,
            metadata={
                "batch_id": batch_id,
                "execution_mode": str(runtime.get("execution_mode", "real")),
                "replica_id": runtime.get("replica_id"),
                "gpu_assignment": list(runtime.get("gpu_assignment", [])),
            },
        )

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _execute_mock(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
    ) -> dict[str, dict[str, Any]]:
        prompt_ids = list(plan.inputs["prompt_ids"])
        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        guidance_scale = float(plan.inputs["guidance_scale"])
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        batch_id = plan.inputs.get("batch_id")

        outputs: dict[str, dict[str, Any]] = {}
        for batch_index, (prompt_id, prompt) in enumerate(zip(prompt_ids, prompts, strict=True)):
            output_path = Path(workdir) / f"{prompt_id}.png"
            effective_seed = seed + batch_index if seed is not None else None
            color = deterministic_color(
                self.model_config.name,
                prompt_id,
                prompt,
                str(effective_seed) if effective_seed is not None else "mock-random",
            )
            write_mock_png(output_path, width=width, height=height, color=color)
            outputs[prompt_id] = {
                "path": str(output_path),
                "seed": effective_seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "mock": True,
            }
        return outputs

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement real image execution yet."
        )


class DiffusersImageAdapterBase(MockCapableImageAdapter):
    pipeline_class_name = "DiffusionPipeline"
    low_cpu_mem_usage = True
    use_safetensors: bool | None = None

    def __init__(self, model_config: Any) -> None:
        super().__init__(model_config)
        self._loaded_pipeline: Any | None = None
        self._loaded_torch: Any | None = None
        self._loaded_device: str | None = None

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_pipeline()

    def unload_persistent_worker(self) -> None:
        self._loaded_pipeline = None
        self._loaded_torch = None
        self._loaded_device = None

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        prompt_ids = list(plan.inputs["prompt_ids"])
        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        guidance_scale = float(plan.inputs["guidance_scale"])
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        batch_id = plan.inputs.get("batch_id")
        negative_prompts = list(plan.inputs.get("negative_prompts", []))
        pipe, torch, device = self._get_or_load_pipeline()

        generator_device = "cuda" if device == "cuda" else "cpu"
        generators = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompt_ids))
            ]
        call_kwargs = {
            "prompt": prompts,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            self.guidance_argument_name: guidance_scale,
        }
        call_kwargs.update(
            _build_diffusers_progress_kwargs(
                pipe=pipe,
                total_steps=num_inference_steps,
                progress_callback=progress_callback,
            )
        )
        if generators is not None:
            call_kwargs["generator"] = generators
        if self.capabilities.supports_negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompts
        if self.include_cfg_normalization:
            call_kwargs["cfg_normalization"] = bool(plan.inputs.get("cfg_normalization", False))
        call_kwargs.update(self.build_pipeline_call_kwargs(plan))

        images = list(pipe(**call_kwargs).images)
        if len(images) != len(prompt_ids):
            raise RuntimeError(
                f"{self.model_config.name} returned {len(images)} images for {len(prompt_ids)} prompts."
            )

        outputs: dict[str, dict[str, Any]] = {}
        for batch_index, (prompt_id, image) in enumerate(zip(prompt_ids, images, strict=True)):
            output_path = Path(workdir) / f"{prompt_id}.png"
            image.save(output_path)
            outputs[prompt_id] = {
                "path": str(output_path),
                "seed": seed + batch_index if seed is not None else None,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "mock": False,
            }
        return outputs

    def _get_or_load_pipeline(self) -> tuple[Any, Any, str]:
        if self._loaded_pipeline is not None and self._loaded_torch is not None and self._loaded_device:
            return self._loaded_pipeline, self._loaded_torch, self._loaded_device

        import torch
        import diffusers

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self.resolve_torch_dtype(torch=torch, device=device)
        pipeline_class = getattr(diffusers, self.pipeline_class_name)
        load_kwargs = self.build_pipeline_load_kwargs(device=device, torch_dtype=dtype)
        pipe = pipeline_class.from_pretrained(resolve_model_reference(self.model_config), **load_kwargs)
        pipe.to(device)
        self._loaded_pipeline = pipe
        self._loaded_torch = torch
        self._loaded_device = device
        return pipe, torch, device

    def resolve_torch_dtype(self, *, torch: Any, device: str) -> Any:
        return torch.bfloat16 if device == "cuda" else torch.float32

    def build_pipeline_load_kwargs(self, *, device: str, torch_dtype: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        cache_dir = resolve_cache_dir(self.model_config)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if self.use_safetensors is not None:
            kwargs["use_safetensors"] = self.use_safetensors
        if device == "cuda":
            variant = self.cuda_variant()
            if variant:
                kwargs["variant"] = variant
        return kwargs

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {}

    def cuda_variant(self) -> str | None:
        return None


def _build_diffusers_progress_kwargs(
    *,
    pipe: Any,
    total_steps: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    if progress_callback is None:
        return {}
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return {}

    def legacy_callback(step_index: int, _timestep: Any, _latents: Any) -> None:
        progress_callback(
            {
                "phase": "generating",
                "current_step": int(step_index) + 1,
                "total_steps": int(total_steps),
                "supports_true_progress": True,
            }
        )

    def callback_on_step_end(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]):
        progress_callback(
            {
                "phase": "generating",
                "current_step": int(step_index) + 1,
                "total_steps": int(total_steps),
                "supports_true_progress": True,
            }
        )
        return callback_kwargs

    parameters = signature.parameters
    if "callback_on_step_end" in parameters:
        kwargs: dict[str, Any] = {"callback_on_step_end": callback_on_step_end}
        if "callback_on_step_end_tensor_inputs" in parameters:
            kwargs["callback_on_step_end_tensor_inputs"] = []
        return kwargs
    if "callback" in parameters:
        kwargs = {"callback": legacy_callback}
        if "callback_steps" in parameters:
            kwargs["callback_steps"] = 1
        return kwargs
    return {}


class ZImageTurboAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=8,
        preferred_batch_size=4,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["image"],
    )
    pipeline_class_name = "ZImagePipeline"
    default_guidance_scale = 0.0
    default_num_inference_steps = 9
    low_cpu_mem_usage = False


class FluxImageAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=4,
        preferred_batch_size=2,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["image"],
    )
    pipeline_class_name = "FluxPipeline"
    default_guidance_scale = 3.5
    default_num_inference_steps = 50

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {
            "max_sequence_length": int(plan.inputs.get("max_sequence_length", 512)),
        }

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"max_sequence_length": int(params.get("max_sequence_length", 512))}


class StableDiffusionXLAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=4,
        preferred_batch_size=2,
        supports_negative_prompt=True,
        supports_seed=True,
        output_types=["image"],
    )
    pipeline_class_name = "DiffusionPipeline"
    default_guidance_scale = 5.0
    default_num_inference_steps = 40
    use_safetensors = True

    def cuda_variant(self) -> str | None:
        return "fp16"

    def resolve_torch_dtype(self, *, torch: Any, device: str) -> Any:
        return torch.float16 if device == "cuda" else torch.float32


class QwenImageAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=4,
        preferred_batch_size=2,
        supports_negative_prompt=True,
        supports_seed=True,
        output_types=["image"],
    )
    pipeline_class_name = "DiffusionPipeline"
    guidance_argument_name = "true_cfg_scale"
    default_guidance_scale = 4.0
    default_num_inference_steps = 50


class HunyuanImageAdapter(MockCapableImageAdapter):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["image"],
    )

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
    ) -> dict[str, dict[str, Any]]:
        import torch
        from transformers import AutoModelForCausalLM

        prompt_ids = list(plan.inputs["prompt_ids"])
        if len(prompt_ids) != 1 or len(prompts) != 1:
            raise RuntimeError("HunyuanImage-3.0 real execution currently expects single-prompt tasks.")

        model_ref = (
            self.model_config.weights.get("local_path")
            or self.model_config.weights.get("weights_path")
            or plan.inputs.get("local_model_path")
            or self.model_config.weights.get("hf_repo")
        )
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": plan.inputs.get("attn_implementation", "sdpa"),
            "moe_impl": plan.inputs.get("moe_impl", "eager"),
        }
        cache_dir = resolve_cache_dir(self.model_config)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        if hasattr(model, "load_tokenizer"):
            model.load_tokenizer(model_ref)

        prompt_id = prompt_ids[0]
        prompt = prompts[0]
        output_path = Path(workdir) / f"{prompt_id}.png"
        image = model.generate_image(prompt=prompt, stream=bool(plan.inputs.get("stream", True)))
        image.save(output_path)
        return {
            prompt_id: {
                "path": str(output_path),
                "seed": int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None,
                "guidance_scale": float(plan.inputs["guidance_scale"]),
                "num_inference_steps": int(plan.inputs["num_inference_steps"]),
                "batch_id": plan.inputs.get("batch_id"),
                "batch_index": 0,
                "mock": False,
            }
        }

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "stream": bool(params.get("stream", True)),
            "attn_implementation": str(params.get("attn_implementation", "sdpa")),
            "moe_impl": str(params.get("moe_impl", "eager")),
            "local_model_path": params.get("local_model_path"),
        }


def resolve_image_dimensions(params: dict[str, Any]) -> tuple[int, int]:
    resolution = params.get("resolution")
    if isinstance(resolution, str) and "x" in resolution:
        left, right = resolution.lower().split("x", maxsplit=1)
        return int(left), int(right)
    width = int(params.get("width", 1024))
    height = int(params.get("height", 1024))
    return width, height


def resolve_negative_prompts(
    *,
    prompts: list[str],
    params: dict[str, Any],
    supports_negative_prompt: bool,
) -> list[str]:
    if not supports_negative_prompt:
        return ["" for _ in prompts]
    negative_prompts = list(params.get("negative_prompts", []))
    if negative_prompts and len(negative_prompts) != len(prompts):
        raise ValueError("negative_prompts must match the prompt batch length.")
    if not negative_prompts:
        return ["" for _ in prompts]
    return [str(item) for item in negative_prompts]


def resolve_model_reference(model_config: ModelInfo) -> str:
    return str(
        model_config.weights.get("local_path")
        or model_config.weights.get("weights_path")
        or model_config.weights.get("hf_repo")
    )


def resolve_cache_dir(model_config: ModelInfo) -> str | None:
    cache_dir = model_config.weights.get("hf_cache_dir")
    if cache_dir in (None, ""):
        return None
    return str(cache_dir)


def extract_png_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Unsupported image format for {path}; expected PNG.")
        _length = handle.read(4)
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"Invalid PNG header for {path}.")
        width, height = struct.unpack(">II", handle.read(8))
    return {"width": width, "height": height, "format": "png"}


def deterministic_color(*parts: str) -> tuple[int, int, int]:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def write_mock_png(
    path: str | Path,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_mock_png(width=width, height=height, color=color))
    return output_path


def build_mock_png(*, width: int, height: int, color: tuple[int, int, int]) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("Mock PNG dimensions must be positive.")
    pixel = bytes(color)
    row = b"\x00" + pixel * width
    image_data = row * height
    compressed = zlib.compress(image_data, level=9)

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", ihdr),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
    )
