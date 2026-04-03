from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import (
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
    ProgressCallback,
)
from whitzard.adapters.images.common import (
    build_diffusers_progress_kwargs,
    deterministic_color,
    extract_png_metadata,
    resolve_cache_dir,
    resolve_image_dimensions,
    resolve_model_reference,
    resolve_negative_prompts,
    write_mock_png,
)


class BaseImageGenerationAdapter(BaseAdapter):
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


class DiffusersImageAdapterBase(BaseImageGenerationAdapter):
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
            build_diffusers_progress_kwargs(
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

        import diffusers
        import torch

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
