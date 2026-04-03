from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import ExecutionPlan, ProgressCallback
from whitzard.adapters.videos.base import BaseVideoGenerationAdapter
from whitzard.adapters.videos.common import (
    build_diffusers_progress_kwargs,
    compute_duration_sec,
    normalize_frame_batches,
    resolve_video_cache_dir,
    resolve_video_model_reference,
)


class DiffusersVideoAdapterBase(BaseVideoGenerationAdapter):
    real_execution_mode = "in_process"
    pipeline_class_name = ""

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
        from diffusers.utils import export_to_video

        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        fps = int(plan.inputs["fps"])
        num_frames = int(plan.inputs["num_frames"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        guidance_scale = float(plan.inputs["guidance_scale"])
        negative_prompts = [str(value) for value in plan.inputs.get("negative_prompts", [""])]
        prompt_ids = [str(value) for value in plan.inputs["prompt_ids"]]
        pipe, torch, device = self._get_or_load_pipeline()
        frame_batches = self.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=prompts,
            negative_prompts=negative_prompts,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            torch=torch,
            device=device,
            progress_callback=progress_callback,
        )
        if len(frame_batches) != len(prompt_ids):
            raise RuntimeError(
                f"{self.model_config.name} returned {len(frame_batches)} videos for {len(prompt_ids)} prompts."
            )
        outputs: dict[str, dict[str, Any]] = {}
        for batch_index, (prompt_id, frames) in enumerate(zip(prompt_ids, frame_batches, strict=True)):
            output_path = Path(workdir) / f"{prompt_id}.mp4"
            export_to_video(frames, str(output_path), fps=fps)
            outputs[prompt_id] = {
                "path": str(output_path),
                "seed": seed + batch_index if seed is not None else None,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": plan.inputs.get("batch_id"),
                "batch_index": batch_index,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": compute_duration_sec(num_frames=num_frames, fps=fps),
                "mock": False,
            }
        return outputs

    def _get_or_load_pipeline(self) -> tuple[Any, Any, str]:
        if self._loaded_pipeline is not None and self._loaded_torch is not None and self._loaded_device:
            return self._loaded_pipeline, self._loaded_torch, self._loaded_device

        import diffusers
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipeline_class = getattr(diffusers, self.pipeline_class_name)
        pipe = self.load_pipeline(
            pipeline_class=pipeline_class,
            torch=torch,
            device=device,
            dtype=dtype,
        )
        self._loaded_pipeline = pipe
        self._loaded_torch = torch
        self._loaded_device = device
        return pipe, torch, device

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        pipe = pipeline_class.from_pretrained(model_ref, **load_kwargs)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "to"):
            pipe.to(device)
        return pipe

    def validate_model_reference(self, model_ref: str) -> None:
        from whitzard.adapters.videos.common import validate_local_diffusers_reference

        validate_local_diffusers_reference(
            model_config=self.model_config,
            model_ref=model_ref,
            required_files=("model_index.json",),
        )

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {}

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: ExecutionPlan,
        prompts: list[str],
        negative_prompts: list[str],
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        torch: Any,
        device: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[Any]]:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        kwargs = {
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        kwargs.update(
            build_diffusers_progress_kwargs(
                pipe=pipe,
                total_steps=num_inference_steps,
                progress_callback=progress_callback,
            )
        )
        if generator is not None:
            kwargs["generator"] = generator
        if self.capabilities.supports_negative_prompt:
            kwargs["negative_prompt"] = (
                negative_prompts[0] if len(negative_prompts) == 1 else negative_prompts
            )
        kwargs.update(self.build_pipeline_call_kwargs(plan))
        output = pipe(**kwargs)
        frames = getattr(output, "frames", None)
        return normalize_frame_batches(self.model_config.name, frames)
