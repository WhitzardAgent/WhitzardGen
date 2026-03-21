from __future__ import annotations

from typing import Any

from aigc.adapters.base import AdapterCapabilities, ExecutionPlan, ProgressCallback
from aigc.adapters.videos.common import (
    build_diffusers_progress_kwargs,
    normalize_frame_batches,
    resolve_video_cache_dir,
    resolve_video_model_reference,
)
from aigc.adapters.videos.diffusers_base import DiffusersVideoAdapterBase


class CogVideoX5BAdapter(DiffusersVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=2,
        preferred_batch_size=2,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )
    pipeline_class_name = "CogVideoXPipeline"
    default_width = 720
    default_height = 480
    default_fps = 8
    default_num_frames = 49
    default_num_inference_steps = 50
    default_guidance_scale = 6.0

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
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

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
        del negative_prompts, width, height
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        output = pipe(
            prompt=prompts[0] if len(prompts) == 1 else prompts,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            **build_diffusers_progress_kwargs(
                pipe=pipe,
                total_steps=num_inference_steps,
                progress_callback=progress_callback,
            ),
            **({"generator": generator} if generator is not None else {}),
        )
        frames = getattr(output, "frames", None)
        return normalize_frame_batches(self.model_config.name, frames)
