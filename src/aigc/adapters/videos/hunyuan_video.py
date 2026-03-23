from __future__ import annotations

from typing import Any

from aigc.adapters.base import AdapterCapabilities, ProgressCallback
from aigc.adapters.videos.common import resolve_video_cache_dir
from aigc.adapters.videos.diffusers_base import DiffusersVideoAdapterBase


class HunyuanVideo15Adapter(DiffusersVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=2,
        preferred_batch_size=2,
        supports_negative_prompt=True,
        supports_seed=True,
        output_types=["video"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )
    pipeline_class_name = "HunyuanVideo15Pipeline"
    default_width = 1280
    default_height = 720
    default_fps = 24
    default_num_frames = 121
    default_num_inference_steps = 50

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        model_ref = (
            self.model_config.weights.get("local_path")
            or self.model_config.weights.get("weights_path")
            or self.model_config.weights.get("diffusers_repo")
            or self.model_config.weights["hf_repo"]
        )
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        pipe = pipeline_class.from_pretrained(model_ref, **load_kwargs)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: Any,
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
        del guidance_scale
        from aigc.adapters.videos.common import build_diffusers_progress_kwargs, normalize_frame_batches

        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        kwargs: dict[str, Any] = {
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
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
        output = pipe(**kwargs)
        frames = getattr(output, "frames", None)
        return normalize_frame_batches(self.model_config.name, frames)
