from __future__ import annotations

from typing import Any

from aigc.adapters.base import AdapterCapabilities, ExecutionPlan, ProgressCallback
from aigc.adapters.videos.common import (
    build_diffusers_progress_kwargs,
    normalize_frame_batches,
    resolve_video_cache_dir,
    resolve_video_model_reference,
    validate_local_diffusers_reference,
)
from aigc.adapters.videos.diffusers_base import DiffusersVideoAdapterBase


class WanT2VDiffusersAdapter(DiffusersVideoAdapterBase):
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
    pipeline_class_name = "WanPipeline"
    default_width = 1280
    default_height = 720
    default_fps = 16
    default_num_frames = 81
    default_num_inference_steps = 40
    default_guidance_scale = 4.0

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"guidance_scale_2": float(params.get("guidance_scale_2", 3.0))}

    def validate_model_reference(self, model_ref: str) -> None:
        validate_local_diffusers_reference(
            model_config=self.model_config,
            model_ref=model_ref,
            required_files=("model_index.json", "vae/config.json"),
            adapter_specific_hint=(
                "For Wan2.2-T2V-A14B-Diffusers, weights_path/local_path should point to the "
                "local Diffusers weights directory for Wan-AI/Wan2.2-T2V-A14B-Diffusers."
            ),
        )

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        from diffusers import AutoencoderKLWan

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        vae = AutoencoderKLWan.from_pretrained(
            model_ref,
            subfolder="vae",
            torch_dtype=torch.float32,
            **({"cache_dir": cache_dir} if cache_dir else {}),
        )
        pipe = pipeline_class.from_pretrained(
            model_ref,
            vae=vae,
            **load_kwargs,
        )
        if hasattr(pipe, "to"):
            pipe.to(device)
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {"guidance_scale_2": float(plan.inputs.get("guidance_scale_2", 3.0))}

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
        output = pipe(
            prompt=prompts[0] if len(prompts) == 1 else prompts,
            negative_prompt=negative_prompts[0] if len(negative_prompts) == 1 else negative_prompts,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=float(plan.inputs.get("guidance_scale_2", 3.0)),
            num_inference_steps=num_inference_steps,
            **build_diffusers_progress_kwargs(
                pipe=pipe,
                total_steps=num_inference_steps,
                progress_callback=progress_callback,
            ),
            **({"generator": generator} if generator is not None else {}),
        )
        frames = getattr(output, "frames", None)
        return normalize_frame_batches(self.model_config.name, frames)
