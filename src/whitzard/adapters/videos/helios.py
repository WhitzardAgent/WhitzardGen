from __future__ import annotations

import json
import inspect
from typing import Any

from whitzard.adapters.base import AdapterCapabilities, ExecutionPlan, ProgressCallback
from whitzard.adapters.videos.common import (
    build_diffusers_progress_kwargs,
    normalize_frame_batches,
    resolve_video_cache_dir,
    resolve_video_model_reference,
    validate_local_diffusers_reference,
)
from whitzard.adapters.videos.diffusers_base import DiffusersVideoAdapterBase


class HeliosPyramidAdapter(DiffusersVideoAdapterBase):
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
    pipeline_class_name = "HeliosPyramidPipeline"
    default_width = 640
    default_height = 384
    default_fps = 24
    default_num_frames = 240
    default_num_inference_steps = 6
    default_guidance_scale = 1.0

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_steps = params.get("pyramid_num_inference_steps_list", [2, 2, 2])
        if isinstance(raw_steps, str):
            stripped = raw_steps.strip()
            if stripped.startswith("["):
                parsed_steps = json.loads(stripped)
            else:
                parsed_steps = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            parsed_steps = raw_steps
        pyramid_steps = [int(value) for value in list(parsed_steps)]
        if not pyramid_steps:
            raise ValueError("pyramid_num_inference_steps_list must not be empty.")
        return {
            "pyramid_num_inference_steps_list": pyramid_steps,
            "is_amplify_first_chunk": bool(params.get("is_amplify_first_chunk", True)),
        }

    def validate_model_reference(self, model_ref: str) -> None:
        validate_local_diffusers_reference(
            model_config=self.model_config,
            model_ref=model_ref,
            required_files=("model_index.json", "vae/config.json"),
            adapter_specific_hint=(
                "For Helios, weights_path/local_path should point to the local Diffusers weights "
                "directory for BestWishYsh/Helios-Distilled."
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
        from diffusers import AutoModel

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        vae = AutoModel.from_pretrained(
            model_ref,
            subfolder="vae",
            torch_dtype=torch.float32,
            **({"cache_dir": cache_dir} if cache_dir else {}),
        )
        pipe = pipeline_class.from_pretrained(model_ref, vae=vae, **load_kwargs)
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
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators

        pyramid_steps = [
            int(value) for value in plan.inputs.get("pyramid_num_inference_steps_list", [2, 2, 2])
        ]
        total_steps = sum(pyramid_steps) if pyramid_steps else num_inference_steps
        kwargs: dict[str, Any] = {
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "num_frames": num_frames,
            "pyramid_num_inference_steps_list": pyramid_steps,
            "guidance_scale": guidance_scale,
            "is_amplify_first_chunk": bool(plan.inputs.get("is_amplify_first_chunk", True)),
        }
        if self.capabilities.supports_negative_prompt:
            kwargs["negative_prompt"] = (
                negative_prompts[0] if len(negative_prompts) == 1 else negative_prompts
            )
        if generator is not None:
            kwargs["generator"] = generator
        kwargs.update(
            build_diffusers_progress_kwargs(
                pipe=pipe,
                total_steps=total_steps,
                progress_callback=progress_callback,
            )
        )

        try:
            signature = inspect.signature(pipe.__call__)
            parameters = signature.parameters
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            parameters = {}
            accepts_var_kwargs = True

        if accepts_var_kwargs or "width" in parameters:
            kwargs["width"] = width
        if accepts_var_kwargs or "height" in parameters:
            kwargs["height"] = height

        output = pipe(**kwargs)
        frames = getattr(output, "frames", None)
        return normalize_frame_batches(self.model_config.name, frames)
