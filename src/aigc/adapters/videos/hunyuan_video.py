from __future__ import annotations

from typing import Any

from aigc.adapters.base import AdapterCapabilities
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
    default_guidance_scale = 4.0

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
