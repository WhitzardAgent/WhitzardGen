from __future__ import annotations

from typing import Any

from whitzard.adapters.base import AdapterCapabilities
from whitzard.adapters.images.base import DiffusersImageAdapterBase


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
