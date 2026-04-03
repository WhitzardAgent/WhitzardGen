from __future__ import annotations

from whitzard.adapters.base import AdapterCapabilities
from whitzard.adapters.images.base import DiffusersImageAdapterBase


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
