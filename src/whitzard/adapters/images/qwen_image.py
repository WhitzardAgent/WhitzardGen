from __future__ import annotations

from whitzard.adapters.base import AdapterCapabilities
from whitzard.adapters.images.base import DiffusersImageAdapterBase


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
