from __future__ import annotations

from typing import Any

from aigc.adapters.base import AdapterCapabilities, ExecutionPlan
from aigc.adapters.images.base import DiffusersImageAdapterBase


class FluxImageAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=4,
        preferred_batch_size=2,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["image"],
    )
    pipeline_class_name = "FluxPipeline"
    default_guidance_scale = 3.5
    default_num_inference_steps = 50

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {
            "max_sequence_length": int(plan.inputs.get("max_sequence_length", 512)),
        }

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"max_sequence_length": int(params.get("max_sequence_length", 512))}
