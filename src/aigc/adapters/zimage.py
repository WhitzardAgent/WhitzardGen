from __future__ import annotations

from aigc.adapters.base import AdapterCapabilities, ExecutionPlan
from aigc.adapters.image_family import DiffusersImageAdapterBase


class ZImageAdapter(DiffusersImageAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=8,
        preferred_batch_size=4,
        supports_negative_prompt=True,
        supports_seed=True,
        output_types=["image"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )
    pipeline_class_name = "ZImagePipeline"
    default_guidance_scale = 4.0
    default_num_inference_steps = 50
    include_cfg_normalization = True
    low_cpu_mem_usage = False

    def build_pipeline_load_kwargs(self, *, device: str, torch_dtype: object) -> dict[str, object]:
        kwargs = super().build_pipeline_load_kwargs(device=device, torch_dtype=torch_dtype)
        kwargs["low_cpu_mem_usage"] = False
        return kwargs

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, object]:
        return {"cfg_normalization": bool(plan.inputs.get("cfg_normalization", False))}
