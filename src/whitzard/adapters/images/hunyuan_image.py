from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import AdapterCapabilities, ExecutionPlan, ProgressCallback
from whitzard.adapters.images.base import BaseImageGenerationAdapter
from whitzard.adapters.images.common import resolve_cache_dir


class HunyuanImageAdapter(BaseImageGenerationAdapter):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["image"],
    )

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        del progress_callback
        import torch
        from transformers import AutoModelForCausalLM

        prompt_ids = list(plan.inputs["prompt_ids"])
        if len(prompt_ids) != 1 or len(prompts) != 1:
            raise RuntimeError("HunyuanImage-3.0 real execution currently expects single-prompt tasks.")

        model_ref = (
            self.model_config.weights.get("local_path")
            or self.model_config.weights.get("weights_path")
            or plan.inputs.get("local_model_path")
            or self.model_config.weights.get("hf_repo")
        )
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": plan.inputs.get("attn_implementation", "sdpa"),
            "moe_impl": plan.inputs.get("moe_impl", "eager"),
        }
        cache_dir = resolve_cache_dir(self.model_config)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        if hasattr(model, "load_tokenizer"):
            model.load_tokenizer(model_ref)

        prompt_id = prompt_ids[0]
        prompt = prompts[0]
        output_path = Path(workdir) / f"{prompt_id}.png"
        image = model.generate_image(prompt=prompt, stream=bool(plan.inputs.get("stream", True)))
        image.save(output_path)
        return {
            prompt_id: {
                "path": str(output_path),
                "seed": int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None,
                "guidance_scale": float(plan.inputs["guidance_scale"]),
                "num_inference_steps": int(plan.inputs["num_inference_steps"]),
                "batch_id": plan.inputs.get("batch_id"),
                "batch_index": 0,
                "mock": False,
            }
        }

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "stream": bool(params.get("stream", True)),
            "attn_implementation": str(params.get("attn_implementation", "sdpa")),
            "moe_impl": str(params.get("moe_impl", "eager")),
            "local_model_path": params.get("local_model_path"),
        }
