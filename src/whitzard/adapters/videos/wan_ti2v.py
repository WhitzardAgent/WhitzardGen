from __future__ import annotations

from typing import Any

from whitzard.adapters.base import AdapterCapabilities
from whitzard.adapters.videos.base import ExternalProcessVideoAdapterBase


class WanTI2VAdapter(ExternalProcessVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video"],
    )
    default_width = 1280
    default_height = 704
    default_fps = 24
    default_num_frames = 121
    default_num_inference_steps = 40
    default_guidance_scale = 4.0

    def build_real_command(
        self,
        *,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
        inputs: dict[str, Any],
    ) -> list[str]:
        del prompt_ids, workdir
        prompt = prompts[0]
        checkpoint_dir = str(
            params.get("checkpoint_dir")
            or self.model_config.weights.get("weights_path")
            or self.model_config.weights.get("local_path")
            or "./Wan2.2-TI2V-5B"
        )
        size = f"{inputs['width']}*{inputs['height']}"
        command = [
            "python",
            "generate.py",
            "--task",
            "ti2v-5B",
            "--size",
            size,
            "--ckpt_dir",
            checkpoint_dir,
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--prompt",
            prompt,
        ]
        image_path = params.get("image_path")
        if image_path:
            command.extend(["--image", str(image_path)])
        return command
