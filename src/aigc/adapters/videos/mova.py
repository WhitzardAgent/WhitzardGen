from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc.adapters.base import AdapterCapabilities
from aigc.adapters.videos.base import ExternalProcessVideoAdapterBase


class MOVAVideoAdapter(ExternalProcessVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video", "audio"],
    )
    default_width = 1280
    default_height = 720
    default_fps = 24
    default_num_frames = 193
    default_num_inference_steps = 25
    default_guidance_scale = 5.0

    def build_real_command(
        self,
        *,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
        inputs: dict[str, Any],
    ) -> list[str]:
        prompt = prompts[0]
        checkpoint_dir = str(
            params.get("checkpoint_dir")
            or self.model_config.weights.get("weights_path")
            or self.model_config.weights.get("local_path")
            or "/path/to/MOVA-720p"
        )
        output_path = str(Path(workdir) / f"{prompt_ids[0]}.mp4")
        cp_size = str(params.get("cp_size", 1))
        command = [
            "torchrun",
            f"--nproc_per_node={cp_size}",
            "scripts/inference_single.py",
            "--ckpt_path",
            checkpoint_dir,
            "--cp_size",
            cp_size,
            "--height",
            str(inputs["height"]),
            "--width",
            str(inputs["width"]),
            "--prompt",
            prompt,
            "--output_path",
            output_path,
            "--offload",
            str(params.get("offload", "cpu")),
        ]
        if inputs.get("seed") not in (None, ""):
            command.extend(["--seed", str(inputs["seed"])])
        ref_path = params.get("ref_path")
        if ref_path:
            command.extend(["--ref_path", str(ref_path)])
        return command
