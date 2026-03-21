from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc.adapters.base import AdapterCapabilities, ProgressCallback
from aigc.adapters.videos.common import (
    normalize_single_video_frames,
    resolve_video_cache_dir,
    resolve_video_model_reference,
    resolve_video_repo_dir,
    temporary_repo_import_path,
    torch_gc,
    validate_local_video_directory,
)
from aigc.adapters.videos.diffusers_base import DiffusersVideoAdapterBase


class LongCatVideoAdapter(DiffusersVideoAdapterBase):
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
    real_execution_mode = "in_process"
    default_width = 1280
    default_height = 720
    default_fps = 30
    default_num_frames = 121
    default_num_inference_steps = 50
    default_guidance_scale = 4.0

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"use_distill": bool(params.get("use_distill", False))}

    def validate_model_reference(self, model_ref: str) -> None:
        validate_local_video_directory(
            model_name=self.model_config.name,
            configured_label="weights_path/local_path",
            configured_path=model_ref,
            required_entries=("tokenizer", "text_encoder", "vae", "scheduler", "dit"),
            repo_path=resolve_video_repo_dir(self.model_config),
            repo_hint=(
                "For LongCat-Video, repo_path should point to the LongCat-Video checkout, "
                "and weights_path/local_path should point to the local checkpoint directory."
            ),
        )

    def _get_or_load_pipeline(self) -> tuple[Any, Any, str]:
        if self._loaded_pipeline is not None and self._loaded_torch is not None and self._loaded_device:
            return self._loaded_pipeline, self._loaded_torch, self._loaded_device

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipe = self.load_pipeline(torch=torch, device=device, dtype=dtype)
        self._loaded_pipeline = pipe
        self._loaded_torch = torch
        self._loaded_device = device
        return pipe, torch, device

    def load_pipeline(
        self,
        *,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        checkpoint_dir = resolve_video_model_reference(self.model_config)
        repo_dir = resolve_video_repo_dir(self.model_config)
        cache_dir = resolve_video_cache_dir(self.model_config)
        self.validate_model_reference(checkpoint_dir)

        component_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if cache_dir:
            component_kwargs["cache_dir"] = cache_dir

        with temporary_repo_import_path(repo_dir):
            from transformers import AutoTokenizer, UMT5EncoderModel

            from longcat_video.context_parallel import context_parallel_util
            from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
            from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
            from longcat_video.modules.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )
            from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

            cp_split_hw = context_parallel_util.get_optimal_split(1)
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                subfolder="tokenizer",
                **component_kwargs,
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                checkpoint_dir,
                subfolder="text_encoder",
                **component_kwargs,
            )
            vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder="vae",
                **component_kwargs,
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir,
                subfolder="scheduler",
                **component_kwargs,
            )
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                cp_split_hw=cp_split_hw,
                **component_kwargs,
            )
            pipe = LongCatVideoPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                scheduler=scheduler,
                dit=dit,
            )
        if hasattr(pipe, "to"):
            pipe.to(device)

        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            dit = getattr(pipe, "dit", None)
            if dit is not None and hasattr(dit, "load_lora"):
                for lora_name in ("cfg_step_lora", "refinement_lora"):
                    lora_path = checkpoint_path / "lora" / f"{lora_name}.safetensors"
                    if lora_path.exists():
                        dit.load_lora(str(lora_path), lora_name)
        return pipe

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: Any,
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
        del progress_callback
        use_distill = bool(plan.inputs.get("use_distill", False))
        effective_steps = 16 if use_distill else num_inference_steps
        effective_guidance = 1.0 if use_distill else guidance_scale
        frame_batches: list[list[Any]] = []
        for batch_index, prompt in enumerate(prompts):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed + batch_index)
            dit = getattr(pipe, "dit", None)
            if use_distill and dit is not None and hasattr(dit, "enable_loras"):
                dit.enable_loras(["cfg_step_lora"])
            output = pipe.generate_t2v(
                prompt=prompt,
                negative_prompt=None if use_distill else negative_prompts[batch_index],
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=effective_steps,
                use_distill=use_distill,
                guidance_scale=effective_guidance,
                **({"generator": generator} if generator is not None else {}),
            )[0]
            if dit is not None and hasattr(dit, "disable_all_loras"):
                dit.disable_all_loras()
            frame_batches.append(normalize_single_video_frames(self.model_config.name, output))
            torch_gc(torch)
        return frame_batches
