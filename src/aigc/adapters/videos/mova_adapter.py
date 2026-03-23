from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from aigc.adapters.base import AdapterCapabilities, ExecutionPlan, ProgressCallback
from aigc.adapters.videos.base import BaseVideoGenerationAdapter
from aigc.adapters.videos.common import compute_duration_sec, resolve_video_repo_dir

try:
    from mova.datasets.transforms.custom import crop_and_resize
    from mova.diffusion.pipelines.pipeline_mova import MOVA
    from mova.utils.data import save_video_with_audio
    _MOVA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional model env
    crop_and_resize = None  # type: ignore[assignment]
    MOVA = None  # type: ignore[assignment]
    save_video_with_audio = None  # type: ignore[assignment]
    _MOVA_IMPORT_ERROR = exc


_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


class MOVAVideoAdapter(BaseVideoGenerationAdapter):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=True,
        supports_seed=True,
        output_types=["video"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )
    real_execution_mode = "in_process"
    default_width = 1280
    default_height = 720
    default_fps = 24
    default_num_frames = 193
    default_num_inference_steps = 50
    default_guidance_scale = 5.0

    def __init__(self, model_config: Any) -> None:
        super().__init__(model_config)
        self._loaded_pipeline: Any | None = None
        self._loaded_torch: Any | None = None
        self._loaded_device: str | None = None
        self._loaded_image_module: Any | None = None
        self._loaded_crop_and_resize: Any | None = None
        self._loaded_save_video_with_audio: Any | None = None

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_pipeline()

    def unload_persistent_worker(self) -> None:
        self._loaded_pipeline = None
        self._loaded_torch = None
        self._loaded_device = None
        self._loaded_image_module = None
        self._loaded_crop_and_resize = None
        self._loaded_save_video_with_audio = None

    def prepare(
        self,
        *,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionPlan:
        plan = super().prepare(
            prompts=prompts,
            prompt_ids=prompt_ids,
            params=params,
            workdir=workdir,
        )
        runtime = dict(plan.inputs.get("runtime", {}))
        if runtime.get("execution_mode") != "mock":
            ref_path = str(plan.inputs.get("ref_path") or "").strip()
            if not ref_path:
                raise RuntimeError(
                    "MOVA-720p requires `ref_path` (or `image_path`) for in-process inference."
                )
            self._resolve_checkpoint_dir(plan.inputs)
            cp_size = int(plan.inputs.get("cp_size", 1))
            if cp_size != 1:
                raise RuntimeError(
                    "Current in-process MOVA integration supports cp_size=1 only. "
                    "Use multiple replicas for multi-GPU scaling."
                )
        return plan

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        ref_path = params.get("ref_path") or params.get("image_path")
        return {
            "checkpoint_dir": str(params.get("checkpoint_dir"))
            if params.get("checkpoint_dir") not in (None, "")
            else None,
            "ref_path": str(ref_path) if ref_path not in (None, "") else None,
            "offload": str(params.get("offload", "cpu")),
            "offload_to_disk_path": str(params.get("offload_to_disk_path"))
            if params.get("offload_to_disk_path") not in (None, "")
            else None,
            "cp_size": int(params.get("cp_size", 1)),
            "attn_type": str(params.get("attn_type", "fa")),
            "sigma_shift": float(params.get("sigma_shift", 5.0)),
            "cfg_scale": float(
                params.get("cfg_scale", params.get("guidance_scale", self.default_guidance_scale))
            ),
            "remove_video_dit": bool(params.get("remove_video_dit", False)),
        }

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        fps = int(plan.inputs["fps"])
        num_frames = int(plan.inputs["num_frames"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        sigma_shift = float(plan.inputs.get("sigma_shift", 5.0))
        cfg_scale = float(
            plan.inputs.get("cfg_scale", plan.inputs.get("guidance_scale", self.default_guidance_scale))
        )
        negative_prompts = [str(value) for value in plan.inputs.get("negative_prompts", [""])]
        prompt_ids = [str(value) for value in plan.inputs["prompt_ids"]]
        ref_path = Path(str(plan.inputs["ref_path"]))
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)

        if progress_callback is not None:
            progress_callback({"phase": "preparing_batch", "supports_true_progress": False})

        pipe, torch, _device, image_module, crop_resize_fn, save_video_fn = self._get_or_load_pipeline(
            plan.inputs
        )
        image = image_module.open(ref_path).convert("RGB")
        reference_image = crop_resize_fn(image, height=height, width=width)

        if progress_callback is not None:
            progress_callback({"phase": "generating", "supports_true_progress": False})

        negative_prompt = negative_prompts[0] or _DEFAULT_NEGATIVE_PROMPT
        prompt = prompts[0]
        video, audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            image=reference_image,
            height=height,
            width=width,
            video_fps=fps,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            cfg_scale=cfg_scale,
            seed=seed,
            cp_mesh=None,
            remove_video_dit=bool(plan.inputs.get("remove_video_dit", False)),
        )

        if progress_callback is not None:
            progress_callback({"phase": "exporting", "supports_true_progress": False})

        output_path = Path(workdir) / f"{prompt_ids[0]}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_sample = audio[0]
        if hasattr(audio_sample, "cpu"):
            audio_sample = audio_sample.cpu()
        if hasattr(audio_sample, "squeeze"):
            audio_sample = audio_sample.squeeze()
        save_video_fn(
            video[0],
            audio_sample,
            str(output_path),
            fps=fps,
            sample_rate=int(getattr(pipe, "audio_sample_rate", 16000)),
            quality=9,
        )

        if progress_callback is not None:
            progress_callback({"phase": "completed", "supports_true_progress": False})

        return {
            prompt_ids[0]: {
                "path": str(output_path),
                "seed": seed,
                "guidance_scale": cfg_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": plan.inputs.get("batch_id"),
                "batch_index": 0,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": compute_duration_sec(num_frames=num_frames, fps=fps),
                "mock": False,
            }
        }

    def _get_or_load_pipeline(
        self,
        load_params: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, str, Any, Any, Any]:
        if (
            self._loaded_pipeline is not None
            and self._loaded_torch is not None
            and self._loaded_device is not None
            and self._loaded_image_module is not None
            and self._loaded_crop_and_resize is not None
            and self._loaded_save_video_with_audio is not None
        ):
            return (
                self._loaded_pipeline,
                self._loaded_torch,
                self._loaded_device,
                self._loaded_image_module,
                self._loaded_crop_and_resize,
                self._loaded_save_video_with_audio,
            )

        if MOVA is None or crop_and_resize is None or save_video_with_audio is None:
            raise RuntimeError(
                "MOVA-720p requires the `mova` Python package to be installed in the target "
                "environment and importable directly. repo_path is no longer used to inject the "
                "package dynamically."
            ) from _MOVA_IMPORT_ERROR

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        params = dict(self.model_config.generation_defaults)
        if load_params:
            params.update(load_params)

        checkpoint_dir = self._resolve_checkpoint_dir(params)
        print(checkpoint_dir)
        repo_dir = resolve_video_repo_dir(self.model_config)
        if repo_dir and not Path(repo_dir).exists():
            raise RuntimeError(f"MOVA-720p configured repo_path does not exist: {repo_dir}")

        pipe = MOVA.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
        )
        print(pipe)

        offload = str(params.get("offload", "cpu"))
        if offload == "none":
            if hasattr(pipe, "to"):
                pipe.to(torch.device("cuda", 0) if device == "cuda" else torch.device("cpu"))
        elif offload == "cpu":
            if hasattr(pipe, "enable_model_cpu_offload"):
                try:
                    pipe.enable_model_cpu_offload(0)
                except TypeError:
                    pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "to"):
                pipe.to(torch.device("cuda", 0) if device == "cuda" else torch.device("cpu"))
        elif offload == "group":
            if not hasattr(pipe, "enable_group_offload"):
                raise RuntimeError("MOVA pipeline does not support group offload in this environment.")
            pipe.enable_group_offload(
                onload_device=torch.device("cuda", 0) if device == "cuda" else torch.device("cpu"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
                low_cpu_mem_usage=True,
                offload_to_disk_path=params.get("offload_to_disk_path"),
            )
        else:
            raise ValueError(f"Unknown MOVA offload strategy: {offload}")

        self._loaded_pipeline = pipe
        self._loaded_torch = torch
        self._loaded_device = device
        self._loaded_image_module = Image
        self._loaded_crop_and_resize = crop_and_resize
        self._loaded_save_video_with_audio = save_video_with_audio
        return pipe, torch, device, Image, crop_and_resize, save_video_with_audio

    def _resolve_checkpoint_dir(self, params: dict[str, Any]) -> str:
        checkpoint_dir = params.get("checkpoint_dir")
        if checkpoint_dir not in (None, ""):
            resolved = str(checkpoint_dir).strip()
        else:
            resolved = str(
                self.model_config.weights.get("weights_path")
                or self.model_config.weights.get("local_path")
                or ""
            ).strip()
        if not resolved:
            raise RuntimeError(
                "MOVA-720p requires a local checkpoint directory via weights_path/local_path "
                "or an explicit checkpoint_dir. hf_repo fallback is intentionally disabled for "
                "this model to avoid accidental remote loading."
            )
        checkpoint_path = Path(resolved)
        if not checkpoint_path.exists():
            raise RuntimeError(f"MOVA-720p checkpoint directory does not exist: {checkpoint_path}")
        return str(checkpoint_path)
