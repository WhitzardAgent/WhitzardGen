from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aigc.adapters.base import (
    AdapterCapabilities,
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
)


class MockCapableVideoAdapter(BaseAdapter):
    real_execution_mode = "in_process"
    default_width = 1280
    default_height = 720
    default_fps = 24
    default_num_frames = 121
    default_num_inference_steps = 40
    default_guidance_scale = 4.0
    supports_negative_prompt = False

    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionPlan:
        runtime = dict(params.get("_runtime_config", {}))
        width, height = resolve_video_dimensions(
            params,
            default_width=self.default_width,
            default_height=self.default_height,
        )
        fps = int(params.get("fps", self.default_fps))
        num_frames = int(params.get("num_frames", self.default_num_frames))
        num_inference_steps = int(
            params.get("num_inference_steps", self.default_num_inference_steps)
        )
        guidance_scale = float(params.get("guidance_scale", self.default_guidance_scale))
        negative_prompts = resolve_video_negative_prompts(
            prompts=prompts,
            params=params,
            supports_negative_prompt=self.capabilities.supports_negative_prompt,
        )
        inputs = {
            "prompt_ids": list(prompt_ids),
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompts": negative_prompts,
            "runtime": runtime,
            "expected_outputs": {
                prompt_id: {"path": str(Path(workdir) / f"{prompt_id}.mp4")}
                for prompt_id in prompt_ids
            },
        }
        if params.get("seed") not in (None, ""):
            inputs["seed"] = int(params["seed"])
        inputs.update(self.extra_prepare_inputs(params))

        if runtime.get("execution_mode") == "mock":
            return ExecutionPlan(mode="in_process", inputs=inputs)
        if self.real_execution_mode == "external_process":
            return ExecutionPlan(
                mode="external_process",
                command=self.build_real_command(
                    prompts=prompts,
                    prompt_ids=prompt_ids,
                    params=params,
                    workdir=workdir,
                    inputs=inputs,
                ),
                cwd=self.real_command_cwd(params=params, workdir=workdir),
                timeout_sec=self.real_command_timeout_sec(params=params),
                inputs=inputs,
            )
        return ExecutionPlan(mode="in_process", inputs=inputs)

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionResult:
        runtime = dict(plan.inputs.get("runtime", {}))
        if runtime.get("execution_mode") == "mock":
            outputs = self._execute_mock(plan=plan, prompts=prompts, workdir=workdir)
            return ExecutionResult(
                exit_code=0,
                logs=f"{self.model_config.name} mock video generation completed.",
                outputs=outputs,
            )
        outputs = self._execute_real(plan=plan, prompts=prompts, workdir=workdir)
        return ExecutionResult(
            exit_code=0,
            logs=f"{self.model_config.name} video generation completed.",
            outputs=outputs,
        )

    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        batch_id = plan.inputs.get("batch_id")
        runtime = dict(plan.inputs.get("runtime", {}))
        expected_outputs = dict(plan.inputs.get("expected_outputs", {}))
        items: list[BatchItemResult] = []
        success_count = 0

        for fallback_index, prompt_id in enumerate(prompt_ids):
            output = dict(expected_outputs.get(prompt_id, {}))
            output.update(exec_result.outputs.get(prompt_id, {}))
            output_path = Path(str(output.get("path", "")))
            if not output_path.exists():
                items.append(
                    BatchItemResult(
                        prompt_id=prompt_id,
                        artifacts=[],
                        status="failed",
                        metadata={
                            "batch_id": output.get("batch_id", batch_id),
                            "batch_index": output.get("batch_index", fallback_index),
                        },
                        error_message=f"Expected video artifact missing for {prompt_id}",
                    )
                )
                continue

            artifact_metadata = extract_video_metadata(
                output_path,
                fallback={
                    "width": output.get("width", plan.inputs.get("width")),
                    "height": output.get("height", plan.inputs.get("height")),
                    "fps": output.get("fps", plan.inputs.get("fps")),
                    "num_frames": output.get("num_frames", plan.inputs.get("num_frames")),
                    "duration_sec": output.get(
                        "duration_sec",
                        compute_duration_sec(
                            num_frames=int(output.get("num_frames", plan.inputs.get("num_frames", 1))),
                            fps=int(output.get("fps", plan.inputs.get("fps", 1))),
                        ),
                    ),
                },
            )
            artifact_metadata.update(
                {
                    "seed": output.get("seed"),
                    "guidance_scale": output.get("guidance_scale"),
                    "num_inference_steps": output.get("num_inference_steps"),
                    "mock": bool(output.get("mock", False)),
                }
            )
            item_metadata = {
                "seed": output.get("seed"),
                "guidance_scale": output.get("guidance_scale"),
                "num_inference_steps": output.get("num_inference_steps"),
                "batch_id": output.get("batch_id", batch_id),
                "batch_index": output.get("batch_index", fallback_index),
                "mock": bool(output.get("mock", False)),
                "replica_id": runtime.get("replica_id"),
                "gpu_assignment": list(runtime.get("gpu_assignment", [])),
            }
            items.append(
                BatchItemResult(
                    prompt_id=prompt_id,
                    artifacts=[
                        ArtifactRecord(
                            type="video",
                            path=str(output_path),
                            metadata=artifact_metadata,
                        )
                    ],
                    status="success",
                    metadata=item_metadata,
                )
            )
            success_count += 1

        if success_count == len(prompt_ids):
            status = "success"
        elif success_count == 0:
            status = "failed"
        else:
            status = "partial_success"

        return ModelResult(
            status=status,
            batch_items=items,
            logs=exec_result.logs,
            metadata={
                "batch_id": batch_id,
                "execution_mode": str(runtime.get("execution_mode", "real")),
                "replica_id": runtime.get("replica_id"),
                "gpu_assignment": list(runtime.get("gpu_assignment", [])),
            },
        )

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def build_real_command(
        self,
        *,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
        inputs: dict[str, Any],
    ) -> list[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement external-process execution."
        )

    def real_command_cwd(self, *, params: dict[str, Any], workdir: str) -> str | None:
        return (
            params.get("repo_dir")
            or self.model_config.weights.get("script_root")
            or self.model_config.weights.get("repo_path")
        )

    def real_command_timeout_sec(self, *, params: dict[str, Any]) -> int | None:
        timeout = params.get("timeout_sec")
        return int(timeout) if timeout is not None else None

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement real in-process video execution yet."
        )

    def _execute_mock(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
    ) -> dict[str, dict[str, Any]]:
        prompt_ids = list(plan.inputs["prompt_ids"])
        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        fps = int(plan.inputs["fps"])
        num_frames = int(plan.inputs["num_frames"])
        guidance_scale = float(plan.inputs["guidance_scale"])
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        batch_id = plan.inputs.get("batch_id")
        duration_sec = compute_duration_sec(num_frames=num_frames, fps=fps)

        outputs: dict[str, dict[str, Any]] = {}
        for batch_index, (prompt_id, prompt) in enumerate(zip(prompt_ids, prompts, strict=True)):
            output_path = Path(workdir) / f"{prompt_id}.mp4"
            effective_seed = seed + batch_index if seed is not None else None
            write_mock_mp4(
                output_path,
                metadata={
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "num_frames": num_frames,
                    "duration_sec": duration_sec,
                    "seed": effective_seed,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "mock": True,
                    "fingerprint": mock_fingerprint(
                        self.model_config.name,
                        prompt_id,
                        prompt,
                        str(effective_seed) if effective_seed is not None else "mock-random",
                    ),
                },
            )
            outputs[prompt_id] = {
                "path": str(output_path),
                "seed": effective_seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": duration_sec,
                "mock": True,
            }
        return outputs


class DiffusersVideoAdapterBase(MockCapableVideoAdapter):
    real_execution_mode = "in_process"
    pipeline_class_name = ""

    def __init__(self, model_config: Any) -> None:
        super().__init__(model_config)
        self._loaded_pipeline: Any | None = None
        self._loaded_torch: Any | None = None
        self._loaded_device: str | None = None

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_pipeline()

    def unload_persistent_worker(self) -> None:
        self._loaded_pipeline = None
        self._loaded_torch = None
        self._loaded_device = None

    def _execute_real(
        self,
        *,
        plan: ExecutionPlan,
        prompts: list[str],
        workdir: str,
    ) -> dict[str, dict[str, Any]]:
        from diffusers.utils import export_to_video

        width = int(plan.inputs["width"])
        height = int(plan.inputs["height"])
        fps = int(plan.inputs["fps"])
        num_frames = int(plan.inputs["num_frames"])
        seed = int(plan.inputs["seed"]) if plan.inputs.get("seed") not in (None, "") else None
        num_inference_steps = int(plan.inputs["num_inference_steps"])
        guidance_scale = float(plan.inputs["guidance_scale"])
        negative_prompts = [str(value) for value in plan.inputs.get("negative_prompts", [""])]
        prompt_ids = [str(value) for value in plan.inputs["prompt_ids"]]
        pipe, torch, device = self._get_or_load_pipeline()
        frame_batches = self.generate_frames_batch(
            pipe=pipe,
            plan=plan,
            prompts=prompts,
            negative_prompts=negative_prompts,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            torch=torch,
            device=device,
        )
        if len(frame_batches) != len(prompt_ids):
            raise RuntimeError(
                f"{self.model_config.name} returned {len(frame_batches)} videos for {len(prompt_ids)} prompts."
            )
        outputs: dict[str, dict[str, Any]] = {}
        for batch_index, (prompt_id, frames) in enumerate(zip(prompt_ids, frame_batches, strict=True)):
            output_path = Path(workdir) / f"{prompt_id}.mp4"
            export_to_video(frames, str(output_path), fps=fps)
            outputs[prompt_id] = {
                "path": str(output_path),
                "seed": seed + batch_index if seed is not None else None,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "batch_id": plan.inputs.get("batch_id"),
                "batch_index": batch_index,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": compute_duration_sec(num_frames=num_frames, fps=fps),
                "mock": False,
            }
        return outputs

    def _get_or_load_pipeline(self) -> tuple[Any, Any, str]:
        if self._loaded_pipeline is not None and self._loaded_torch is not None and self._loaded_device:
            return self._loaded_pipeline, self._loaded_torch, self._loaded_device

        import torch
        import diffusers

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipeline_class = getattr(diffusers, self.pipeline_class_name)
        pipe = self.load_pipeline(
            pipeline_class=pipeline_class,
            torch=torch,
            device=device,
            dtype=dtype,
        )
        self._loaded_pipeline = pipe
        self._loaded_torch = torch
        self._loaded_device = device
        return pipe, torch, device

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        pipe = pipeline_class.from_pretrained(model_ref, **load_kwargs)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "to"):
            pipe.to(device)
        return pipe

    def validate_model_reference(self, model_ref: str) -> None:
        validate_local_diffusers_reference(
            model_config=self.model_config,
            model_ref=model_ref,
            required_files=("model_index.json",),
        )

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {}

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: ExecutionPlan,
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
    ) -> list[list[Any]]:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        kwargs = {
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if generator is not None:
            kwargs["generator"] = generator
        if self.capabilities.supports_negative_prompt:
            kwargs["negative_prompt"] = (
                negative_prompts[0] if len(negative_prompts) == 1 else negative_prompts
            )
        kwargs.update(self.build_pipeline_call_kwargs(plan))
        output = pipe(**kwargs)
        frames = getattr(output, "frames", None)
        if not frames:
            raise RuntimeError(f"{self.model_config.name} did not return video frames.")
        return [list(video_frames) for video_frames in frames]


class WanT2VDiffusersAdapter(DiffusersVideoAdapterBase):
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
    pipeline_class_name = "WanPipeline"
    default_width = 1280
    default_height = 720
    default_fps = 16
    default_num_frames = 81
    default_num_inference_steps = 40
    default_guidance_scale = 4.0

    def extra_prepare_inputs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"guidance_scale_2": float(params.get("guidance_scale_2", 3.0))}

    def validate_model_reference(self, model_ref: str) -> None:
        validate_local_diffusers_reference(
            model_config=self.model_config,
            model_ref=model_ref,
            required_files=("model_index.json", "vae/config.json"),
            adapter_specific_hint=(
                "For Wan2.2-T2V-A14B-Diffusers, weights_path/local_path should point to the "
                "local Diffusers weights directory for Wan-AI/Wan2.2-T2V-A14B-Diffusers."
            ),
        )

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        from diffusers import AutoencoderKLWan

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        vae = AutoencoderKLWan.from_pretrained(
            model_ref,
            subfolder="vae",
            torch_dtype=torch.float32,
            **({"cache_dir": cache_dir} if cache_dir else {}),
        )
        pipe = pipeline_class.from_pretrained(
            model_ref,
            vae=vae,
            **load_kwargs,
        )
        if hasattr(pipe, "to"):
            pipe.to(device)
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

    def build_pipeline_call_kwargs(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {"guidance_scale_2": float(plan.inputs.get("guidance_scale_2", 3.0))}

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: ExecutionPlan,
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
    ) -> list[list[Any]]:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        output = pipe(
            prompt=prompts[0] if len(prompts) == 1 else prompts,
            negative_prompt=negative_prompts[0] if len(negative_prompts) == 1 else negative_prompts,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=float(plan.inputs.get("guidance_scale_2", 3.0)),
            num_inference_steps=num_inference_steps,
            **({"generator": generator} if generator is not None else {}),
        )
        frames = getattr(output, "frames", None)
        if not frames:
            raise RuntimeError(f"{self.model_config.name} did not return video frames.")
        return [list(video_frames) for video_frames in frames]


class HunyuanVideo15Adapter(DiffusersVideoAdapterBase):
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
    pipeline_class_name = "HunyuanVideo15Pipeline"
    default_width = 1280
    default_height = 720
    default_fps = 24
    default_num_frames = 121
    default_num_inference_steps = 50
    default_guidance_scale = 4.0

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        model_ref = (
            self.model_config.weights.get("local_path")
            or self.model_config.weights.get("weights_path")
            or self.model_config.weights.get("diffusers_repo")
            or self.model_config.weights["hf_repo"]
        )
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        pipe = pipeline_class.from_pretrained(model_ref, **load_kwargs)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe


class CogVideoX5BAdapter(DiffusersVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=2,
        preferred_batch_size=2,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )
    pipeline_class_name = "CogVideoXPipeline"
    default_width = 720
    default_height = 480
    default_fps = 8
    default_num_frames = 49
    default_num_inference_steps = 50
    default_guidance_scale = 6.0

    def load_pipeline(
        self,
        *,
        pipeline_class: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        cache_dir = resolve_video_cache_dir(self.model_config)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model_ref = resolve_video_model_reference(self.model_config)
        self.validate_model_reference(model_ref)
        pipe = pipeline_class.from_pretrained(model_ref, **load_kwargs)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "to"):
            pipe.to(device)
        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe

    def generate_frames_batch(
        self,
        *,
        pipe: Any,
        plan: ExecutionPlan,
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
    ) -> list[list[Any]]:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = None
        if seed is not None:
            generators = [
                torch.Generator(generator_device).manual_seed(seed + batch_index)
                for batch_index in range(len(prompts))
            ]
            generator = generators[0] if len(generators) == 1 else generators
        output = pipe(
            prompt=prompts[0] if len(prompts) == 1 else prompts,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            **({"generator": generator} if generator is not None else {}),
        )
        frames = getattr(output, "frames", None)
        if not frames:
            raise RuntimeError(f"{self.model_config.name} did not return video frames.")
        return [list(video_frames) for video_frames in frames]


class ExternalProcessVideoAdapterBase(MockCapableVideoAdapter):
    real_execution_mode = "external_process"


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


class LongCatVideoAdapter(ExternalProcessVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video"],
    )
    default_width = 1280
    default_height = 720
    default_fps = 30
    default_num_frames = 121
    default_num_inference_steps = 50
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
        checkpoint_dir = str(
            params.get("checkpoint_dir")
            or self.model_config.weights.get("weights_path")
            or self.model_config.weights.get("local_path")
            or "./weights/LongCat-Video"
        )
        command = [
            "torchrun",
            "run_demo_text_to_video.py",
            f"--checkpoint_dir={checkpoint_dir}",
            "--enable_compile",
        ]
        context_parallel_size = params.get("context_parallel_size")
        if context_parallel_size is not None:
            command.extend(["--context_parallel_size", str(context_parallel_size)])
        return command


class MOVAVideoAdapter(ExternalProcessVideoAdapterBase):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=False,
        max_batch_size=1,
        preferred_batch_size=1,
        supports_negative_prompt=False,
        supports_seed=True,
        output_types=["video"],
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


def resolve_video_dimensions(
    params: dict[str, Any],
    *,
    default_width: int = 1280,
    default_height: int = 720,
) -> tuple[int, int]:
    resolution = params.get("resolution")
    if isinstance(resolution, str) and "x" in resolution:
        left, right = resolution.lower().split("x", maxsplit=1)
        return int(left), int(right)
    width = int(params.get("width", default_width))
    height = int(params.get("height", default_height))
    return width, height


def resolve_video_negative_prompts(
    *,
    prompts: list[str],
    params: dict[str, Any],
    supports_negative_prompt: bool,
) -> list[str]:
    if not supports_negative_prompt:
        return ["" for _ in prompts]
    negative_prompts = list(params.get("negative_prompts", []))
    if negative_prompts and len(negative_prompts) != len(prompts):
        raise ValueError("negative_prompts must match the prompt batch length.")
    if not negative_prompts:
        return ["" for _ in prompts]
    return [str(item) for item in negative_prompts]


def compute_duration_sec(*, num_frames: int, fps: int) -> float:
    safe_fps = max(fps, 1)
    return round(num_frames / safe_fps, 4)


def resolve_video_model_reference(model_config: Any) -> str:
    return str(
        model_config.weights.get("weights_path")
        or model_config.weights.get("local_path")
        or model_config.weights.get("diffusers_repo")
        or model_config.weights.get("hf_repo")
    )


def validate_local_diffusers_reference(
    *,
    model_config: Any,
    model_ref: str,
    required_files: tuple[str, ...],
    adapter_specific_hint: str | None = None,
) -> None:
    configured_field = None
    configured_path = None
    for field in ("weights_path", "local_path"):
        raw_value = model_config.weights.get(field)
        if raw_value not in (None, ""):
            configured_field = field
            configured_path = Path(str(raw_value))
            break

    if configured_path is not None:
        if not configured_path.exists():
            raise RuntimeError(
                f"{model_config.name} configured {configured_field} does not exist: {configured_path}"
            )
        missing = [
            relative_path
            for relative_path in required_files
            if not (configured_path / relative_path).exists()
        ]
        if missing:
            message_lines = [
                f"{model_config.name} local weights path does not look like a Diffusers model directory: {configured_path}",
                f"Configured field: {configured_field}",
                f"Missing required files: {', '.join(missing)}",
            ]
            repo_path = model_config.weights.get("repo_path")
            if repo_path not in (None, ""):
                message_lines.append(f"Configured repo_path: {repo_path}")
            if adapter_specific_hint:
                message_lines.append(adapter_specific_hint)
            raise RuntimeError("\n".join(message_lines))

    candidate_path = Path(model_ref)
    if candidate_path.exists():
        missing = [
            relative_path
            for relative_path in required_files
            if not (candidate_path / relative_path).exists()
        ]
        if missing:
            raise RuntimeError(
                f"{model_config.name} local model reference is missing required Diffusers files "
                f"under {candidate_path}: {', '.join(missing)}"
            )


def resolve_video_cache_dir(model_config: Any) -> str | None:
    cache_dir = model_config.weights.get("hf_cache_dir")
    if cache_dir in (None, ""):
        return None
    return str(cache_dir)


def metadata_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}.metadata.json")


def extract_video_metadata(path: str | Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path)
    sidecar = metadata_sidecar_path(target)
    metadata = dict(fallback or {})
    if sidecar.exists():
        metadata.update(json.loads(sidecar.read_text(encoding="utf-8")))
    metadata.setdefault("format", target.suffix.lstrip(".").lower() or "mp4")
    return metadata


def recover_single_video_output(
    *,
    workdir: str | Path,
    exclude: Path | None = None,
) -> Path | None:
    candidates = [
        path
        for path in Path(workdir).rglob("*.mp4")
        if exclude is None or path != exclude
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def mock_fingerprint(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def write_mock_mp4(path: str | Path, *, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_mock_mp4(mock_fingerprint(json.dumps(metadata, sort_keys=True))))
    metadata_sidecar_path(output_path).write_text(
        json.dumps({**metadata, "format": "mp4"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def build_mock_mp4(fingerprint: str) -> bytes:
    ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    payload = ("MOCK-AIGC-VIDEO:" + fingerprint).encode("utf-8")
    mdat_size = len(payload) + 8
    mdat = mdat_size.to_bytes(4, "big") + b"mdat" + payload
    return ftyp + mdat
