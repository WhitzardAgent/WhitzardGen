from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import (
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
    ProgressCallback,
)
from whitzard.adapters.videos.common import (
    compute_duration_sec,
    extract_video_metadata,
    metadata_sidecar_path,
    mock_fingerprint,
    recover_single_video_output,
    resolve_video_dimensions,
    resolve_video_negative_prompts,
    write_mock_mp4,
)


class BaseVideoGenerationAdapter(BaseAdapter):
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
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        runtime = dict(plan.inputs.get("runtime", {}))
        if runtime.get("execution_mode") == "mock":
            outputs = self._execute_mock(plan=plan, prompts=prompts, workdir=workdir)
            return ExecutionResult(
                exit_code=0,
                logs=f"{self.model_config.name} mock video generation completed.",
                outputs=outputs,
            )
        outputs = self._execute_real(
            plan=plan,
            prompts=prompts,
            workdir=workdir,
            progress_callback=progress_callback,
        )
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
        progress_callback: ProgressCallback | None = None,
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


class ExternalProcessVideoAdapterBase(BaseVideoGenerationAdapter):
    real_execution_mode = "external_process"
