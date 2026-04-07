from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import ExecutionPlan, ExecutionResult, ProgressCallback
from whitzard.adapters.texts.base import BaseTextGenerationAdapter


class GptOssTextAdapter(BaseTextGenerationAdapter):
    _pipeline = None
    _torch = None
    _model_device = "cpu"

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_pipeline()

    def unload_persistent_worker(self) -> None:
        self._pipeline = None
        self._torch = None
        self._model_device = "cpu"

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, Any],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        del workdir
        pipe, _torch = self._get_or_load_pipeline()
        prompt_ids = list(plan.inputs.get("prompt_ids", []))
        system_prompt = str(
            params.get(
                "system_prompt",
                self.model_config.generation_defaults.get("system_prompt", ""),
            )
        ).strip()

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "preparing_batch",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                    "message": "building gpt-oss chat inputs",
                }
            )

        generation_kwargs = {
            "max_new_tokens": int(
                params.get(
                    "max_new_tokens",
                    self.model_config.generation_defaults.get("max_new_tokens", 256),
                )
            ),
        }
        if "temperature" in params or "temperature" in self.model_config.generation_defaults:
            generation_kwargs["temperature"] = float(
                params.get(
                    "temperature",
                    self.model_config.generation_defaults.get("temperature", 0.7),
                )
            )
        if "top_p" in params or "top_p" in self.model_config.generation_defaults:
            generation_kwargs["top_p"] = float(
                params.get(
                    "top_p",
                    self.model_config.generation_defaults.get("top_p", 0.95),
                )
            )
        if "do_sample" in params or "do_sample" in self.model_config.generation_defaults:
            generation_kwargs["do_sample"] = bool(
                params.get(
                    "do_sample",
                    self.model_config.generation_defaults.get("do_sample", True),
                )
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "generating",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                }
            )

        outputs: dict[str, str] = {}
        for prompt_id, prompt in zip(prompt_ids, prompts, strict=True):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            pipeline_outputs = pipe(messages, **generation_kwargs)
            outputs[prompt_id] = self._extract_pipeline_text(pipeline_outputs).strip()

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "completed",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                }
            )

        return ExecutionResult(
            exit_code=0,
            logs=f"gpt-oss text generation on {self._model_device}",
            outputs=outputs,
        )

    def _get_or_load_pipeline(self):
        if self._pipeline is not None and self._torch is not None:
            return self._pipeline, self._torch

        import torch  # type: ignore
        from transformers import pipeline  # type: ignore

        model_ref = self._resolve_model_reference()
        pipe = pipeline(
            "text-generation",
            model=model_ref,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._pipeline = pipe
        self._torch = torch
        self._model_device = self._resolve_device_from_pipeline(pipe)
        return pipe, torch

    def _resolve_model_reference(self) -> str:
        for key in ("weights_path", "local_path", "hf_repo"):
            value = self.model_config.weights.get(key)
            if value not in (None, ""):
                return str(Path(str(value)).expanduser()) if key != "hf_repo" else str(value)
        raise RuntimeError(
            f"{self.model_config.name} requires one of weights_path, local_path, or hf_repo."
        )

    def _resolve_device_from_pipeline(self, pipe) -> str:
        model = getattr(pipe, "model", None)
        if model is not None and hasattr(model, "device"):
            return str(model.device)
        return "cpu"

    def _extract_pipeline_text(self, pipeline_outputs: Any) -> str:
        if not isinstance(pipeline_outputs, list) or not pipeline_outputs:
            return str(pipeline_outputs)
        first = pipeline_outputs[0]
        if not isinstance(first, dict):
            return str(first)
        generated_text = first.get("generated_text")
        if isinstance(generated_text, list) and generated_text:
            last = generated_text[-1]
            if isinstance(last, dict):
                value = last.get("content") or last.get("text")
                if value not in (None, ""):
                    return str(value)
            return str(last)
        if generated_text not in (None, ""):
            return str(generated_text)
        for key in ("text", "generated_token_ids"):
            value = first.get(key)
            if value not in (None, ""):
                return str(value)
        return str(first)
