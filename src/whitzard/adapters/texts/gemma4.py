from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import ExecutionPlan, ExecutionResult, ProgressCallback
from whitzard.adapters.texts.base import BaseTextGenerationAdapter


class Gemma4TextAdapter(BaseTextGenerationAdapter):
    _processor = None
    _model = None
    _torch = None
    _model_device = "cpu"

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_model()

    def unload_persistent_worker(self) -> None:
        self._processor = None
        self._model = None
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
        processor, model, torch = self._get_or_load_model()
        prompt_ids = list(plan.inputs.get("prompt_ids", []))
        system_prompt = str(
            params.get(
                "system_prompt",
                self.model_config.generation_defaults.get(
                    "system_prompt",
                    "You are a helpful assistant.",
                ),
            )
        ).strip()
        enable_thinking = bool(
            params.get(
                "enable_thinking",
                self.model_config.generation_defaults.get("enable_thinking", False),
            )
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "preparing_batch",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                    "message": "building Gemma-4 chat inputs",
                }
            )

        generation_kwargs = {
            "max_new_tokens": int(
                params.get(
                    "max_new_tokens",
                    self.model_config.generation_defaults.get("max_new_tokens", 1024),
                )
            ),
            "do_sample": bool(
                params.get(
                    "do_sample",
                    self.model_config.generation_defaults.get("do_sample", True),
                )
            ),
            "temperature": float(
                params.get(
                    "temperature",
                    self.model_config.generation_defaults.get("temperature", 0.7),
                )
            ),
            "top_p": float(
                params.get(
                    "top_p",
                    self.model_config.generation_defaults.get("top_p", 0.95),
                )
            ),
        }

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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = processor(text=text, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs, model)
            input_ids = self._extract_input_ids(inputs)
            input_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids[0])
            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs)
            output_ids = generated[0][input_len:]
            response = processor.decode(output_ids, skip_special_tokens=False)
            outputs[prompt_id] = self._parse_response(processor=processor, response=response)

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
            logs=f"gemma4 text generation on {self._model_device}",
            outputs=outputs,
        )

    def _get_or_load_model(self):
        if self._processor is not None and self._model is not None and self._torch is not None:
            return self._processor, self._model, self._torch

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        model_ref = self._resolve_model_reference()
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._model_device = self._resolve_model_device(model)
        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch

    def _resolve_model_reference(self) -> str:
        for key in ("weights_path", "local_path", "hf_repo"):
            value = self.model_config.weights.get(key)
            if value not in (None, ""):
                return str(Path(str(value)).expanduser()) if key != "hf_repo" else str(value)
        raise RuntimeError(
            f"{self.model_config.name} requires one of weights_path, local_path, or hf_repo."
        )

    def _resolve_model_device(self, model) -> str:
        if hasattr(model, "device"):
            return str(model.device)
        try:
            first_parameter = next(model.parameters())
        except StopIteration:
            return "cpu"
        return str(first_parameter.device)

    def _move_inputs_to_device(self, inputs, model):
        if hasattr(inputs, "to"):
            return inputs.to(model.device)
        if isinstance(inputs, dict):
            return {key: value.to(model.device) for key, value in inputs.items()}
        return inputs

    def _extract_input_ids(self, inputs):
        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is None and isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Gemma-4 processor inputs are missing input_ids.")
        return input_ids

    def _parse_response(self, *, processor, response: str) -> str:
        parsed = processor.parse_response(response) if hasattr(processor, "parse_response") else None
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("text", "content", "response", "output"):
                value = parsed.get(key)
                if value not in (None, ""):
                    return str(value).strip()
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                value = first.get("content") or first.get("text")
                if value not in (None, ""):
                    return str(value).strip()
            return str(first).strip()
        return str(response).strip()
