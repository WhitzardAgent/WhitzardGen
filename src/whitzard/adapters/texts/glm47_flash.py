from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import ExecutionPlan, ExecutionResult, ProgressCallback
from whitzard.adapters.texts.base import BaseTextGenerationAdapter


class GLM47FlashTextAdapter(BaseTextGenerationAdapter):
    _tokenizer = None
    _model = None
    _torch = None
    _model_device = "cpu"

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_model()

    def unload_persistent_worker(self) -> None:
        self._tokenizer = None
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
        tokenizer, model, torch = self._get_or_load_model()
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
                    "message": "building GLM-4.7-Flash chat inputs",
                }
            )

        generation_kwargs = {
            "max_new_tokens": int(
                params.get(
                    "max_new_tokens",
                    self.model_config.generation_defaults.get("max_new_tokens", 128),
                )
            ),
            "do_sample": bool(
                params.get(
                    "do_sample",
                    self.model_config.generation_defaults.get("do_sample", False),
                )
            ),
        }
        if "temperature" in params or "temperature" in self.model_config.generation_defaults:
            generation_kwargs["temperature"] = float(
                params.get(
                    "temperature",
                    self.model_config.generation_defaults.get("temperature", 1.0),
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
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = self._move_inputs_to_device(inputs, model)
            input_ids = self._extract_input_ids(inputs)
            input_len = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            output_ids = generated_ids[0][input_len:]
            outputs[prompt_id] = tokenizer.decode(output_ids).strip()

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
            logs=f"glm-4.7-flash text generation on {self._model_device}",
            outputs=outputs,
        )

    def _get_or_load_model(self):
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_ref = self._resolve_model_reference()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._model_device = self._resolve_model_device(model)
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        return tokenizer, model, torch

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
            raise RuntimeError("GLM-4.7-Flash tokenizer inputs are missing input_ids.")
        return input_ids
