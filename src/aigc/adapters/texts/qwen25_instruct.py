from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc.adapters.base import ExecutionPlan, ExecutionResult, ProgressCallback
from aigc.adapters.texts.base import BaseTextGenerationAdapter


class Qwen25InstructTextAdapter(BaseTextGenerationAdapter):
    _tokenizer = None
    _model = None
    _torch = None
    _input_device = "cpu"

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_model()

    def unload_persistent_worker(self) -> None:
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._input_device = "cpu"

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

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "preparing_batch",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                    "message": "building chat template inputs",
                }
            )

        system_prompt = str(
            params.get(
                "system_prompt",
                self.model_config.generation_defaults.get(
                    "system_prompt",
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                ),
            )
        ).strip()
        rendered_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        encoded = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(self._input_device) for key, value in encoded.items()}

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "generating",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                }
            )

        generation_kwargs = {
            "max_new_tokens": int(
                params.get(
                    "max_new_tokens",
                    self.model_config.generation_defaults.get("max_new_tokens", 512),
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
                    self.model_config.generation_defaults.get("top_p", 0.8),
                )
            ),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if "repetition_penalty" in params or "repetition_penalty" in self.model_config.generation_defaults:
            generation_kwargs["repetition_penalty"] = float(
                params.get(
                    "repetition_penalty",
                    self.model_config.generation_defaults.get("repetition_penalty", 1.0),
                )
            )

        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)

        outputs: dict[str, str] = {}
        attention_mask = encoded.get("attention_mask")
        for index, prompt_id in enumerate(prompt_ids):
            input_length = int(attention_mask[index].sum().item()) if attention_mask is not None else 0
            generated_ids = generated[index][input_length:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            outputs[prompt_id] = completion

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
            logs=f"qwen2.5 instruct text generation on {self._input_device}",
            outputs=outputs,
        )

    def _get_or_load_model(self):
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_ref = self._resolve_model_reference()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        self._input_device = self._resolve_input_device(model)
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

    def _resolve_input_device(self, model) -> str:
        if hasattr(model, "device"):
            return str(model.device)
        try:
            first_parameter = next(model.parameters())
        except StopIteration:
            return "cuda" if self._torch is not None and self._torch.cuda.is_available() else "cpu"
        return str(first_parameter.device)
