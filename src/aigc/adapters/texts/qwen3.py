from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc.adapters.base import (
    ArtifactRecord,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
    ProgressCallback,
)
from aigc.adapters.texts.base import BaseTextGenerationAdapter


class Qwen3TextAdapter(BaseTextGenerationAdapter):
    _END_THINKING_TOKEN_ID = 151668

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
        enable_thinking = bool(
            params.get(
                "enable_thinking",
                self.model_config.generation_defaults.get("enable_thinking", True),
            )
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "preparing_batch",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                    "message": "building chat template inputs",
                }
            )

        rendered_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
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
                    self.model_config.generation_defaults.get("max_new_tokens", 2048),
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
                    self.model_config.generation_defaults.get("temperature", 0.6),
                )
            ),
            "top_p": float(
                params.get(
                    "top_p",
                    self.model_config.generation_defaults.get("top_p", 0.95),
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

        outputs: dict[str, dict[str, str]] = {}
        attention_mask = encoded.get("attention_mask")
        for index, prompt_id in enumerate(prompt_ids):
            input_length = int(attention_mask[index].sum().item()) if attention_mask is not None else 0
            output_ids = generated[index][input_length:].tolist()
            thinking_content, content = self._split_thinking_content(
                tokenizer=tokenizer,
                output_ids=output_ids,
            )
            outputs[prompt_id] = {
                "content": content,
                "thinking_content": thinking_content,
                "raw_text": tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            }

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
            logs=f"qwen3 text generation on {self._input_device}",
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
        workdir_path = Path(workdir)
        batch_items: list[BatchItemResult] = []
        status = "success"

        for prompt_id, prompt_text in zip(prompt_ids, prompts, strict=True):
            payload = exec_result.outputs.get(prompt_id, {})
            if isinstance(payload, dict):
                content = str(payload.get("content", "")).strip()
                thinking_content = str(payload.get("thinking_content", "")).strip()
                raw_text = str(payload.get("raw_text", content)).strip()
            else:
                content = str(payload).strip()
                thinking_content = ""
                raw_text = content
            output_path = workdir_path / f"{prompt_id}.txt"
            output_path.write_text(content, encoding="utf-8")
            batch_items.append(
                BatchItemResult(
                    prompt_id=prompt_id,
                    artifacts=[
                        ArtifactRecord(
                            type="text",
                            path=str(output_path),
                            metadata={
                                "format": "txt",
                                "source_prompt": prompt_text,
                                "thinking_content": thinking_content,
                                "raw_text": raw_text,
                            },
                        )
                    ],
                    status="success",
                    metadata={
                        "format": "txt",
                        "thinking_content": thinking_content,
                    },
                )
            )

        if exec_result.exit_code != 0:
            status = "failed"

        return ModelResult(status=status, batch_items=batch_items, logs=exec_result.logs)

    def _get_or_load_model(self):
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_ref = self._resolve_model_reference()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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

    def _split_thinking_content(
        self,
        *,
        tokenizer,
        output_ids: list[int],
    ) -> tuple[str, str]:
        try:
            end_index = len(output_ids) - output_ids[::-1].index(self._END_THINKING_TOKEN_ID)
        except ValueError:
            return "", tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n ")
        thinking_content = tokenizer.decode(
            output_ids[:end_index],
            skip_special_tokens=True,
        ).strip("\n ")
        content = tokenizer.decode(
            output_ids[end_index:],
            skip_special_tokens=True,
        ).strip("\n ")
        if not content:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n ")
        return thinking_content, content
