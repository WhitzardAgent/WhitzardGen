from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from whitzard.adapters.base import (
    ArtifactRecord,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
    ProgressCallback,
)
from whitzard.adapters.texts.base import BaseTextGenerationAdapter
from whitzard.providers import OpenAICompatibleClient


class BaseRemoteTextAdapter(BaseTextGenerationAdapter):
    provider_type: str = ""
    _client = None

    def load_for_persistent_worker(self) -> None:
        self._get_or_load_client()

    def unload_persistent_worker(self) -> None:
        self._client = None

    def execute(
        self,
        plan: ExecutionPlan,
        prompts: list[str],
        params: dict[str, Any],
        workdir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        del workdir
        prompt_ids = list(plan.inputs.get("prompt_ids", []))
        runtime = dict(plan.inputs.get("runtime", {}))
        if runtime.get("execution_mode") == "mock":
            return self._execute_mock(prompt_ids=prompt_ids, prompts=prompts)

        client = self._get_or_load_client()
        outputs: dict[str, dict[str, Any]] = {}
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "preparing_batch",
                    "batch_size": len(prompts),
                    "supports_true_progress": False,
                    "message": f"dispatching remote requests via {self.provider_type}",
                }
            )
        for index, (prompt_id, prompt_text) in enumerate(zip(prompt_ids, prompts, strict=True), start=1):
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "generating",
                        "batch_size": len(prompts),
                        "supports_true_progress": False,
                        "message": f"remote request {index}/{len(prompts)}",
                    }
                )
            outputs[prompt_id] = client.generate_text(
                prompt=prompt_text,
                params=params,
                generation_defaults=self.model_config.generation_defaults,
            )
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
            logs=(
                f"remote provider={self.provider_type} "
                f"request_api={self.model_config.provider.get('request_api')} "
                f"model={self.model_config.provider.get('model_name')} "
                f"requests={len(prompts)}"
            ),
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
                response_payload = payload.get("response")
            else:
                content = str(payload).strip()
                response_payload = None
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
                                "provider_type": self.provider_type,
                                "request_api": self.model_config.provider.get("request_api"),
                                "response": response_payload,
                            },
                        )
                    ],
                    status="success",
                    metadata={
                        "format": "txt",
                        "provider_type": self.provider_type,
                    },
                )
            )
        if exec_result.exit_code != 0:
            status = "failed"
        return ModelResult(status=status, batch_items=batch_items, logs=exec_result.logs)

    def _get_or_load_client(self):
        if self._client is not None:
            return self._client
        self._client = self._build_client()
        return self._client

    def _build_client(self):
        raise NotImplementedError

    def _execute_mock(
        self,
        *,
        prompt_ids: list[str],
        prompts: list[str],
    ) -> ExecutionResult:
        outputs: dict[str, dict[str, str]] = {}
        for prompt_id, prompt_text in zip(prompt_ids, prompts, strict=True):
            structured_payload = self._build_mock_structured_payload(prompt_text)
            if structured_payload is not None:
                content = json.dumps(structured_payload, ensure_ascii=False)
            else:
                original_prompt = self._extract_original_prompt(prompt_text)
                content = f"{original_prompt} [rewritten]"
            outputs[prompt_id] = {
                "content": content,
                "response": {"mock": True, "provider_type": self.provider_type},
            }
        return ExecutionResult(
            exit_code=0,
            logs=f"mock remote provider={self.provider_type} requests={len(prompts)}",
            outputs=outputs,
        )

    def _extract_original_prompt(self, instruction: str) -> str:
        marker = "Original prompt:"
        if marker not in instruction:
            return instruction.strip()
        tail = instruction.split(marker, 1)[1]
        terminator = "\n\nFew-shot examples:"
        if terminator in tail:
            tail = tail.split(terminator, 1)[0]
        return tail.strip() or instruction.strip()

    def _build_mock_structured_payload(self, instruction: str) -> dict[str, Any] | None:
        match = re.search(
            r"Return JSON only with keys:\s*([a-zA-Z0-9_,\s-]+)\.",
            instruction,
        )
        if match is None:
            return None
        keys = [item.strip() for item in match.group(1).split(",") if item.strip()]
        payload: dict[str, Any] = {}
        for key in keys:
            normalized = key.lower()
            if "label" in normalized:
                payload[key] = ["mock_label"]
            elif "confidence" in normalized or "score" in normalized:
                payload[key] = 0.9
            else:
                payload[key] = f"mock_{key}"
        return payload


class OpenAICompatibleTextAdapter(BaseRemoteTextAdapter):
    provider_type = "openai_compatible"

    def _build_client(self):
        provider = dict(self.model_config.provider)
        provider_type = str(provider.get("type") or "").strip()
        if provider_type != self.provider_type:
            raise RuntimeError(
                f"{self.model_config.name} requires provider.type={self.provider_type}."
            )
        base_url = str(provider.get("base_url") or "").strip()
        api_key_env = str(provider.get("api_key_env") or "").strip()
        model_name = str(provider.get("model_name") or "").strip()
        request_api = str(provider.get("request_api") or "").strip()
        if not base_url:
            raise RuntimeError(f"{self.model_config.name} is missing provider.base_url.")
        if not api_key_env:
            raise RuntimeError(f"{self.model_config.name} is missing provider.api_key_env.")
        if not model_name:
            raise RuntimeError(f"{self.model_config.name} is missing provider.model_name.")
        api_key = os.environ.get(api_key_env)
        if api_key in (None, ""):
            raise RuntimeError(
                f"{self.model_config.name} requires env var {api_key_env} for remote API access."
            )
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            request_api=request_api or "chat_completions",
            default_headers=dict(provider.get("default_headers", {}) or {}),
            organization=str(provider.get("organization") or "").strip() or None,
            project=str(provider.get("project") or "").strip() or None,
            timeout_sec=float(provider.get("timeout_sec", 60.0) or 60.0),
            max_retries=int(provider.get("max_retries", 3) or 3),
            initial_backoff_sec=float(provider.get("initial_backoff_sec", 1.0) or 1.0),
        )
