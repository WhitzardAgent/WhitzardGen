from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        request_api: str,
        default_headers: dict[str, Any] | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.request_api = request_api
        self.default_headers = dict(default_headers or {})
        self.organization = organization
        self.project = project
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(int(max_retries), 0)
        self.initial_backoff_sec = max(float(initial_backoff_sec), 0.0)

    def generate_text(
        self,
        *,
        prompt: str,
        params: dict[str, Any],
        generation_defaults: dict[str, Any],
    ) -> dict[str, Any]:
        if self.request_api == "chat_completions":
            payload = self._build_chat_completions_payload(
                prompt=prompt,
                params=params,
                generation_defaults=generation_defaults,
            )
            response_payload = self._request_json(
                endpoint_path="/chat/completions",
                payload=payload,
            )
            text = self._extract_chat_completions_text(response_payload)
        elif self.request_api == "responses":
            payload = self._build_responses_payload(
                prompt=prompt,
                params=params,
                generation_defaults=generation_defaults,
            )
            response_payload = self._request_json(
                endpoint_path="/responses",
                payload=payload,
            )
            text = self._extract_responses_text(response_payload)
        else:
            raise RuntimeError(f"Unsupported openai-compatible request_api: {self.request_api}")
        return {
            "content": text,
            "response": response_payload,
        }

    def _build_chat_completions_payload(
        self,
        *,
        prompt: str,
        params: dict[str, Any],
        generation_defaults: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        temperature = _resolve_generation_value("temperature", params, generation_defaults)
        top_p = _resolve_generation_value("top_p", params, generation_defaults)
        max_new_tokens = _resolve_generation_value("max_new_tokens", params, generation_defaults)
        seed = _resolve_generation_value("seed", params, generation_defaults)
        if temperature not in (None, ""):
            payload["temperature"] = float(temperature)
        if top_p not in (None, ""):
            payload["top_p"] = float(top_p)
        if max_new_tokens not in (None, ""):
            payload["max_tokens"] = int(max_new_tokens)
        if seed not in (None, ""):
            payload["seed"] = int(seed)
        return payload

    def _build_responses_payload(
        self,
        *,
        prompt: str,
        params: dict[str, Any],
        generation_defaults: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
        }
        temperature = _resolve_generation_value("temperature", params, generation_defaults)
        top_p = _resolve_generation_value("top_p", params, generation_defaults)
        max_new_tokens = _resolve_generation_value("max_new_tokens", params, generation_defaults)
        seed = _resolve_generation_value("seed", params, generation_defaults)
        if temperature not in (None, ""):
            payload["temperature"] = float(temperature)
        if top_p not in (None, ""):
            payload["top_p"] = float(top_p)
        if max_new_tokens not in (None, ""):
            payload["max_output_tokens"] = int(max_new_tokens)
        if seed not in (None, ""):
            payload["seed"] = int(seed)
        return payload

    def _request_json(
        self,
        *,
        endpoint_path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        for key, value in self.default_headers.items():
            if value not in (None, ""):
                headers[str(key)] = str(value)

        url = self.base_url + endpoint_path
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                url,
                data=request_body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                decoded = json.loads(raw) if raw.strip() else {}
                if not isinstance(decoded, dict):
                    raise RuntimeError("Provider response must be a JSON object.")
                return decoded
            except urllib.error.HTTPError as exc:
                last_error = exc
                if not _should_retry_http_status(int(exc.code)) or attempt >= attempts:
                    raise RuntimeError(
                        f"OpenAI-compatible request failed status={exc.code}: {exc}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= attempts:
                    raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
            if attempt < attempts:
                time.sleep(self.initial_backoff_sec * (2 ** (attempt - 1)))
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")

    def _extract_chat_completions_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible chat response is missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenAI-compatible chat choice must be an object.")
        message = dict(first_choice.get("message", {}) or {})
        content = message.get("content")
        text = _extract_text_from_content(content)
        if not text:
            raise RuntimeError("OpenAI-compatible chat response did not contain text content.")
        return text

    def _extract_responses_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if output_text not in (None, ""):
            return str(output_text).strip()
        output_items = payload.get("output")
        if isinstance(output_items, list):
            chunks: list[str] = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                extracted = _extract_text_from_content(content)
                if extracted:
                    chunks.append(extracted)
            text = "\n".join(chunk for chunk in chunks if chunk).strip()
            if text:
                return text
        raise RuntimeError("OpenAI-compatible responses payload did not contain output text.")


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            if item.get("text") not in (None, ""):
                chunks.append(str(item.get("text")).strip())
                continue
            nested_text = item.get("content")
            if nested_text not in (None, "") and isinstance(nested_text, str):
                chunks.append(nested_text.strip())
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _resolve_generation_value(
    key: str,
    params: dict[str, Any],
    generation_defaults: dict[str, Any],
) -> Any:
    if key in params:
        return params[key]
    return generation_defaults.get(key)


def _should_retry_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600
