import json
import sys
import tempfile
import unittest
import urllib.error
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from whitzard.registry import load_registry
from whitzard.providers.openai_compatible import OpenAICompatibleClient


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class _FakeMaskRow:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def sum(self) -> _FakeScalar:
        return _FakeScalar(sum(self._values))


class _FakeBatchTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def to(self, device: str):
        del device
        return self

    def __iter__(self):
        for row in self._rows:
            yield _FakeOutputRow(row)

    def __getitem__(self, index: int) -> _FakeMaskRow:
        return _FakeMaskRow(self._rows[index])


class _FakeOutputRow(list):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return _FakeOutputRow(result)
        return result

    def tolist(self) -> list[int]:
        return list(self)


class _FakeGenerated:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = [_FakeOutputRow(row) for row in rows]

    def __getitem__(self, index: int) -> _FakeOutputRow:
        return self._rows[index]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.padding_side = "right"

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append(
            {
                "prompt": messages[0]["content"],
                "enable_thinking": kwargs.get("enable_thinking"),
            }
        )
        return f"chat::{messages[0]['content']}"

    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return {
            "input_ids": _FakeBatchTensor([[10, 11], [20, 21]]),
            "attention_mask": _FakeBatchTensor([[1, 1], [1, 1]]),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        key = tuple(ids)
        mapping = {
            (101, 151668): "draft reasoning",
            (201, 202): "final answer",
            (101, 151668, 201, 202): "draft reasoning final answer",
            (301, 302): "plain completion",
        }
        return mapping.get(key, "decoded text")

    def batch_decode(self, batch_ids, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


class _FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generation_kwargs = None

    def eval(self):
        return self

    def to(self, device: str):
        self.device = device
        return self

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return _FakeGenerated(
            [
                [10, 11, 101, 151668, 201, 202],
                [20, 21, 301, 302],
            ]
        )


class _FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


class TextAdapterTests(unittest.TestCase):
    def test_qwen3_adapter_uses_chat_template_and_splits_thinking_content(self) -> None:
        registry = load_registry()
        adapter = registry.instantiate_adapter("Qwen3-32B")
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        adapter._get_or_load_model = lambda: (tokenizer, model, _FakeTorch())  # type: ignore[attr-defined]
        adapter._input_device = "cuda:0"  # type: ignore[attr-defined]

        plan = adapter.prepare(
            prompts=["Tell me about reefs", "Give me a short intro"],
            prompt_ids=["p001", "p002"],
            params={"_runtime_config": {}},
            workdir=tempfile.mkdtemp(),
        )
        result = adapter.execute(
            plan=plan,
            prompts=["Tell me about reefs", "Give me a short intro"],
            params={"enable_thinking": True, "max_new_tokens": 128},
            workdir=tempfile.mkdtemp(),
        )

        self.assertEqual(tokenizer.calls[0]["enable_thinking"], True)
        self.assertEqual(result.outputs["p001"]["thinking_content"], "draft reasoning")
        self.assertEqual(result.outputs["p001"]["content"], "final answer")
        self.assertEqual(result.outputs["p002"]["thinking_content"], "")
        self.assertEqual(result.outputs["p002"]["content"], "plain completion")
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 128)

        workdir = Path(tempfile.mkdtemp())
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["Tell me about reefs", "Give me a short intro"],
            prompt_ids=["p001", "p002"],
            workdir=str(workdir),
        )
        self.assertEqual(collected.status, "success")
        self.assertEqual((workdir / "p001.txt").read_text(encoding="utf-8"), "final answer")
        self.assertEqual(
            collected.batch_items[0].artifacts[0].metadata["thinking_content"],
            "draft reasoning",
        )

    def test_qwen25_instruct_adapter_uses_system_and_user_chat_template(self) -> None:
        from whitzard.adapters.texts.qwen25_instruct import Qwen25InstructTextAdapter

        registry = load_registry()
        adapter = Qwen25InstructTextAdapter(
            model_config=registry.get_model("Qwen2.5-32B-Instruct")
        )
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        adapter._get_or_load_model = lambda: (tokenizer, model, _FakeTorch())  # type: ignore[attr-defined]
        adapter._input_device = "cuda:0"  # type: ignore[attr-defined]

        captured_messages = []

        def fake_apply_chat_template(messages, **kwargs) -> str:
            captured_messages.append({"messages": messages, "kwargs": kwargs})
            return f"chat::{messages[-1]['content']}"

        tokenizer.apply_chat_template = fake_apply_chat_template  # type: ignore[method-assign]

        plan = adapter.prepare(
            prompts=["Give me a short introduction to large language model."],
            prompt_ids=["p001"],
            params={"_runtime_config": {}},
            workdir=tempfile.mkdtemp(),
        )
        result = adapter.execute(
            plan=plan,
            prompts=["Give me a short introduction to large language model."],
            params={"system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            workdir=tempfile.mkdtemp(),
        )

        self.assertEqual(len(captured_messages), 1)
        self.assertEqual(captured_messages[0]["messages"][0]["role"], "system")
        self.assertEqual(captured_messages[0]["messages"][1]["role"], "user")
        self.assertEqual(
            captured_messages[0]["messages"][0]["content"],
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        )
        self.assertEqual(
            captured_messages[0]["messages"][1]["content"],
            "Give me a short introduction to large language model.",
        )
        self.assertTrue(captured_messages[0]["kwargs"]["add_generation_prompt"])
        self.assertIn("p001", result.outputs)

    def test_qwen3_adapter_sets_left_padding_for_decoder_only_batches(self) -> None:
        from whitzard.adapters.texts.qwen3 import Qwen3TextAdapter

        registry = load_registry()
        adapter = Qwen3TextAdapter(model_config=registry.get_model("Qwen3-32B"))

        tokenizer = _FakeTokenizer()

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_ref: str, trust_remote_code: bool = True):
                del model_ref, trust_remote_code
                return tokenizer

        class _FakeAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_ref: str, **kwargs):
                del model_ref, kwargs
                return _FakeModel()

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _FakeTorchModule:
            cuda = _FakeCuda()

        fake_transformers = type(
            "_FakeTransformersModule",
            (),
            {
                "AutoTokenizer": _FakeAutoTokenizer,
                "AutoModelForCausalLM": _FakeAutoModelForCausalLM,
            },
        )()

        with patch.dict(
            sys.modules,
            {
                "torch": _FakeTorchModule(),
                "transformers": fake_transformers,
            },
        ):
            adapter._get_or_load_model()  # type: ignore[attr-defined]

        self.assertEqual(tokenizer.padding_side, "left")

    def test_local_transformers_adapter_sets_left_padding_for_decoder_only_batches(self) -> None:
        from whitzard.adapters.texts.local_transformers import LocalTransformersTextAdapter

        registry = load_registry()
        adapter = LocalTransformersTextAdapter(model_config=registry.get_model("Qwen3-32B"))

        tokenizer = _FakeTokenizer()

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_ref: str, trust_remote_code: bool = True):
                del model_ref, trust_remote_code
                return tokenizer

        class _FakeAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_ref: str, **kwargs):
                del model_ref, kwargs
                return _FakeModel()

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _FakeTorchModule:
            cuda = _FakeCuda()
            bfloat16 = "bfloat16"
            float32 = "float32"

        fake_transformers = type(
            "_FakeTransformersModule",
            (),
            {
                "AutoTokenizer": _FakeAutoTokenizer,
                "AutoModelForCausalLM": _FakeAutoModelForCausalLM,
            },
        )()

        with patch.dict(
            sys.modules,
            {
                "torch": _FakeTorchModule(),
                "transformers": fake_transformers,
            },
        ):
            adapter._get_or_load_model()  # type: ignore[attr-defined]

        self.assertEqual(tokenizer.padding_side, "left")

    def test_openai_compatible_client_builds_chat_completions_request(self) -> None:
        captured_request = {}

        class _FakeResponse(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        def fake_urlopen(request, timeout):
            captured_request["url"] = request.full_url
            captured_request["timeout"] = timeout
            captured_request["body"] = json.loads(request.data.decode("utf-8"))
            captured_request["headers"] = dict(request.header_items())
            return _FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "hello from chat completions",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        client = OpenAICompatibleClient(
            base_url="https://example.test/v1",
            api_key="test-key",
            model_name="example-chat-model",
            request_api="chat_completions",
            timeout_sec=12.5,
        )

        with patch("whitzard.providers.openai_compatible.urllib.request.urlopen", side_effect=fake_urlopen):
            payload = client.generate_text(
                prompt="Tell me about reefs",
                params={"temperature": 0.2, "max_new_tokens": 128, "seed": 7},
                generation_defaults={"top_p": 0.95},
            )

        self.assertEqual(payload["content"], "hello from chat completions")
        self.assertEqual(captured_request["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured_request["timeout"], 12.5)
        self.assertEqual(captured_request["body"]["model"], "example-chat-model")
        self.assertEqual(
            captured_request["body"]["messages"],
            [{"role": "user", "content": "Tell me about reefs"}],
        )
        self.assertEqual(captured_request["body"]["temperature"], 0.2)
        self.assertEqual(captured_request["body"]["top_p"], 0.95)
        self.assertEqual(captured_request["body"]["max_tokens"], 128)
        self.assertEqual(captured_request["body"]["seed"], 7)

    def test_openai_compatible_client_builds_responses_request(self) -> None:
        captured_request = {}

        class _FakeResponse(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        def fake_urlopen(request, timeout):
            captured_request["url"] = request.full_url
            captured_request["timeout"] = timeout
            captured_request["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(json.dumps({"output_text": "hello from responses"}).encode("utf-8"))

        client = OpenAICompatibleClient(
            base_url="https://example.test/v1",
            api_key="test-key",
            model_name="example-responses-model",
            request_api="responses",
        )

        with patch("whitzard.providers.openai_compatible.urllib.request.urlopen", side_effect=fake_urlopen):
            payload = client.generate_text(
                prompt="Summarize the scene",
                params={"top_p": 0.8},
                generation_defaults={"temperature": 0.4, "max_new_tokens": 64},
            )

        self.assertEqual(payload["content"], "hello from responses")
        self.assertEqual(captured_request["url"], "https://example.test/v1/responses")
        self.assertEqual(captured_request["body"]["model"], "example-responses-model")
        self.assertEqual(captured_request["body"]["input"], "Summarize the scene")
        self.assertEqual(captured_request["body"]["temperature"], 0.4)
        self.assertEqual(captured_request["body"]["top_p"], 0.8)
        self.assertEqual(captured_request["body"]["max_output_tokens"], 64)

    def test_openai_compatible_client_retries_retryable_http_failures(self) -> None:
        attempts = {"count": 0}

        class _FakeResponse(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        def fake_urlopen(request, timeout):
            del request, timeout
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib.error.HTTPError(
                    url="https://example.test/v1/chat/completions",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=None,
                    fp=None,
                )
            return _FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "retry succeeded",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        client = OpenAICompatibleClient(
            base_url="https://example.test/v1",
            api_key="test-key",
            model_name="example-chat-model",
            request_api="chat_completions",
            max_retries=2,
            initial_backoff_sec=0.0,
        )

        with patch("whitzard.providers.openai_compatible.urllib.request.urlopen", side_effect=fake_urlopen):
            payload = client.generate_text(
                prompt="Tell me about reefs",
                params={},
                generation_defaults={},
            )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(payload["content"], "retry succeeded")

    def test_openai_compatible_adapter_executes_real_path_and_collects_text_artifacts(self) -> None:
        registry = load_registry()
        adapter = registry.instantiate_adapter("OpenAI-Compatible-Chat")

        class _FakeClient:
            def generate_text(self, *, prompt, params, generation_defaults):
                del params, generation_defaults
                return {
                    "content": f"remote::{prompt}",
                    "response": {"provider": "openai_compatible"},
                }

        adapter._build_client = lambda: _FakeClient()  # type: ignore[method-assign]
        adapter._client = None  # type: ignore[attr-defined]

        plan = adapter.prepare(
            prompts=["A calm lake", "A busy city street"],
            prompt_ids=["p001", "p002"],
            params={"_runtime_config": {}},
            workdir=tempfile.mkdtemp(),
        )
        result = adapter.execute(
            plan=plan,
            prompts=["A calm lake", "A busy city street"],
            params={"temperature": 0.1},
            workdir=tempfile.mkdtemp(),
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.outputs["p001"]["content"], "remote::A calm lake")

        workdir = Path(tempfile.mkdtemp())
        collected = adapter.collect(
            plan=plan,
            exec_result=result,
            prompts=["A calm lake", "A busy city street"],
            prompt_ids=["p001", "p002"],
            workdir=str(workdir),
        )
        self.assertEqual(collected.status, "success")
        self.assertEqual((workdir / "p001.txt").read_text(encoding="utf-8"), "remote::A calm lake")
        self.assertEqual(
            collected.batch_items[0].artifacts[0].metadata["provider_type"],
            "openai_compatible",
        )
