import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from aigc.registry import load_registry


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


class _FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generation_kwargs = None

    def eval(self):
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
