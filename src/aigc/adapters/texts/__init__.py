"""Text-generation adapter implementations."""

from aigc.adapters.texts.base import BaseTextGenerationAdapter
from aigc.adapters.texts.local_transformers import LocalTransformersTextAdapter
from aigc.adapters.texts.qwen25_instruct import Qwen25InstructTextAdapter
from aigc.adapters.texts.qwen3 import Qwen3TextAdapter
from aigc.adapters.texts.remote_api import BaseRemoteTextAdapter, OpenAICompatibleTextAdapter

__all__ = [
    "BaseTextGenerationAdapter",
    "LocalTransformersTextAdapter",
    "Qwen25InstructTextAdapter",
    "Qwen3TextAdapter",
    "BaseRemoteTextAdapter",
    "OpenAICompatibleTextAdapter",
]
