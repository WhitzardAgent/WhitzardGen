"""Text-generation adapter implementations."""

from whitzard.adapters.texts.base import BaseTextGenerationAdapter
from whitzard.adapters.texts.gemma4 import Gemma4TextAdapter
from whitzard.adapters.texts.glm47_flash import GLM47FlashTextAdapter
from whitzard.adapters.texts.gpt_oss import GptOssTextAdapter
from whitzard.adapters.texts.local_transformers import LocalTransformersTextAdapter
from whitzard.adapters.texts.qwen25_instruct import Qwen25InstructTextAdapter
from whitzard.adapters.texts.qwen3 import Qwen3TextAdapter
from whitzard.adapters.texts.remote_api import BaseRemoteTextAdapter, OpenAICompatibleTextAdapter

__all__ = [
    "BaseTextGenerationAdapter",
    "Gemma4TextAdapter",
    "GLM47FlashTextAdapter",
    "GptOssTextAdapter",
    "LocalTransformersTextAdapter",
    "Qwen25InstructTextAdapter",
    "Qwen3TextAdapter",
    "BaseRemoteTextAdapter",
    "OpenAICompatibleTextAdapter",
]
