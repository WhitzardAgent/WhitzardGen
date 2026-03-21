"""Text-generation adapter implementations."""

from aigc.adapters.texts.base import BaseTextGenerationAdapter
from aigc.adapters.texts.local_transformers import LocalTransformersTextAdapter
from aigc.adapters.texts.qwen3 import Qwen3TextAdapter

__all__ = [
    "BaseTextGenerationAdapter",
    "LocalTransformersTextAdapter",
    "Qwen3TextAdapter",
]
