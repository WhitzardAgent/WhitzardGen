from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_LANGUAGES = {"en", "zh"}


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    language: str
    negative_prompt: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None


class PromptValidationError(ValueError):
    """Raised when prompt inputs fail validation."""
