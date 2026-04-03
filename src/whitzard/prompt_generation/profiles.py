from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.prompt_generation.config import (
    DEFAULT_PROMPT_GENERATION_CONFIG_PATH,
    PromptGenerationConfigError,
    load_prompt_generation_catalog,
)

DEFAULT_PROMPT_GENERATION_PROFILES_PATH = DEFAULT_PROMPT_GENERATION_CONFIG_PATH


class PromptGenerationProfileError(PromptGenerationConfigError):
    """Raised when prompt-generation profile config is invalid."""


def load_prompt_generation_profiles(
    path: str | Path = DEFAULT_PROMPT_GENERATION_PROFILES_PATH,
) -> dict[str, dict[str, Any]]:
    return dict(load_prompt_generation_catalog(path)["profiles"])
