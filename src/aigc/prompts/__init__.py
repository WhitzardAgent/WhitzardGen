"""Prompt input system."""

from aigc.prompts.loader import (
    generate_prompt_id,
    infer_language,
    load_prompts,
    normalize_text,
    validate_prompts,
)
from aigc.prompts.models import PromptRecord, PromptValidationError, SUPPORTED_LANGUAGES

__all__ = [
    "PromptRecord",
    "PromptValidationError",
    "SUPPORTED_LANGUAGES",
    "generate_prompt_id",
    "infer_language",
    "load_prompts",
    "normalize_text",
    "validate_prompts",
]
