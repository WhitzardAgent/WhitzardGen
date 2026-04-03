from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc.benchmarking.service import load_yaml_file


class ExperimentRecipeError(ValueError):
    """Raised when experiment recipe config is invalid."""


def load_experiment_recipe(path: str | Path) -> dict[str, Any]:
    payload = load_yaml_file(Path(path))
    if not payload:
        raise ExperimentRecipeError(f"Experiment recipe is empty: {path}")
    return payload
