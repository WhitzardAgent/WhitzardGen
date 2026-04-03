from __future__ import annotations

from pathlib import Path

from whitzard.launching.interfaces import LaunchRequest
from whitzard.launching.models import LaunchPlan, ServiceHealth


class LaunchError(RuntimeError):
    """Raised when launcher planning or health inspection fails."""


def plan_model_launch(
    *,
    requested_models: list[str],
    auto_launch: bool = False,
    config_path: str | Path | None = None,
) -> LaunchPlan:
    notes = []
    if auto_launch:
        notes.append(
            "Auto-launch backend not configured; returning a no-op launch plan placeholder."
        )
    return LaunchPlan(
        requested_models=list(requested_models),
        auto_launch=auto_launch,
        strategy="noop",
        services=[],
        notes=notes,
    )


def inspect_service_health(
    *,
    requested_models: list[str],
    config_path: str | Path | None = None,
) -> list[ServiceHealth]:
    return [
        ServiceHealth(
            service_id=f"service::{model_name}",
            model_name=model_name,
            status="unknown",
            endpoint=None,
            details={"message": "No launcher backend configured."},
        )
        for model_name in requested_models
    ]
