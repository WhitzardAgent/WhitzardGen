from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc.registry.models import ModelInfo
from aigc.runtime.payloads import TaskPayload


def build_dataset_records(
    *,
    run_id: str,
    model: ModelInfo,
    task_payload: TaskPayload,
    task_result: dict[str, Any],
    record_start_index: int = 1,
) -> list[dict[str, Any]]:
    prompt_lookup = {prompt.prompt_id: prompt for prompt in task_payload.prompts}
    records: list[dict[str, Any]] = []
    next_index = record_start_index

    for batch_item in task_result["model_result"]["batch_items"]:
        if batch_item["status"] != "success":
            continue
        prompt = prompt_lookup[batch_item["prompt_id"]]
        batch_metadata = dict(batch_item.get("metadata", {}))
        for artifact in batch_item["artifacts"]:
            records.append(
                {
                    "record_id": f"rec_{next_index:08d}",
                    "run_id": run_id,
                    "task_id": task_payload.task_id,
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.prompt,
                    "negative_prompt": prompt.negative_prompt,
                    "language": prompt.language,
                    "model_name": model.name,
                    "model_version": model.version,
                    "adapter_name": model.adapter,
                    "modality": model.modality,
                    "task_type": model.task_type,
                    "artifact_type": artifact["type"],
                    "artifact_path": artifact["path"],
                    "artifact_metadata": artifact.get("metadata", {}),
                    "generation_params": dict(task_payload.params),
                    "prompt_metadata": dict(prompt.metadata),
                    "execution_metadata": {
                        "status": batch_item["status"],
                        "batch_id": batch_metadata.get("batch_id", task_payload.batch_id),
                        "batch_index": batch_metadata.get("batch_index"),
                        "execution_mode": batch_metadata.get(
                            "execution_mode",
                            task_result.get("execution_mode", task_payload.execution_mode),
                        ),
                        "mock": batch_metadata.get(
                            "mock",
                            (
                                batch_metadata.get(
                                    "execution_mode",
                                    task_result.get("execution_mode", task_payload.execution_mode),
                                )
                                == "mock"
                            ),
                        ),
                    },
                }
            )
            next_index += 1
    return records


def export_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
