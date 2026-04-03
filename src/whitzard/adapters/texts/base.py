from __future__ import annotations

from pathlib import Path
from typing import Any

from whitzard.adapters.base import (
    AdapterCapabilities,
    ArtifactRecord,
    BaseAdapter,
    BatchItemResult,
    ExecutionPlan,
    ExecutionResult,
    ModelResult,
)


class BaseTextGenerationAdapter(BaseAdapter):
    capabilities = AdapterCapabilities(
        supports_batch_prompts=True,
        max_batch_size=8,
        preferred_batch_size=4,
        supports_negative_prompt=False,
        supports_seed=False,
        output_types=["text"],
        supports_persistent_worker=True,
        preferred_worker_strategy="persistent_worker",
    )

    def prepare(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        params: dict[str, Any],
        workdir: str,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            mode="in_process",
            inputs={
                "prompt_ids": list(prompt_ids),
                "prompt_count": len(prompts),
                "runtime": dict(params.get("_runtime_config", {})),
            },
        )

    def collect(
        self,
        plan: ExecutionPlan,
        exec_result: ExecutionResult,
        prompts: list[str],
        prompt_ids: list[str],
        workdir: str,
    ) -> ModelResult:
        workdir_path = Path(workdir)
        batch_items: list[BatchItemResult] = []
        status = "success"

        for prompt_id, prompt_text in zip(prompt_ids, prompts, strict=True):
            text_output = str(exec_result.outputs.get(prompt_id, "")).strip()
            output_path = workdir_path / f"{prompt_id}.txt"
            output_path.write_text(text_output, encoding="utf-8")
            batch_items.append(
                BatchItemResult(
                    prompt_id=prompt_id,
                    artifacts=[
                        ArtifactRecord(
                            type="text",
                            path=str(output_path),
                            metadata={"format": "txt", "source_prompt": prompt_text},
                        )
                    ],
                    status="success",
                    metadata={"format": "txt"},
                )
            )

        if exec_result.exit_code != 0:
            status = "failed"

        return ModelResult(status=status, batch_items=batch_items, logs=exec_result.logs)
