from __future__ import annotations

import json
from pathlib import Path

from aigc.prompts.models import PromptRecord


def write_prompt_records_jsonl(*, prompts: list[PromptRecord], output_path: Path | None) -> Path:
    if output_path is None:
        output_path = Path.cwd() / ".codex_benchmark_requests.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "language": prompt.language,
                        "metadata": prompt.metadata,
                        "parameters": prompt.parameters,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path
