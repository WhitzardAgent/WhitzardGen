from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO


LEDGER_FILENAME = "samples.jsonl"


@dataclass(slots=True)
class SampleLedgerRecord:
    timestamp: str
    run_id: str
    task_id: str
    model_name: str
    prompt_id: str
    prompt: str
    status: str
    artifact_type: str | None
    artifact_path: str | None
    error_message: str | None
    replica_id: int | None = None
    batch_id: str | None = None
    batch_index: int | None = None
    execution_mode: str | None = None
    negative_prompt: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunLedgerWriter:
    def __init__(self, run_root: Path, run_id: str) -> None:
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.ledger_path = self.run_root / LEDGER_FILENAME
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._byte_file: BinaryIO | None = None

    def open(self) -> "RunLedgerWriter":
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._byte_file = self.ledger_path.open("ab")
        self._file = open(self._byte_file.fileno(), mode="a", encoding="utf-8", closefd=False)
        return self

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                finally:
                    self._file = None
            if self._byte_file is not None:
                try:
                    self._byte_file.close()
                finally:
                    self._byte_file = None

    def __enter__(self) -> "RunLedgerWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def append_success(
        self,
        *,
        task_id: str,
        model_name: str,
        prompt_id: str,
        prompt: str,
        artifact_type: str,
        artifact_path: str,
        replica_id: int | None = None,
        batch_id: str | None = None,
        batch_index: int | None = None,
        execution_mode: str | None = None,
        negative_prompt: str | None = None,
        language: str | None = None,
    ) -> None:
        record = SampleLedgerRecord(
            timestamp=datetime.now(UTC).isoformat(),
            run_id=self.run_id,
            task_id=task_id,
            model_name=model_name,
            prompt_id=prompt_id,
            prompt=prompt,
            status="success",
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            error_message=None,
            replica_id=replica_id,
            batch_id=batch_id,
            batch_index=batch_index,
            execution_mode=execution_mode,
            negative_prompt=negative_prompt,
            language=language,
        )
        self._write_record(record)

    def append_failure(
        self,
        *,
        task_id: str,
        model_name: str,
        prompt_id: str,
        prompt: str,
        error_message: str,
        artifact_type: str | None = None,
        artifact_path: str | None = None,
        replica_id: int | None = None,
        batch_id: str | None = None,
        batch_index: int | None = None,
        execution_mode: str | None = None,
        negative_prompt: str | None = None,
        language: str | None = None,
    ) -> None:
        record = SampleLedgerRecord(
            timestamp=datetime.now(UTC).isoformat(),
            run_id=self.run_id,
            task_id=task_id,
            model_name=model_name,
            prompt_id=prompt_id,
            prompt=prompt,
            status="failed",
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            error_message=error_message,
            replica_id=replica_id,
            batch_id=batch_id,
            batch_index=batch_index,
            execution_mode=execution_mode,
            negative_prompt=negative_prompt,
            language=language,
        )
        self._write_record(record)

    def append_from_task_result(
        self,
        *,
        task_id: str,
        model_name: str,
        prompts: list[dict[str, Any]],
        batch_items: list[dict[str, Any]],
        execution_mode: str | None = None,
        replica_id: int | None = None,
        batch_id: str | None = None,
    ) -> None:
        prompt_lookup = {prompt["prompt_id"]: prompt for prompt in prompts}
        for batch_item in batch_items:
            prompt_id = batch_item.get("prompt_id")
            if not prompt_id:
                continue
            prompt_data = prompt_lookup.get(prompt_id, {})
            prompt_text = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt")
            language = prompt_data.get("language")
            status = batch_item.get("status", "unknown")
            batch_metadata = dict(batch_item.get("metadata", {}))
            item_batch_id = batch_metadata.get("batch_id", batch_id)
            item_batch_index = batch_metadata.get("batch_index")
            item_replica_id = batch_metadata.get("replica_id", replica_id)

            if status == "success":
                artifacts = batch_item.get("artifacts", [])
                if artifacts:
                    for artifact in artifacts:
                        self.append_success(
                            task_id=task_id,
                            model_name=model_name,
                            prompt_id=prompt_id,
                            prompt=prompt_text,
                            artifact_type=artifact.get("type"),
                            artifact_path=artifact.get("path"),
                            replica_id=item_replica_id,
                            batch_id=item_batch_id,
                            batch_index=item_batch_index,
                            execution_mode=execution_mode,
                            negative_prompt=negative_prompt,
                            language=language,
                        )
                else:
                    self.append_success(
                        task_id=task_id,
                        model_name=model_name,
                        prompt_id=prompt_id,
                        prompt=prompt_text,
                        artifact_type=None,
                        artifact_path=None,
                        replica_id=item_replica_id,
                        batch_id=item_batch_id,
                        batch_index=item_batch_index,
                        execution_mode=execution_mode,
                        negative_prompt=negative_prompt,
                        language=language,
                    )
            else:
                error_message = batch_item.get("error") or batch_item.get("error_message") or "Unknown error"
                self.append_failure(
                    task_id=task_id,
                    model_name=model_name,
                    prompt_id=prompt_id,
                    prompt=prompt_text,
                    error_message=error_message,
                    artifact_type=None,
                    artifact_path=None,
                    replica_id=item_replica_id,
                    batch_id=item_batch_id,
                    batch_index=item_batch_index,
                    execution_mode=execution_mode,
                    negative_prompt=negative_prompt,
                    language=language,
                )

    def _write_record(self, record: SampleLedgerRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            if self._file is None:
                raise RuntimeError("Ledger writer is not open")
            self._file.write(line)
            self._file.flush()
            if self._byte_file is not None:
                self._byte_file.flush()


def load_ledger_records(run_root: Path) -> list[dict[str, Any]]:
    ledger_path = Path(run_root) / LEDGER_FILENAME
    if not ledger_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
