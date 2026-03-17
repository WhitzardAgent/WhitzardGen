from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aigc.runtime.payloads import TaskPayload
from aigc.runtime.worker import build_adapter, execute_task_payload
from aigc.utils.runtime_logging import print_log_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aigc-persistent-worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--execution-mode", choices=("mock", "real"), default="real")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--gpu-assignment", default="")
    parser.add_argument("--registry-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _registry, _model, adapter = build_adapter(
        args.model_name,
        registry_path=args.registry_file,
    )
    gpu_assignment = _parse_gpu_assignment(args.gpu_assignment)

    _log(args.model_name, args.replica_id, gpu_assignment, "starting persistent worker")
    if args.execution_mode == "real":
        _log(args.model_name, args.replica_id, gpu_assignment, "loading model...")
        start = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            adapter.load_for_persistent_worker()
        elapsed = time.monotonic() - start
        _log(
            args.model_name,
            args.replica_id,
            gpu_assignment,
            f"model loaded successfully in {elapsed:.2f}s",
        )
    else:
        _log(args.model_name, args.replica_id, gpu_assignment, "mock execution mode; skipping model load")
    _log(args.model_name, args.replica_id, gpu_assignment, "ready")
    _emit(
        {
            "event": "ready",
            "model_name": args.model_name,
            "replica_id": args.replica_id,
            "gpu_assignment": gpu_assignment,
        }
    )

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            message = json.loads(line)
            command = str(message.get("command", ""))
            if command == "shutdown":
                _log(args.model_name, args.replica_id, gpu_assignment, "shutting down")
                adapter.unload_persistent_worker()
                _emit(
                    {
                        "event": "shutdown",
                        "model_name": args.model_name,
                        "replica_id": args.replica_id,
                    }
                )
                return 0
            if command != "run_task":
                raise RuntimeError(f"Unsupported persistent worker command: {command}")

            task_file = Path(str(message["task_file"]))
            result_file = Path(str(message["result_file"]))
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            _log(
                args.model_name,
                args.replica_id,
                gpu_assignment,
                f"running task {payload.task_id} batch_size={len(payload.prompts)}",
            )
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = execute_task_payload(
                        payload,
                        registry_path=args.registry_file,
                        adapter=adapter,
                    )
                result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                status = str(result["model_result"]["status"])
                artifact_count = _count_artifacts(result)
                _log(
                    args.model_name,
                    args.replica_id,
                    gpu_assignment,
                    f"finished task {payload.task_id} status={status} artifacts={artifact_count}",
                )
                _emit(
                    {
                        "event": "task_complete",
                        "replica_id": args.replica_id,
                        "task_id": payload.task_id,
                        "status": status,
                        "result_file": str(result_file),
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised via subprocess tests
                traceback_text = traceback.format_exc()
                print(traceback_text, file=sys.stderr, flush=True)
                failure = {
                    "task_id": payload.task_id,
                    "model_name": payload.model_name,
                    "execution_mode": payload.execution_mode,
                    "worker_strategy": payload.worker_strategy,
                    "plan": None,
                    "execution_result": {"exit_code": 1, "logs": traceback_text, "outputs": {}},
                    "model_result": {
                        "status": "failed",
                        "batch_items": [],
                        "logs": traceback_text,
                        "metadata": {"error_type": exc.__class__.__name__},
                    },
                }
                result_file.write_text(json.dumps(failure, indent=2), encoding="utf-8")
                _log(
                    args.model_name,
                    args.replica_id,
                    gpu_assignment,
                    f"finished task {payload.task_id} status=failed artifacts=0",
                )
                _emit(
                    {
                        "event": "task_complete",
                        "replica_id": args.replica_id,
                        "task_id": payload.task_id,
                        "status": "failed",
                        "result_file": str(result_file),
                        "error": str(exc),
                    }
                )
    finally:
        adapter.unload_persistent_worker()
    return 0


def _count_artifacts(result: dict) -> int:
    artifact_count = 0
    for item in result.get("model_result", {}).get("batch_items", []):
        if item.get("status") != "success":
            continue
        artifact_count += len(item.get("artifacts", []))
    return artifact_count


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _log(model_name: str, replica_id: int, gpu_assignment: list[int], message: str) -> None:
    print_log_line(
        f"[worker][{model_name}][replica={replica_id}] GPUs={gpu_assignment} {message}",
        stream=sys.stderr,
    )


def _parse_gpu_assignment(raw_value: str) -> list[int]:
    cleaned = raw_value.strip()
    if not cleaned:
        return []
    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
