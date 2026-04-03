from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from queue import Empty

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from whitzard.runtime.payloads import TaskPayload
from whitzard.runtime.persistent_ipc import (
    PersistentWorkerQueueManager,
    register_client_queues,
)
from whitzard.runtime.progress import TaskProgressReporter
from whitzard.runtime.worker import build_adapter, execute_task_payload
from whitzard.utils.runtime_logging import format_log_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whitzard-persistent-worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--execution-mode", choices=("mock", "real"), default="real")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--gpu-assignment", default="")
    parser.add_argument("--registry-file")
    parser.add_argument("--manager-address", required=True)
    parser.add_argument("--manager-authkey", required=True)
    parser.add_argument("--command-method", required=True)
    parser.add_argument("--event-method", required=True)
    parser.add_argument("--log-method", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_persistent_worker_loop(
        model_name=args.model_name,
        execution_mode=args.execution_mode,
        replica_id=args.replica_id,
        gpu_assignment=_parse_gpu_assignment(args.gpu_assignment),
        registry_path=args.registry_file,
        manager_address=args.manager_address,
        manager_authkey=args.manager_authkey,
        command_method=args.command_method,
        event_method=args.event_method,
        log_method=args.log_method,
    )


def run_persistent_worker_loop(
    *,
    model_name: str,
    execution_mode: str,
    replica_id: int,
    gpu_assignment: list[int],
    registry_path: str | None,
    manager_address: str,
    manager_authkey: str,
    command_method: str,
    event_method: str,
    log_method: str,
) -> int:
    register_client_queues(
        command_method=command_method,
        event_method=event_method,
        log_method=log_method,
    )
    manager = PersistentWorkerQueueManager(
        address=manager_address,
        authkey=bytes.fromhex(manager_authkey),
    )
    last_error: Exception | None = None
    for _attempt in range(30):
        try:
            manager.connect()
            break
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        raise RuntimeError(
            f"Failed to connect to persistent worker queue manager at {manager_address}: {last_error}"
        )
    command_queue = getattr(manager, command_method)()
    event_queue = getattr(manager, event_method)()
    log_queue = getattr(manager, log_method)()

    adapter = None
    current_task_id: str | None = None
    phase = "startup"
    stdio_sink = _QueueTextStream(
        log_queue=log_queue,
        model_name=model_name,
        replica_id=replica_id,
        gpu_assignment=gpu_assignment,
    )
    try:
        _registry, _model, adapter = build_adapter(
            model_name,
            registry_path=registry_path,
        )
        _log(log_queue, model_name, replica_id, gpu_assignment, "starting persistent worker")
        if execution_mode == "real":
            _log(log_queue, model_name, replica_id, gpu_assignment, "loading model...")
            start = time.monotonic()
            with contextlib.redirect_stdout(stdio_sink), contextlib.redirect_stderr(stdio_sink):
                adapter.load_for_persistent_worker()
            elapsed = time.monotonic() - start
            _log(
                log_queue,
                model_name,
                replica_id,
                gpu_assignment,
                f"model loaded successfully in {elapsed:.2f}s",
            )
        else:
            _log(
                log_queue,
                model_name,
                replica_id,
                gpu_assignment,
                "mock execution mode; skipping model load",
            )
        _log(log_queue, model_name, replica_id, gpu_assignment, "ready")
        _emit_event(
            event_queue,
            {
                "event": "ready",
                "model_name": model_name,
                "replica_id": replica_id,
                "gpu_assignment": gpu_assignment,
            },
        )
        phase = "idle"

        while True:
            try:
                message = command_queue.get(timeout=0.5)
            except Empty:
                continue
            command = str(message.get("command", ""))
            if command == "shutdown":
                phase = "shutdown"
                _log(log_queue, model_name, replica_id, gpu_assignment, "shutting down")
                if adapter is not None:
                    adapter.unload_persistent_worker()
                _emit_event(
                    event_queue,
                    {
                        "event": "shutdown",
                        "model_name": model_name,
                        "replica_id": replica_id,
                    },
                )
                return 0
            if command != "run_task":
                raise RuntimeError(f"Unsupported persistent worker command: {command}")

            phase = "task_dispatch"
            task_file = Path(str(message["task_file"]))
            result_file = Path(str(message["result_file"]))
            payload = TaskPayload.from_dict(json.loads(task_file.read_text(encoding="utf-8")))
            current_task_id = payload.task_id
            _emit_event(
                event_queue,
                {
                    "event": "task_started",
                    "model_name": model_name,
                    "replica_id": replica_id,
                    "task_id": payload.task_id,
                },
            )
            _log(
                log_queue,
                model_name,
                replica_id,
                gpu_assignment,
                f"running task {payload.task_id} batch_size={len(payload.prompts)}",
            )
            progress_reporter = TaskProgressReporter(
                model_name=model_name,
                replica_id=replica_id,
                task_id=payload.task_id,
                batch_id=payload.batch_id,
                batch_size=len(payload.prompts),
                emit_event=lambda progress_payload: _emit_event(
                    event_queue,
                    {
                        **progress_payload,
                        "event": "task_progress",
                        "model_name": model_name,
                        "replica_id": replica_id,
                        "task_id": payload.task_id,
                        "batch_id": payload.batch_id,
                        "batch_size": len(payload.prompts),
                    },
                ),
                emit_log=lambda line: _log(
                    log_queue,
                    model_name,
                    replica_id,
                    gpu_assignment,
                    line,
                ),
            )
            progress_reporter.phase("preparing_batch")
            phase = "task_execution"
            try:
                with contextlib.redirect_stdout(stdio_sink), contextlib.redirect_stderr(stdio_sink):
                    progress_reporter.phase("generating")
                    result = execute_task_payload(
                        payload,
                        registry_path=registry_path,
                        adapter=adapter,
                        progress_callback=lambda progress_payload: _handle_progress_payload(
                            reporter=progress_reporter,
                            payload=progress_payload,
                        ),
                    )
                progress_reporter.phase("exporting")
                result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                status = str(result["model_result"]["status"])
                artifact_count = _count_artifacts(result)
                progress_reporter.phase("completed")
                _log(
                    log_queue,
                    model_name,
                    replica_id,
                    gpu_assignment,
                    f"finished task {payload.task_id} status={status} artifacts={artifact_count}",
                )
                _emit_event(
                    event_queue,
                    {
                        "event": "task_complete",
                        "model_name": model_name,
                        "replica_id": replica_id,
                        "task_id": payload.task_id,
                        "status": status,
                        "result_file": str(result_file),
                    },
                )
            except Exception as exc:  # pragma: no cover - exercised through subprocess boundary
                traceback_text = traceback.format_exc()
                stdio_sink.write(traceback_text)
                progress_reporter.phase("failed", message=str(exc))
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
                    log_queue,
                    model_name,
                    replica_id,
                    gpu_assignment,
                    f"finished task {payload.task_id} status=failed artifacts=0",
                )
                _emit_event(
                    event_queue,
                    {
                        "event": "task_complete",
                        "model_name": model_name,
                        "replica_id": replica_id,
                        "task_id": payload.task_id,
                        "status": "failed",
                        "result_file": str(result_file),
                        "error": str(exc),
                    },
                )
            finally:
                current_task_id = None
                phase = "idle"
    except BaseException as exc:  # pragma: no cover - exercised through subprocess boundary
        traceback_text = traceback.format_exc()
        stdio_sink.write(traceback_text)
        _safe_emit_worker_failed(
            event_queue=event_queue,
            model_name=model_name,
            replica_id=replica_id,
            task_id=current_task_id,
            phase=phase,
            error=str(exc),
            traceback_text=traceback_text,
        )
        if adapter is not None:
            with contextlib.suppress(Exception):
                adapter.unload_persistent_worker()
        return 1
    finally:
        stdio_sink.flush()


def _count_artifacts(result: dict) -> int:
    artifact_count = 0
    for item in result.get("model_result", {}).get("batch_items", []):
        if item.get("status") != "success":
            continue
        artifact_count += len(item.get("artifacts", []))
    return artifact_count


def _emit_event(queue_proxy, payload: dict[str, object]) -> None:
    queue_proxy.put(payload)


def _safe_emit_worker_failed(
    *,
    event_queue,
    model_name: str,
    replica_id: int,
    task_id: str | None,
    phase: str,
    error: str,
    traceback_text: str,
) -> None:
    with contextlib.suppress(Exception):
        _emit_event(
            event_queue,
            {
                "event": "worker_failed",
                "model_name": model_name,
                "replica_id": replica_id,
                "task_id": task_id,
                "phase": phase,
                "error": error,
                "traceback": traceback_text,
            },
        )


def _handle_progress_payload(*, reporter: TaskProgressReporter, payload: dict[str, object]) -> None:
    phase = str(payload.get("phase", "generating"))
    current_step = payload.get("current_step")
    total_steps = payload.get("total_steps")
    message = str(payload.get("message", "")).strip() or None
    supports_true_progress = bool(payload.get("supports_true_progress", False))
    if current_step is not None and total_steps is not None:
        reporter.step(
            int(current_step),
            int(total_steps),
            phase=phase,
            message=message,
            supports_true_progress=supports_true_progress,
        )
        return
    reporter.phase(phase, message=message)


def _log(log_queue, model_name: str, replica_id: int, gpu_assignment: list[int], message: str) -> None:
    line = format_log_line(
        f"[worker][{model_name}][replica={replica_id}] GPUs={gpu_assignment} {message}"
    )
    log_queue.put(line)


def _parse_gpu_assignment(raw_value: str) -> list[int]:
    cleaned = raw_value.strip()
    if not cleaned:
        return []
    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


class _QueueTextStream:
    def __init__(self, *, log_queue, model_name: str, replica_id: int, gpu_assignment: list[int]) -> None:
        self.log_queue = log_queue
        self.prefix = f"[worker][{model_name}][replica={replica_id}] GPUs={gpu_assignment}"
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return False

    def _emit(self, line: str) -> None:
        text = line.rstrip()
        if not text:
            return
        self.log_queue.put(format_log_line(f"{self.prefix} {text}"))


if __name__ == "__main__":
    raise SystemExit(main())
