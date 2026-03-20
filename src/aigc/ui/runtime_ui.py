from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - rich is optional in non-terminal contexts
    from rich.text import Text
except Exception:  # pragma: no cover - graceful fallback when rich is unavailable
    Text = None  # type: ignore[assignment]


@dataclass(slots=True)
class ReplicaStatus:
    replica_id: int
    gpu_assignment: str = "[]"
    state: str = "starting"
    completed_tasks: int = 0
    assigned_tasks: int = 0


class RuntimeTerminalUI:
    """Render a cleaner terminal-friendly operational view for long runs."""

    _TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+")
    _TAG_RE = re.compile(r"^\[(?P<tag>[A-Z]+)(?P<detail>[^\]]*)\]")
    _WORKER_RE = re.compile(
        r"^\[worker\]\[(?P<model>[^\]]+)\]\[replica=(?P<replica>\d+)\] "
        r"GPUs=(?P<gpus>\[[^\]]*\]) (?P<message>.+)$"
    )
    _RUN_MODEL_RE = re.compile(r"^\[run\]\[(?P<model>[^\]]+)\] (?P<message>.+)$")
    _KV_VALUE_PATTERNS = {
        "run_id": re.compile(r"\brun_id=(?P<value>\S+)"),
        "model": re.compile(r"\bmodel=(?P<value>[^|\s]+)"),
        "models": re.compile(r"\bmodels=(?P<value>[^|]+?)(?=\s+\||$)"),
        "profile": re.compile(r"\bprofile=(?P<value>\S+)"),
        "replica": re.compile(r"\breplica=(?P<value>r?\d+)"),
        "gpus": re.compile(r"\bgpus=(?P<value>\[[^\]]*\])"),
        "GPUs": re.compile(r"\bGPUs=(?P<value>\[[^\]]*\])"),
        "prompts": re.compile(r"\bprompts=(?P<value>\d+(?:/\d+)?)"),
        "tasks": re.compile(r"\btasks=(?P<value>\d+)"),
        "success": re.compile(r"\bsuccess=(?P<value>\d+)"),
        "failed": re.compile(r"\bfailed=(?P<value>\d+)"),
        "rate": re.compile(r"\brate=(?P<value>[0-9.]+/min)"),
        "eta": re.compile(r"\beta=(?P<value>\S+)"),
        "artifacts": re.compile(r"\bartifacts=(?P<value>\d+)"),
        "status": re.compile(r"\bstatus=(?P<value>[A-Za-z_]+)"),
        "mode": re.compile(r"\bmode=(?P<value>\S+)"),
    }
    _PATH_VALUE_PATTERN = re.compile(
        r"\b(?:source|out|log|dataset|manifest|failures|path)=(?P<value>\S+)"
    )
    _SUMMARY_STATUS_PATTERN = re.compile(r"^\[SUMMARY\]\s+(?P<value>completed_with_failures|completed|failed)\b")
    _STATE_WORD_PATTERN = re.compile(
        r"\b(?P<value>starting|loading|loaded|ready|running|failed|stopped|shutdown|shutting down)\b"
    )
    _WARN_TOKEN_RE = re.compile(r"\b(?:warning|futurewarning|userwarning|deprecationwarning)\b", re.IGNORECASE)
    _SEMANTIC_STYLES = {
        "timestamp": "dim",
        "run": "bold bright_cyan",
        "stage": "bold blue",
        "sched": "bold magenta",
        "worker": "bold cyan",
        "replica": "bold cyan",
        "throughput": "bold yellow",
        "warn": "bold bright_yellow",
        "error": "bold bright_red",
        "summary_success": "bold green",
        "summary_partial": "bold yellow",
        "summary_failed": "bold bright_red",
        "metric": "bold white",
        "secondary": "dim",
        "state_starting": "bold cyan",
        "state_loading": "bold blue",
        "state_loaded": "bold cyan",
        "state_ready": "bold green",
        "state_running": "bold bright_cyan",
        "state_failed": "bold bright_red",
        "state_stopped": "bold bright_black",
        "state_shutdown": "bold bright_black",
    }
    _TAG_STYLE_MAP = {
        "RUN": "run",
        "STAGE": "stage",
        "SCHED": "sched",
        "WORKER": "worker",
        "TASK": "worker",
        "REPLICA": "replica",
        "THROUGHPUT": "throughput",
        "WARN": "warn",
        "ERROR": "error",
        "SUMMARY": "run",
        "ENV": "stage",
    }

    def __init__(self, *, enable_color: bool = False) -> None:
        self._replicas: dict[str, dict[int, ReplicaStatus]] = {}
        self._enable_color = enable_color and Text is not None

    def render_header(self, header) -> list[str]:
        lines = [
            f"[RUN] {header.run_id} | mode={header.execution_mode}",
            f"[RUN] models={', '.join(header.model_names)}",
            f"[RUN] prompts={header.prompt_count if header.prompt_count is not None else '-'} | source={self._shorten_path(header.prompt_source)}",
            f"[RUN] out={self._shorten_path(header.output_dir)}",
            f"[RUN] log={self._shorten_path(header.running_log_path)}",
        ]
        if getattr(header, "profile_label", None):
            lines.append(f"[RUN] profile={header.profile_label}")
        return lines

    def render_stage_start(self, index: int, total: int, name: str) -> str:
        return f"[STAGE {index}/{total}] {name}..."

    def render_stage_end(self, index: int, total: int, name: str) -> str:
        return f"[STAGE {index}/{total}] {name} - done"

    def render_task_start(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        prompts: int,
        execution_mode: str,
    ) -> str:
        return f"[TASK] {current}/{total} model={model_name} prompts={prompts} mode={execution_mode}"

    def render_task_end(
        self,
        *,
        current: int,
        total: int,
        model_name: str,
        status: str,
        artifacts: int | None = None,
    ) -> str:
        suffix = f" artifacts={artifacts}" if artifacts is not None else ""
        return f"[TASK] {current}/{total} model={model_name} status={status}{suffix}"

    def render_summary(self, summary) -> list[str]:
        if summary.status == "completed":
            title = "[SUMMARY] completed"
        elif summary.status == "completed_with_failures":
            title = "[SUMMARY] completed_with_failures"
        else:
            title = "[SUMMARY] failed"
        lines = [
            f"{title} run_id={summary.run_id}",
            f"[SUMMARY] mode={summary.execution_mode} models={', '.join(summary.model_names)}",
            f"[SUMMARY] prompts={summary.prompt_count} tasks={summary.task_count} success={summary.success_tasks} failed={summary.failed_tasks}",
        ]
        if getattr(summary, "processed_prompt_outputs", None) is not None:
            throughput_suffix = ""
            if getattr(summary, "throughput_per_min", None) is not None:
                throughput_suffix = f" rate={summary.throughput_per_min:.1f}/min"
            lines.append(
                f"[SUMMARY] prompt_outputs={summary.processed_prompt_outputs} failed_outputs={getattr(summary, 'failed_prompt_outputs', 0) or 0}{throughput_suffix}"
            )
        if getattr(summary, "wall_time_sec", None) is not None:
            lines.append(f"[SUMMARY] wall_time_sec={summary.wall_time_sec:.2f}")
        lines.extend(
            [
                f"[SUMMARY] out={self._shorten_path(summary.output_dir)}",
                f"[SUMMARY] dataset={self._shorten_path(summary.dataset_path)}",
                f"[SUMMARY] manifest={self._shorten_path(summary.manifest_path)}",
            ]
        )
        if summary.failures_path:
            lines.append(f"[SUMMARY] failures={self._shorten_path(summary.failures_path)}")
        if summary.running_log_path:
            lines.append(f"[SUMMARY] log={self._shorten_path(summary.running_log_path)}")
        return lines

    def render_event(self, message: str) -> list[str]:
        normalized = self._TIMESTAMP_RE.sub("", message.strip(), count=1)
        if not normalized:
            return []
        if self._is_noise(normalized):
            return []
        if normalized.startswith("[THROUGHPUT]") or normalized.startswith("[REPLICA]"):
            return [normalized]
        if normalized.startswith("[run] ERROR:"):
            return [f"[ERROR] {normalized.removeprefix('[run] ERROR:').strip()}"]
        if normalized.startswith("[run] WARN:") or normalized.startswith("[run] WARNING:"):
            warning_text = normalized.split(":", 1)[1].strip()
            return [f"[WARN] {warning_text}"]
        if normalized.startswith("Ensuring environment for model: "):
            model_name = normalized.removeprefix("Ensuring environment for model: ").strip()
            return [f"[ENV] model={model_name} ensuring environment"]
        worker_match = self._WORKER_RE.match(normalized)
        if worker_match:
            return self._render_worker_event(
                model_name=worker_match.group("model"),
                replica_id=int(worker_match.group("replica")),
                gpu_assignment=worker_match.group("gpus"),
                message=worker_match.group("message"),
            )
        run_match = self._RUN_MODEL_RE.match(normalized)
        if run_match:
            return self._render_run_model_event(
                model_name=run_match.group("model"),
                message=run_match.group("message"),
            )
        if normalized.startswith("[run] "):
            return [f"[RUN] {normalized.removeprefix('[run] ').strip()}"]
        return [normalized]

    def _render_worker_event(
        self,
        *,
        model_name: str,
        replica_id: int,
        gpu_assignment: str,
        message: str,
    ) -> list[str]:
        state = self._replicas.setdefault(model_name, {}).setdefault(
            replica_id,
            ReplicaStatus(replica_id=replica_id),
        )
        state.gpu_assignment = gpu_assignment
        lines: list[str] = []

        if message == "starting persistent worker":
            state.state = "starting"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} gpus={gpu_assignment} starting")
        elif message == "loading model...":
            state.state = "loading"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} gpus={gpu_assignment} loading")
        elif message.startswith("model loaded successfully in "):
            state.state = "loaded"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} {message}")
        elif message == "ready":
            state.state = "ready"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} ready")
        elif message.startswith("running task "):
            state.state = "running"
            lines.append(f"[TASK] model={model_name} replica=r{replica_id} {message}")
        elif message.startswith("finished task "):
            state.state = "ready"
            state.completed_tasks += 1
            lines.append(f"[TASK] model={model_name} replica=r{replica_id} {message}")
        elif message == "shutting down":
            state.state = "stopped"
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} shutting down")
        elif self._looks_like_warning(message):
            lines.append(f"[WARN] model={model_name} replica=r{replica_id} {message}")
        else:
            lines.append(f"[WORKER] model={model_name} replica=r{replica_id} {message}")

        snapshot = self._render_replica_snapshot(model_name)
        if snapshot:
            lines.append(snapshot)
        return lines

    def _render_run_model_event(self, *, model_name: str, message: str) -> list[str]:
        lines: list[str] = []
        if message.startswith("available_gpus="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("gpus_per_replica="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("starting ") and " replicas" in message:
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("replica=") and "assigned" in message:
            replica_id = int(message.split("replica=", 1)[1].split()[0])
            assigned_tasks = int(message.split("assigned ", 1)[1].split()[0])
            gpu_assignment = message.split("GPUs=", 1)[1].strip()
            state = self._replicas.setdefault(model_name, {}).setdefault(
                replica_id,
                ReplicaStatus(replica_id=replica_id),
            )
            state.assigned_tasks = assigned_tasks
            state.gpu_assignment = gpu_assignment
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("warming replica "):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif "ready (" in message and message.startswith("replica="):
            lines.append(f"[SCHED] model={model_name} {message}")
        elif message.startswith("all replicas ready"):
            lines.append(f"[SCHED] model={model_name} {message}")
        else:
            lines.append(f"[RUN] model={model_name} {message}")

        snapshot = self._render_replica_snapshot(model_name)
        if snapshot:
            lines.append(snapshot)
        return lines

    def _render_replica_snapshot(self, model_name: str) -> str | None:
        states = self._replicas.get(model_name, {})
        if not states:
            return None
        if len(states) < 2 and not any(state.assigned_tasks for state in states.values()):
            return None
        parts = []
        for replica_id in sorted(states):
            state = states[replica_id]
            total = state.assigned_tasks if state.assigned_tasks else "?"
            parts.append(
                f"r{replica_id} {state.gpu_assignment} {state.state} {state.completed_tasks}/{total}"
            )
        return f"[REPLICA] model={model_name} " + " | ".join(parts)

    def _is_noise(self, message: str) -> bool:
        noisy_tokens = (
            "Loading pipeline components...",
            "Loading checkpoint shards:",
            "Loading weights:",
            "Downloading shards:",
            "%|",
        )
        return any(token in message for token in noisy_tokens)

    def render_console_line(self, line: str):
        if not self._enable_color or Text is None:
            return line

        timestamp_match = self._TIMESTAMP_RE.match(line)
        timestamp = timestamp_match.group(0).rstrip() if timestamp_match else None
        body = line[timestamp_match.end() :] if timestamp_match else line

        rendered = Text()
        if timestamp:
            rendered.append(timestamp, style=self._style("timestamp"))
            rendered.append(" ")

        styled_body = Text(body)
        self._apply_semantic_styles(styled_body)
        rendered.append(styled_body)
        return rendered

    def _apply_semantic_styles(self, text: Text) -> None:
        plain = text.plain
        tag_match = self._TAG_RE.match(plain)
        if tag_match:
            tag = tag_match.group("tag")
            style_key = self._TAG_STYLE_MAP.get(tag)
            if tag == "SUMMARY":
                style_key = self._summary_style_key(plain)
            if style_key:
                text.stylize(self._style(style_key), tag_match.start(), tag_match.end())

        if plain.startswith("[STAGE"):
            self._stylize_regex(
                text,
                re.compile(r"^\[STAGE [^\]]+\]\s+(?P<value>.+?)(?:\s+-\s+done|\.{3})$"),
                self._style("stage"),
            )
        elif plain.startswith("[THROUGHPUT]"):
            self._stylize_metrics(text)
        elif plain.startswith("[REPLICA]"):
            self._stylize_metrics(text)
            self._stylize_replica_states(text)
        elif plain.startswith("[WORKER]") or plain.startswith("[SCHED]") or plain.startswith("[TASK]"):
            self._stylize_metrics(text)
            self._stylize_replica_states(text)
        elif plain.startswith("[RUN]") or plain.startswith("[ENV]"):
            self._stylize_metrics(text)
        elif plain.startswith("[WARN]"):
            self._stylize_metrics(text)
            text.stylize(self._style("warn"), 0, len(plain))
            tag_match = self._TAG_RE.match(plain)
            if tag_match:
                text.stylize(self._style("warn"), tag_match.start(), tag_match.end())
        elif plain.startswith("[ERROR]"):
            self._stylize_metrics(text)
            text.stylize(self._style("error"), 0, len(plain))
            tag_match = self._TAG_RE.match(plain)
            if tag_match:
                text.stylize(self._style("error"), tag_match.start(), tag_match.end())
        elif plain.startswith("[SUMMARY]"):
            self._stylize_metrics(text)
            self._stylize_regex(text, self._SUMMARY_STATUS_PATTERN, self._style(self._summary_style_key(plain)))

        self._stylize_regex(text, self._PATH_VALUE_PATTERN, self._style("secondary"))

    def _stylize_metrics(self, text: Text) -> None:
        for pattern in self._KV_VALUE_PATTERNS.values():
            self._stylize_regex(text, pattern, self._style("metric"))

    def _stylize_replica_states(self, text: Text) -> None:
        for match in self._STATE_WORD_PATTERN.finditer(text.plain):
            value = match.group("value").lower()
            text.stylize(self._style(self._state_style_key(value)), match.start("value"), match.end("value"))

    def _stylize_regex(self, text: Text, pattern: re.Pattern[str], style: str) -> None:
        for match in pattern.finditer(text.plain):
            try:
                start = match.start("value")
                end = match.end("value")
            except IndexError:
                start = match.start()
                end = match.end()
            text.stylize(style, start, end)

    def _summary_style_key(self, line: str) -> str:
        if "completed_with_failures" in line:
            return "summary_partial"
        if "completed" in line:
            return "summary_success"
        return "summary_failed"

    def _state_style_key(self, state: str) -> str:
        if state in {"starting"}:
            return "state_starting"
        if state in {"loading"}:
            return "state_loading"
        if state in {"loaded"}:
            return "state_loaded"
        if state in {"ready"}:
            return "state_ready"
        if state in {"running"}:
            return "state_running"
        if state in {"failed"}:
            return "state_failed"
        return "state_stopped" if state in {"stopped", "shutdown", "shutting down"} else "metric"

    def _looks_like_warning(self, message: str) -> bool:
        return bool(self._WARN_TOKEN_RE.search(message))

    def _style(self, key: str) -> str:
        return self._SEMANTIC_STYLES[key]

    def _shorten_path(self, value: str, *, max_len: int = 88) -> str:
        if len(value) <= max_len:
            return value
        path = Path(value)
        parts = path.parts
        if len(parts) >= 3:
            shortened = Path(parts[0]) / "..." / Path(*parts[-3:])
            shortened_str = str(shortened)
            if len(shortened_str) <= max_len:
                return shortened_str
        keep = max_len // 2 - 2
        return f"{value[:keep]}...{value[-keep:]}"
