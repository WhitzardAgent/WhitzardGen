from __future__ import annotations

import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import IO

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")


def current_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_log_line(message: str, *, timestamp: str | None = None) -> str:
    ts = timestamp or current_timestamp()
    return f"{ts} {message}"


def print_log_line(message: str, *, stream: IO[str] | None = None) -> None:
    target = stream or sys.stderr
    print(format_log_line(message), file=target, flush=True)


class RunLogger:
    """Small timestamped logger for run-scoped file and console output."""

    def __init__(
        self,
        *,
        log_path: str | Path,
        console_stream: IO[str] | None = None,
        console_enabled: bool = False,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._console_stream = console_stream or sys.stderr
        self._console_enabled = console_enabled
        self._lock = threading.Lock()
        self._handle = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def log(
        self,
        message: str,
        *,
        to_console: bool = False,
        already_timestamped: bool = False,
    ) -> None:
        if not message:
            return
        lines = [line for line in message.splitlines() if line.strip()]
        if not lines:
            return
        with self._lock:
            for line in lines:
                rendered = line if already_timestamped or _TIMESTAMP_RE.match(line) else format_log_line(line)
                self._handle.write(rendered + "\n")
                if self._console_enabled or to_console:
                    print(rendered, file=self._console_stream, flush=True)
            self._handle.flush()

