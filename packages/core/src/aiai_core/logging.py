"""Structured JSONL logging for aiai."""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class JSONLLogger:
    """Append-only JSONL logger for structured event logging.

    Thread-safe. Each log entry is a single JSON line with a timestamp,
    level, event name, and arbitrary structured data.
    """

    def __init__(self, path: str | Path | None = None, stream: TextIO | None = None) -> None:
        """Initialize logger with file path and/or stream output.

        Args:
            path: File path for JSONL output. Created if it doesn't exist.
            stream: Optional stream (e.g., sys.stderr) for additional output.
        """
        self._lock = threading.Lock()
        self._stream = stream
        self._file: TextIO | None = None

        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._file = Path.open(p, "a")  # noqa: SIM115

    def log(
        self,
        level: str,
        event: str,
        **data: Any,
    ) -> None:
        """Write a structured log entry.

        Args:
            level: Log level (debug, info, warn, error).
            event: Event name (e.g., "model.request", "cost.record").
            **data: Arbitrary key-value pairs to include.
        """
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "event": event,
            **data,
        }
        line = json.dumps(entry, default=str)

        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
            if self._stream is not None:
                self._stream.write(line + "\n")
                self._stream.flush()

    def info(self, event: str, **data: Any) -> None:
        """Log at info level."""
        self.log("info", event, **data)

    def warn(self, event: str, **data: Any) -> None:
        """Log at warn level."""
        self.log("warn", event, **data)

    def error(self, event: str, **data: Any) -> None:
        """Log at error level."""
        self.log("error", event, **data)

    def debug(self, event: str, **data: Any) -> None:
        """Log at debug level."""
        self.log("debug", event, **data)

    def close(self) -> None:
        """Flush and close the file handle."""
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None


# Module-level default logger
_default_logger: JSONLLogger | None = None
_default_lock = threading.Lock()


def get_logger(name: str | None = None) -> JSONLLogger:
    """Get or create the default logger.

    Uses AIAI_LOG_FILE env var for file path, defaults to logs/aiai.jsonl.
    Logs to stderr in debug mode (AIAI_DEBUG=1).
    """
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            log_file = os.environ.get("AIAI_LOG_FILE", "logs/aiai.jsonl")
            stream = sys.stderr if os.environ.get("AIAI_DEBUG") == "1" else None
            _default_logger = JSONLLogger(path=log_file, stream=stream)
        return _default_logger
