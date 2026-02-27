"""Tests for aiai_core.logging."""

import io
import json
import tempfile
from pathlib import Path

from aiai_core.logging import JSONLLogger


class TestJSONLLogger:
    def test_log_to_stream(self) -> None:
        buf = io.StringIO()
        logger = JSONLLogger(stream=buf)
        logger.info("test.event", key="value")
        line = buf.getvalue().strip()
        entry = json.loads(line)
        assert entry["level"] == "info"
        assert entry["event"] == "test.event"
        assert entry["key"] == "value"
        assert "ts" in entry

    def test_log_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(path=path)
            logger.info("file.event", count=42)
            logger.close()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["event"] == "file.event"
            assert entry["count"] == 42

    def test_multiple_levels(self) -> None:
        buf = io.StringIO()
        logger = JSONLLogger(stream=buf)
        logger.debug("d")
        logger.info("i")
        logger.warn("w")
        logger.error("e")
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 4
        levels = [json.loads(line)["level"] for line in lines]
        assert levels == ["debug", "info", "warn", "error"]

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "test.jsonl"
            logger = JSONLLogger(path=path)
            logger.info("nested")
            logger.close()
            assert path.exists()
