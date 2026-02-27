"""Tests for CostTracker and budget enforcement."""

from __future__ import annotations

import io
import json

import pytest
from aiai_core.config import CostConfig
from aiai_core.logging import JSONLLogger
from aiai_core.types import ModelTier, RouteResponse
from aiai_router.cost import BudgetExceededError, CostTracker


def _make_response(cost_usd: float = 0.01, model: str = "test/model") -> RouteResponse:
    return RouteResponse(
        content="response",
        model=model,
        tier=ModelTier.FAST,
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost_usd,
        latency_ms=200.0,
    )


class TestCostTracker:
    def test_record_returns_cost_record(self) -> None:
        stream = io.StringIO()
        logger = JSONLLogger(stream=stream)
        config = CostConfig(daily_budget_usd=50.0, warn_threshold_usd=1.0)
        tracker = CostTracker(config, logger=logger)

        response = _make_response(cost_usd=0.05, model="test/fast-model")
        record = tracker.record(response, task_id="task-1")

        assert record.model == "test/fast-model"
        assert record.tier == "fast"
        assert record.task_id == "task-1"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost_usd == 0.05
        assert record.daily_total_usd == 0.05

    def test_daily_total_accumulates(self) -> None:
        logger = JSONLLogger(stream=io.StringIO())
        config = CostConfig(daily_budget_usd=50.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=1.0))
        tracker.record(_make_response(cost_usd=2.0))
        tracker.record(_make_response(cost_usd=0.5))

        assert tracker.daily_total() == pytest.approx(3.5)

    def test_budget_exceeded_raises(self) -> None:
        logger = JSONLLogger(stream=io.StringIO())
        config = CostConfig(daily_budget_usd=1.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=0.5))

        with pytest.raises(BudgetExceededError, match="Daily budget exceeded"):
            tracker.record(_make_response(cost_usd=0.6))

    def test_budget_remaining(self) -> None:
        logger = JSONLLogger(stream=io.StringIO())
        config = CostConfig(daily_budget_usd=10.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=3.0))
        assert tracker.budget_remaining() == pytest.approx(7.0)

    def test_warn_threshold_logs_warning(self) -> None:
        stream = io.StringIO()
        logger = JSONLLogger(stream=stream)
        config = CostConfig(daily_budget_usd=50.0, warn_threshold_usd=0.5)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=0.6))

        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split("\n") if line.strip()]
        events = [line["event"] for line in lines]
        assert "cost.high_request" in events

    def test_no_warn_below_threshold(self) -> None:
        stream = io.StringIO()
        logger = JSONLLogger(stream=stream)
        config = CostConfig(daily_budget_usd=50.0, warn_threshold_usd=1.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=0.5))

        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split("\n") if line.strip()]
        events = [line["event"] for line in lines]
        assert "cost.high_request" not in events

    def test_jsonl_logging(self) -> None:
        stream = io.StringIO()
        logger = JSONLLogger(stream=stream)
        config = CostConfig(daily_budget_usd=50.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=0.1), task_id="task-42")

        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split("\n") if line.strip()]
        cost_line = next(entry for entry in lines if entry["event"] == "cost.record")
        assert cost_line["cost_usd"] == 0.1
        assert cost_line["task_id"] == "task-42"
        assert "ts" in cost_line

    def test_date_reset(self) -> None:
        logger = JSONLLogger(stream=io.StringIO())
        config = CostConfig(daily_budget_usd=50.0)
        tracker = CostTracker(config, logger=logger)

        tracker.record(_make_response(cost_usd=5.0))
        assert tracker.daily_total() == pytest.approx(5.0)

        # Simulate date change by modifying internal state
        from datetime import date, timedelta

        tracker._current_date = date.today() - timedelta(days=1)

        assert tracker.daily_total() == pytest.approx(0.0)
