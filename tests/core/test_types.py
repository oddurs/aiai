"""Tests for aiai_core.types."""

from datetime import UTC, datetime

from aiai_core.types import (
    Complexity,
    CostRecord,
    ModelTier,
    RouteRequest,
    RouteResponse,
    TaskMetrics,
    TaskStatus,
)


class TestComplexity:
    def test_values(self) -> None:
        assert Complexity.TRIVIAL == "trivial"
        assert Complexity.SIMPLE == "simple"
        assert Complexity.MEDIUM == "medium"
        assert Complexity.COMPLEX == "complex"
        assert Complexity.CRITICAL == "critical"

    def test_from_string(self) -> None:
        assert Complexity("trivial") is Complexity.TRIVIAL
        assert Complexity("critical") is Complexity.CRITICAL


class TestModelTier:
    def test_values(self) -> None:
        assert ModelTier.NANO == "nano"
        assert ModelTier.FAST == "fast"
        assert ModelTier.BALANCED == "balanced"
        assert ModelTier.POWERFUL == "powerful"
        assert ModelTier.MAX == "max"


class TestRouteRequest:
    def test_creation(self) -> None:
        req = RouteRequest(prompt="Hello", complexity=Complexity.SIMPLE)
        assert req.prompt == "Hello"
        assert req.complexity == Complexity.SIMPLE
        assert req.max_tokens is None
        assert req.metadata == {}

    def test_frozen(self) -> None:
        req = RouteRequest(prompt="Hello", complexity=Complexity.SIMPLE)
        try:
            req.prompt = "World"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestRouteResponse:
    def test_creation(self) -> None:
        resp = RouteResponse(
            content="Hi",
            model="anthropic/claude-haiku-4-5",
            tier=ModelTier.NANO,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            latency_ms=150.0,
        )
        assert resp.content == "Hi"
        assert resp.model == "anthropic/claude-haiku-4-5"
        assert resp.tier == ModelTier.NANO


class TestCostRecord:
    def test_to_dict(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        record = CostRecord(
            timestamp=ts,
            model="anthropic/claude-haiku-4-5",
            tier="nano",
            task_id="test-1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            daily_total_usd=0.5,
        )
        d = record.to_dict()
        assert d["model"] == "anthropic/claude-haiku-4-5"
        assert d["cost_usd"] == 0.001
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"


class TestTaskMetrics:
    def test_duration_pending(self) -> None:
        m = TaskMetrics(task_id="t1", complexity=Complexity.SIMPLE)
        assert m.duration_ms is None

    def test_duration_completed(self) -> None:
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=UTC)
        m = TaskMetrics(
            task_id="t1",
            complexity=Complexity.SIMPLE,
            status=TaskStatus.SUCCESS,
            start_time=start,
            end_time=end,
        )
        assert m.duration_ms == 1500.0
