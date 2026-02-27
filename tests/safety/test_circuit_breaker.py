"""Tests for aiai_safety.circuit_breaker."""

from __future__ import annotations

import threading

import pytest
from aiai_core.config import SafetyConfig
from aiai_safety.circuit_breaker import CircuitBreaker, CircuitBreakerTrippedError


def _make_config(**overrides: object) -> SafetyConfig:
    """Create a SafetyConfig with defaults suitable for testing."""
    defaults = {
        "max_failures_per_task": 3,
        "max_cost_per_task_usd": 2.0,
        "max_retries_per_task": 2,
        "max_tasks_per_hour": 50,
    }
    defaults.update(overrides)
    return SafetyConfig(**defaults)  # type: ignore[arg-type]


class TestCircuitBreakerFailures:
    def test_allows_up_to_threshold(self) -> None:
        cb = CircuitBreaker(_make_config(max_failures_per_task=3))
        # 3 failures should be fine (threshold is >3)
        for _ in range(3):
            cb.check("task-1", failed=True)

    def test_trips_on_exceeding_failure_threshold(self) -> None:
        cb = CircuitBreaker(_make_config(max_failures_per_task=3))
        for _ in range(3):
            cb.check("task-1", failed=True)
        with pytest.raises(CircuitBreakerTrippedError, match="max failures"):
            cb.check("task-1", failed=True)

    def test_separate_tasks_have_separate_counters(self) -> None:
        cb = CircuitBreaker(_make_config(max_failures_per_task=2))
        cb.check("task-a", failed=True)
        cb.check("task-a", failed=True)
        # task-b should still be fine
        cb.check("task-b", failed=True)
        cb.check("task-b", failed=True)
        # Now both should trip
        with pytest.raises(CircuitBreakerTrippedError):
            cb.check("task-a", failed=True)
        with pytest.raises(CircuitBreakerTrippedError):
            cb.check("task-b", failed=True)


class TestCircuitBreakerCost:
    def test_allows_up_to_cost_threshold(self) -> None:
        cb = CircuitBreaker(_make_config(max_cost_per_task_usd=5.0))
        cb.check("task-1", cost=2.5)
        cb.check("task-1", cost=2.5)

    def test_trips_on_exceeding_cost(self) -> None:
        cb = CircuitBreaker(_make_config(max_cost_per_task_usd=5.0))
        cb.check("task-1", cost=3.0)
        cb.check("task-1", cost=2.0)
        with pytest.raises(CircuitBreakerTrippedError, match="max cost"):
            cb.check("task-1", cost=0.5)


class TestCircuitBreakerRetries:
    def test_allows_up_to_retry_threshold(self) -> None:
        cb = CircuitBreaker(_make_config(max_retries_per_task=2))
        cb.check("task-1", is_retry=True)
        cb.check("task-1", is_retry=True)

    def test_trips_on_exceeding_retries(self) -> None:
        cb = CircuitBreaker(_make_config(max_retries_per_task=2))
        cb.check("task-1", is_retry=True)
        cb.check("task-1", is_retry=True)
        with pytest.raises(CircuitBreakerTrippedError, match="max retries"):
            cb.check("task-1", is_retry=True)


class TestCircuitBreakerReset:
    def test_reset_clears_counters(self) -> None:
        cb = CircuitBreaker(_make_config(max_failures_per_task=2))
        cb.check("task-1", failed=True)
        cb.check("task-1", failed=True)
        cb.reset("task-1")
        # Should be able to fail again after reset
        cb.check("task-1", failed=True)
        cb.check("task-1", failed=True)

    def test_reset_nonexistent_task_is_noop(self) -> None:
        cb = CircuitBreaker(_make_config())
        cb.reset("nonexistent")  # Should not raise

    def test_get_counters_after_events(self) -> None:
        cb = CircuitBreaker(_make_config())
        cb.check("task-1", cost=1.5, failed=True, is_retry=True)
        failures, cost, retries = cb.get_counters("task-1")
        assert failures == 1
        assert cost == 1.5
        assert retries == 1


class TestCircuitBreakerConcurrency:
    def test_concurrent_access_is_thread_safe(self) -> None:
        cb = CircuitBreaker(_make_config(
            max_failures_per_task=1000,
            max_tasks_per_hour=10000,
        ))
        errors: list[Exception] = []

        def worker(task_id: str) -> None:
            try:
                for _ in range(100):
                    cb.check(task_id, cost=0.001, failed=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"task-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Verify each task got exactly 100 failures
        for i in range(10):
            failures, cost, _ = cb.get_counters(f"task-{i}")
            assert failures == 100


class TestCircuitBreakerTrippedErrorException:
    def test_exception_has_reason(self) -> None:
        exc = CircuitBreakerTrippedError("too many failures")
        assert exc.reason == "too many failures"
        assert str(exc) == "too many failures"
