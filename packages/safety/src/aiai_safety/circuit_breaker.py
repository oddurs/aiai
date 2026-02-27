"""Circuit breaker: stops runaway operations before they drain budget or loop forever."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiai_core.config import SafetyConfig


class CircuitBreakerTrippedError(Exception):
    """Raised when a circuit breaker threshold is exceeded."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class _TaskCounters:
    """Mutable counters for a single task."""

    failure_count: int = 0
    cost_total: float = 0.0
    retry_count: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker with configurable thresholds from SafetyConfig.

    Tracks per-task failures, cost, and retries, plus a global tasks-per-hour rate.
    Raises CircuitBreakerTripped when any threshold is exceeded.
    """

    def __init__(self, config: SafetyConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._tasks: dict[str, _TaskCounters] = defaultdict(_TaskCounters)
        self._task_timestamps: list[float] = []

    def check(
        self,
        task_id: str,
        cost: float = 0.0,
        failed: bool = False,
        is_retry: bool = False,
    ) -> None:
        """Record an event and raise CircuitBreakerTripped if a threshold is breached.

        Args:
            task_id: Identifier for the task being tracked.
            cost: Incremental cost in USD for this event.
            failed: Whether this event represents a failure.
            is_retry: Whether this event represents a retry attempt.
        """
        with self._lock:
            counters = self._tasks[task_id]

            if failed:
                counters.failure_count += 1
            if cost > 0:
                counters.cost_total += cost
            if is_retry:
                counters.retry_count += 1

            # Record task timestamp for rate limiting
            now = time.monotonic()
            self._task_timestamps.append(now)
            # Prune timestamps older than 1 hour
            cutoff = now - 3600
            self._task_timestamps = [t for t in self._task_timestamps if t > cutoff]

            # Check thresholds
            if counters.failure_count > self._config.max_failures_per_task:
                raise CircuitBreakerTrippedError(
                    f"Task {task_id} exceeded max failures: "
                    f"{counters.failure_count} > {self._config.max_failures_per_task}"
                )

            if counters.cost_total > self._config.max_cost_per_task_usd:
                raise CircuitBreakerTrippedError(
                    f"Task {task_id} exceeded max cost: "
                    f"${counters.cost_total:.2f} > ${self._config.max_cost_per_task_usd:.2f}"
                )

            if counters.retry_count > self._config.max_retries_per_task:
                raise CircuitBreakerTrippedError(
                    f"Task {task_id} exceeded max retries: "
                    f"{counters.retry_count} > {self._config.max_retries_per_task}"
                )

            if len(self._task_timestamps) > self._config.max_tasks_per_hour:
                raise CircuitBreakerTrippedError(
                    f"Global rate exceeded max tasks per hour: "
                    f"{len(self._task_timestamps)} > {self._config.max_tasks_per_hour}"
                )

    def reset(self, task_id: str) -> None:
        """Reset all counters for a specific task."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]

    def get_counters(self, task_id: str) -> tuple[int, float, int]:
        """Return (failure_count, cost_total, retry_count) for a task."""
        with self._lock:
            c = self._tasks[task_id]
            return c.failure_count, c.cost_total, c.retry_count
