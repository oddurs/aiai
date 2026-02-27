"""Metrics collector context manager for task execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiai_core.types import Complexity, CostRecord, TaskMetrics, TaskStatus

if TYPE_CHECKING:
    from types import TracebackType

    from aiai_metrics.store import MetricsStore


class MetricsCollector:
    """Async context manager that collects metrics during task execution.

    Usage::

        async with MetricsCollector(store, "task-1", Complexity.MEDIUM) as mc:
            mc.record_model_call(cost_usd=0.01, tokens=500, model="gpt-4o")
            # ... do work ...
    """

    def __init__(
        self,
        store: MetricsStore,
        task_id: str,
        complexity: Complexity,
    ) -> None:
        self._store = store
        self._metrics = TaskMetrics(
            task_id=task_id,
            complexity=complexity,
            status=TaskStatus.RUNNING,
        )

    @property
    def metrics(self) -> TaskMetrics:
        """Current task metrics snapshot."""
        return self._metrics

    def record_model_call(
        self,
        cost_usd: float,
        tokens: int,
        model: str,
        tier: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a single model call, updating accumulators."""
        self._metrics.model_calls += 1
        self._metrics.total_cost_usd += cost_usd
        self._metrics.total_tokens += tokens

        cost_record = CostRecord(
            model=model,
            tier=tier,
            task_id=self._metrics.task_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self._store.insert_cost(cost_record)

    def record_retry(self) -> None:
        """Increment the retry counter."""
        self._metrics.retries += 1

    async def __aenter__(self) -> MetricsCollector:
        self._metrics.start_time = datetime.now(UTC)
        self._metrics.status = TaskStatus.RUNNING
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._metrics.end_time = datetime.now(UTC)
        if exc_type is not None:
            self._metrics.status = TaskStatus.FAILED
            self._metrics.error = str(exc_val)
        else:
            self._metrics.status = TaskStatus.SUCCESS
        self._store.insert_task(self._metrics)
