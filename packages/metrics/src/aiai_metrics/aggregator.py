"""Metrics aggregation and rollup queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiai_metrics.store import MetricsStore


@dataclass(frozen=True)
class DailySummary:
    """Aggregated metrics for a single day."""

    date: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_cost_usd: float
    total_tokens: int
    success_rate: float


@dataclass(frozen=True)
class ModelPerformance:
    """Per-model performance statistics."""

    model: str
    call_count: int
    total_cost: float
    avg_latency: float
    error_rate: float


class MetricsAggregator:
    """Computes rollups and aggregations from the metrics store."""

    def __init__(self, store: MetricsStore) -> None:
        self._store = store

    def daily_summary(self, date: datetime) -> DailySummary:
        """Compute aggregated metrics for a single day.

        Args:
            date: Any datetime on the target day (timezone-aware).
        """
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        tasks = self._store.query_tasks(since=day_start, until=day_end)
        total = len(tasks)
        successful = sum(1 for t in tasks if t["status"] == "success")
        failed = sum(1 for t in tasks if t["status"] == "failed")
        cost_values = [float(t["total_cost_usd"] or 0) for t in tasks]  # type: ignore[arg-type]
        cost = sum(cost_values)
        tokens = sum(int(t["total_tokens"] or 0) for t in tasks)  # type: ignore[call-overload]
        rate = successful / total if total > 0 else 0.0

        return DailySummary(
            date=day_start.strftime("%Y-%m-%d"),
            total_tasks=total,
            successful_tasks=successful,
            failed_tasks=failed,
            total_cost_usd=round(cost, 6),
            total_tokens=tokens,
            success_rate=round(rate, 4),
        )

    def model_performance(self, since: datetime) -> list[ModelPerformance]:
        """Compute per-model performance statistics.

        Args:
            since: Start of the time window (timezone-aware).
        """
        costs = self._store.query_costs(since=since)
        if not costs:
            return []

        # Group by model
        models: dict[str, list[dict[str, object]]] = {}
        for c in costs:
            model = str(c["model"])
            models.setdefault(model, []).append(c)

        # Also get task data for error rates
        tasks = self._store.query_tasks(since=since)
        task_errors: dict[str, int] = {}
        for t in tasks:
            # Approximate: attribute errors to the task's model calls
            status = t["status"]
            if status == "failed":
                # Increment error count for all models used by failed tasks
                task_id = str(t["task_id"])
                for c in costs:
                    if str(c["task_id"]) == task_id:
                        m = str(c["model"])
                        task_errors[m] = task_errors.get(m, 0) + 1

        result = []
        for model, records in models.items():
            call_count = len(records)
            cost_vals = [float(r["cost_usd"] or 0) for r in records]  # type: ignore[arg-type]
            total_cost = sum(cost_vals)
            errors = task_errors.get(model, 0)
            error_rate = errors / call_count if call_count > 0 else 0.0

            result.append(
                ModelPerformance(
                    model=model,
                    call_count=call_count,
                    total_cost=round(total_cost, 6),
                    avg_latency=0.0,  # Latency not tracked in cost_records
                    error_rate=round(error_rate, 4),
                )
            )

        return sorted(result, key=lambda mp: mp.total_cost, reverse=True)

    def cost_by_tier(self, since: datetime) -> dict[str, float]:
        """Compute total cost broken down by model tier.

        Args:
            since: Start of the time window (timezone-aware).
        """
        costs = self._store.query_costs(since=since)
        tiers: dict[str, float] = {}
        for c in costs:
            tier = str(c["tier"]) if c["tier"] else "unknown"
            tiers[tier] = tiers.get(tier, 0) + float(c["cost_usd"] or 0)  # type: ignore[arg-type]

        return {k: round(v, 6) for k, v in sorted(tiers.items())}
