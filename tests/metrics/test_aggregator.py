"""Tests for MetricsAggregator."""

from datetime import UTC, datetime, timedelta

from aiai_core.types import Complexity, CostRecord, TaskMetrics, TaskStatus
from aiai_metrics.aggregator import MetricsAggregator
from aiai_metrics.store import MetricsStore


def make_store() -> MetricsStore:
    return MetricsStore(":memory:")


def populate_store(store: MetricsStore) -> datetime:
    """Insert sample data and return the base time."""
    now = datetime.now(UTC)
    # Use a time safely in the past today to avoid future-timestamp issues
    base = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

    # 3 successful tasks, 1 failed
    for i in range(3):
        store.insert_task(
            TaskMetrics(
                task_id=f"t-{i}",
                complexity=Complexity.MEDIUM,
                status=TaskStatus.SUCCESS,
                start_time=base + timedelta(minutes=i * 10),
                end_time=base + timedelta(minutes=i * 10 + 5),
                total_cost_usd=0.01 * (i + 1),
                total_tokens=100 * (i + 1),
                model_calls=i + 1,
            )
        )

    store.insert_task(
        TaskMetrics(
            task_id="t-fail",
            complexity=Complexity.COMPLEX,
            status=TaskStatus.FAILED,
            start_time=base + timedelta(minutes=30),
            end_time=base + timedelta(minutes=35),
            total_cost_usd=0.05,
            total_tokens=500,
            model_calls=2,
            error="timeout",
        )
    )

    # Cost records for different models
    store.insert_cost(
        CostRecord(
            timestamp=base,
            model="gpt-4o",
            tier="balanced",
            task_id="t-0",
            input_tokens=50,
            output_tokens=50,
            cost_usd=0.01,
        )
    )
    store.insert_cost(
        CostRecord(
            timestamp=base + timedelta(minutes=10),
            model="gpt-4o-mini",
            tier="fast",
            task_id="t-1",
            input_tokens=100,
            output_tokens=100,
            cost_usd=0.005,
        )
    )
    store.insert_cost(
        CostRecord(
            timestamp=base + timedelta(minutes=20),
            model="gpt-4o",
            tier="balanced",
            task_id="t-2",
            input_tokens=150,
            output_tokens=150,
            cost_usd=0.02,
        )
    )
    store.insert_cost(
        CostRecord(
            timestamp=base + timedelta(minutes=30),
            model="claude-3",
            tier="powerful",
            task_id="t-fail",
            input_tokens=200,
            output_tokens=300,
            cost_usd=0.05,
        )
    )

    return base


def test_daily_summary() -> None:
    store = make_store()
    base = populate_store(store)
    agg = MetricsAggregator(store)

    summary = agg.daily_summary(base)
    assert summary.total_tasks == 4
    assert summary.successful_tasks == 3
    assert summary.failed_tasks == 1
    assert summary.success_rate == 0.75
    assert summary.total_cost_usd > 0
    assert summary.total_tokens > 0
    assert summary.date == base.strftime("%Y-%m-%d")
    store.close()


def test_daily_summary_empty() -> None:
    store = make_store()
    agg = MetricsAggregator(store)
    yesterday = datetime.now(UTC) - timedelta(days=10)
    summary = agg.daily_summary(yesterday)
    assert summary.total_tasks == 0
    assert summary.success_rate == 0.0
    store.close()


def test_model_performance() -> None:
    store = make_store()
    base = populate_store(store)
    agg = MetricsAggregator(store)

    perfs = agg.model_performance(since=base - timedelta(hours=1))
    assert len(perfs) == 3  # gpt-4o, gpt-4o-mini, claude-3

    models = {p.model: p for p in perfs}
    assert "gpt-4o" in models
    assert models["gpt-4o"].call_count == 2
    assert models["gpt-4o"].total_cost == 0.03
    assert "gpt-4o-mini" in models
    assert models["gpt-4o-mini"].call_count == 1
    assert "claude-3" in models
    store.close()


def test_model_performance_empty() -> None:
    store = make_store()
    agg = MetricsAggregator(store)
    perfs = agg.model_performance(since=datetime.now(UTC))
    assert perfs == []
    store.close()


def test_cost_by_tier() -> None:
    store = make_store()
    base = populate_store(store)
    agg = MetricsAggregator(store)

    tiers = agg.cost_by_tier(since=base - timedelta(hours=1))
    assert "balanced" in tiers
    assert "fast" in tiers
    assert "powerful" in tiers
    assert tiers["balanced"] == 0.03  # 0.01 + 0.02
    assert tiers["fast"] == 0.005
    assert tiers["powerful"] == 0.05
    store.close()


def test_cost_by_tier_empty() -> None:
    store = make_store()
    agg = MetricsAggregator(store)
    tiers = agg.cost_by_tier(since=datetime.now(UTC))
    assert tiers == {}
    store.close()
