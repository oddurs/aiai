"""Tests for MetricsStore DuckDB operations."""

from datetime import UTC, datetime, timedelta

from aiai_core.types import Complexity, CostRecord, TaskMetrics, TaskStatus
from aiai_metrics.store import MetricsStore


def make_store() -> MetricsStore:
    return MetricsStore(":memory:")


def test_insert_and_query_task() -> None:
    store = make_store()
    now = datetime.now(UTC)
    metrics = TaskMetrics(
        task_id="t-1",
        complexity=Complexity.MEDIUM,
        status=TaskStatus.SUCCESS,
        start_time=now - timedelta(minutes=5),
        end_time=now,
        total_cost_usd=0.05,
        total_tokens=1000,
        model_calls=3,
    )
    store.insert_task(metrics)

    results = store.query_tasks(since=now - timedelta(hours=1))
    assert len(results) == 1
    assert results[0]["task_id"] == "t-1"
    assert results[0]["complexity"] == "medium"
    assert results[0]["status"] == "success"
    assert results[0]["total_cost_usd"] == 0.05
    assert results[0]["total_tokens"] == 1000
    assert results[0]["model_calls"] == 3
    store.close()


def test_insert_and_query_cost() -> None:
    store = make_store()
    now = datetime.now(UTC)
    record = CostRecord(
        timestamp=now,
        model="gpt-4o",
        tier="balanced",
        task_id="t-1",
        input_tokens=200,
        output_tokens=300,
        cost_usd=0.01,
        daily_total_usd=0.05,
    )
    store.insert_cost(record)

    results = store.query_costs(since=now - timedelta(hours=1))
    assert len(results) == 1
    assert results[0]["model"] == "gpt-4o"
    assert results[0]["tier"] == "balanced"
    assert results[0]["cost_usd"] == 0.01
    assert results[0]["input_tokens"] == 200
    assert results[0]["output_tokens"] == 300
    store.close()


def test_query_tasks_time_range() -> None:
    store = make_store()
    now = datetime.now(UTC)

    # Insert tasks at different times
    for i in range(5):
        t = now - timedelta(hours=i)
        store.insert_task(
            TaskMetrics(
                task_id=f"t-{i}",
                complexity=Complexity.SIMPLE,
                status=TaskStatus.SUCCESS,
                start_time=t,
                end_time=t + timedelta(minutes=1),
            )
        )

    # Query last 2 hours only
    results = store.query_tasks(since=now - timedelta(hours=2))
    assert len(results) == 3  # t-0, t-1, t-2
    store.close()


def test_query_empty_store() -> None:
    store = make_store()
    now = datetime.now(UTC)
    assert store.query_tasks(since=now - timedelta(hours=1)) == []
    assert store.query_costs(since=now - timedelta(hours=1)) == []
    store.close()


def test_task_with_error() -> None:
    store = make_store()
    now = datetime.now(UTC)
    metrics = TaskMetrics(
        task_id="t-err",
        complexity=Complexity.COMPLEX,
        status=TaskStatus.FAILED,
        start_time=now,
        end_time=now + timedelta(seconds=10),
        error="something went wrong",
    )
    store.insert_task(metrics)

    results = store.query_tasks(since=now - timedelta(minutes=1))
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error"] == "something went wrong"
    store.close()


def test_retention_policy() -> None:
    store = make_store()
    now = datetime.now(UTC)

    # Insert old and new records
    old_time = now - timedelta(days=100)
    new_time = now - timedelta(hours=1)

    store.insert_task(
        TaskMetrics(
            task_id="old",
            complexity=Complexity.TRIVIAL,
            status=TaskStatus.SUCCESS,
            start_time=old_time,
            end_time=old_time + timedelta(minutes=1),
        )
    )
    store.insert_task(
        TaskMetrics(
            task_id="new",
            complexity=Complexity.TRIVIAL,
            status=TaskStatus.SUCCESS,
            start_time=new_time,
            end_time=new_time + timedelta(minutes=1),
        )
    )
    store.insert_cost(
        CostRecord(timestamp=old_time, model="m1", cost_usd=0.01)
    )
    store.insert_cost(
        CostRecord(timestamp=new_time, model="m2", cost_usd=0.02)
    )

    deleted = store.apply_retention(max_age_days=30)
    assert deleted == 2  # 1 old task + 1 old cost

    tasks = store.query_tasks(since=old_time)
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "new"

    costs = store.query_costs(since=old_time)
    assert len(costs) == 1
    assert costs[0]["model"] == "m2"
    store.close()


def test_multiple_costs_same_task() -> None:
    store = make_store()
    now = datetime.now(UTC)

    for i in range(3):
        store.insert_cost(
            CostRecord(
                timestamp=now + timedelta(seconds=i),
                model=f"model-{i}",
                task_id="t-1",
                cost_usd=0.01 * (i + 1),
            )
        )

    results = store.query_costs(
        since=now - timedelta(minutes=1),
        until=now + timedelta(minutes=1),
    )
    assert len(results) == 3
    assert sum(r["cost_usd"] for r in results) == 0.06
    store.close()
