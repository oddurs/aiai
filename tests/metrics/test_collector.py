"""Tests for MetricsCollector context manager."""

from datetime import UTC

import pytest
from aiai_core.types import Complexity, TaskStatus
from aiai_metrics.collector import MetricsCollector
from aiai_metrics.store import MetricsStore


def make_store() -> MetricsStore:
    return MetricsStore(":memory:")


@pytest.mark.asyncio
async def test_collector_success() -> None:
    store = make_store()
    async with MetricsCollector(store, "t-1", Complexity.MEDIUM) as mc:
        mc.record_model_call(cost_usd=0.01, tokens=500, model="gpt-4o")
        mc.record_model_call(cost_usd=0.02, tokens=800, model="gpt-4o")

    assert mc.metrics.status == TaskStatus.SUCCESS
    assert mc.metrics.model_calls == 2
    assert mc.metrics.total_cost_usd == pytest.approx(0.03)
    assert mc.metrics.total_tokens == 1300
    assert mc.metrics.end_time is not None
    assert mc.metrics.error is None

    # Verify persisted to store
    from datetime import datetime

    tasks = store.query_tasks(since=datetime.min.replace(tzinfo=UTC))
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "t-1"
    assert tasks[0]["status"] == "success"
    store.close()


@pytest.mark.asyncio
async def test_collector_failure() -> None:
    store = make_store()
    with pytest.raises(ValueError, match="test error"):
        async with MetricsCollector(store, "t-2", Complexity.COMPLEX) as mc:
            mc.record_model_call(cost_usd=0.05, tokens=1000, model="claude")
            raise ValueError("test error")

    assert mc.metrics.status == TaskStatus.FAILED
    assert mc.metrics.error == "test error"
    assert mc.metrics.model_calls == 1

    from datetime import datetime

    tasks = store.query_tasks(since=datetime.min.replace(tzinfo=UTC))
    assert len(tasks) == 1
    assert tasks[0]["status"] == "failed"
    store.close()


@pytest.mark.asyncio
async def test_collector_records_cost_entries() -> None:
    store = make_store()
    async with MetricsCollector(store, "t-3", Complexity.SIMPLE) as mc:
        mc.record_model_call(
            cost_usd=0.01,
            tokens=100,
            model="gpt-4o-mini",
            tier="fast",
            input_tokens=60,
            output_tokens=40,
        )

    from datetime import datetime

    costs = store.query_costs(since=datetime.min.replace(tzinfo=UTC))
    assert len(costs) == 1
    assert costs[0]["model"] == "gpt-4o-mini"
    assert costs[0]["tier"] == "fast"
    assert costs[0]["input_tokens"] == 60
    assert costs[0]["output_tokens"] == 40
    store.close()


@pytest.mark.asyncio
async def test_collector_retry() -> None:
    store = make_store()
    async with MetricsCollector(store, "t-4", Complexity.MEDIUM) as mc:
        mc.record_retry()
        mc.record_retry()
        mc.record_model_call(cost_usd=0.01, tokens=100, model="gpt-4o")

    assert mc.metrics.retries == 2
    assert mc.metrics.model_calls == 1
    store.close()


@pytest.mark.asyncio
async def test_collector_timing() -> None:
    store = make_store()
    async with MetricsCollector(store, "t-5", Complexity.TRIVIAL) as mc:
        pass  # no-op task

    assert mc.metrics.start_time is not None
    assert mc.metrics.end_time is not None
    assert mc.metrics.end_time >= mc.metrics.start_time
    duration = mc.metrics.duration_ms
    assert duration is not None
    assert duration >= 0
    store.close()
