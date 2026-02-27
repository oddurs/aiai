"""Tests for FastAPI metrics API."""

from datetime import UTC, datetime, timedelta

from aiai_core.types import Complexity, CostRecord, TaskMetrics, TaskStatus
from aiai_metrics.api import create_app
from aiai_metrics.store import MetricsStore
from starlette.testclient import TestClient


def make_client() -> tuple[TestClient, MetricsStore]:
    store = MetricsStore(":memory:")
    app = create_app(store)
    return TestClient(app), store


def test_health() -> None:
    client, store = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    store.close()


def test_ingest_task() -> None:
    client, store = make_client()
    now = datetime.now(UTC)
    payload = {
        "task_id": "t-api-1",
        "complexity": "medium",
        "status": "success",
        "start_time": (now - timedelta(minutes=5)).isoformat(),
        "end_time": now.isoformat(),
        "total_cost_usd": 0.05,
        "total_tokens": 1000,
        "model_calls": 3,
    }
    resp = client.post("/metrics/task", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["task_id"] == "t-api-1"

    # Verify persisted
    tasks = store.query_tasks(since=now - timedelta(hours=1))
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "t-api-1"
    store.close()


def test_ingest_cost() -> None:
    client, store = make_client()
    now = datetime.now(UTC)
    payload = {
        "timestamp": now.isoformat(),
        "model": "gpt-4o",
        "tier": "balanced",
        "task_id": "t-api-1",
        "input_tokens": 200,
        "output_tokens": 300,
        "cost_usd": 0.02,
    }
    resp = client.post("/metrics/cost", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    costs = store.query_costs(since=now - timedelta(hours=1))
    assert len(costs) == 1
    assert costs[0]["model"] == "gpt-4o"
    store.close()


def test_get_summary() -> None:
    client, store = make_client()
    now = datetime.now(UTC)

    # Insert a task today
    store.insert_task(
        TaskMetrics(
            task_id="t-sum",
            complexity=Complexity.SIMPLE,
            status=TaskStatus.SUCCESS,
            start_time=now - timedelta(minutes=10),
            end_time=now,
            total_cost_usd=0.01,
            total_tokens=500,
        )
    )

    resp = client.get("/metrics/summary?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "summaries" in data
    assert len(data["summaries"]) == 1
    assert data["summaries"][0]["total_tasks"] >= 1
    store.close()


def test_get_models() -> None:
    client, store = make_client()
    now = datetime.now(UTC)

    store.insert_cost(
        CostRecord(
            timestamp=now,
            model="gpt-4o",
            tier="balanced",
            task_id="t-m",
            cost_usd=0.01,
        )
    )

    resp = client.get("/metrics/models?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 1
    assert data["models"][0]["model"] == "gpt-4o"
    store.close()


def test_ingest_task_with_error() -> None:
    client, store = make_client()
    now = datetime.now(UTC)
    payload = {
        "task_id": "t-err",
        "complexity": "complex",
        "status": "failed",
        "start_time": now.isoformat(),
        "end_time": (now + timedelta(seconds=10)).isoformat(),
        "error": "something broke",
    }
    resp = client.post("/metrics/task", json=payload)
    assert resp.status_code == 200

    tasks = store.query_tasks(since=now - timedelta(minutes=1))
    assert len(tasks) == 1
    assert tasks[0]["status"] == "failed"
    assert tasks[0]["error"] == "something broke"
    store.close()


def test_ingest_cost_no_timestamp() -> None:
    client, store = make_client()
    payload = {
        "model": "gpt-4o-mini",
        "cost_usd": 0.001,
    }
    resp = client.post("/metrics/cost", json=payload)
    assert resp.status_code == 200

    now = datetime.now(UTC)
    costs = store.query_costs(since=now - timedelta(hours=1))
    assert len(costs) == 1
    assert costs[0]["model"] == "gpt-4o-mini"
    store.close()
