"""FastAPI ingestion and query API for metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aiai_core.types import Complexity, CostRecord, TaskMetrics, TaskStatus
from fastapi import FastAPI, Query
from pydantic import BaseModel

from aiai_metrics.aggregator import MetricsAggregator
from aiai_metrics.store import MetricsStore

# --- Pydantic request/response models ---


class TaskMetricsRequest(BaseModel):
    """Request body for ingesting task metrics."""

    task_id: str
    complexity: str
    status: str
    start_time: str
    end_time: str | None = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    model_calls: int = 0
    retries: int = 0
    error: str | None = None


class CostRecordRequest(BaseModel):
    """Request body for ingesting cost records."""

    timestamp: str | None = None
    model: str = ""
    tier: str = ""
    task_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    daily_total_usd: float = 0.0


class TaskMetricsResponse(BaseModel):
    """Response confirming task metrics ingestion."""

    status: str = "ok"
    task_id: str


class CostRecordResponse(BaseModel):
    """Response confirming cost record ingestion."""

    status: str = "ok"


class DailySummaryResponse(BaseModel):
    """Response containing daily summaries."""

    summaries: list[dict[str, object]]


class ModelPerformanceResponse(BaseModel):
    """Response containing model performance data."""

    models: list[dict[str, object]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"


# --- App factory ---


def create_app(store: MetricsStore | None = None) -> FastAPI:
    """Create the FastAPI app with optional injected store."""
    app = FastAPI(title="aiai-metrics", version="0.1.0")

    if store is None:
        store = MetricsStore(":memory:")

    aggregator = MetricsAggregator(store)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/metrics/task", response_model=TaskMetricsResponse)
    async def ingest_task(req: TaskMetricsRequest) -> TaskMetricsResponse:
        end_time = (
            datetime.fromisoformat(req.end_time) if req.end_time else None
        )
        metrics = TaskMetrics(
            task_id=req.task_id,
            complexity=Complexity(req.complexity),
            status=TaskStatus(req.status),
            start_time=datetime.fromisoformat(req.start_time),
            end_time=end_time,
            total_cost_usd=req.total_cost_usd,
            total_tokens=req.total_tokens,
            model_calls=req.model_calls,
            retries=req.retries,
            error=req.error,
        )
        store.insert_task(metrics)  # type: ignore[union-attr]
        return TaskMetricsResponse(task_id=req.task_id)

    @app.post("/metrics/cost", response_model=CostRecordResponse)
    async def ingest_cost(req: CostRecordRequest) -> CostRecordResponse:
        ts = (
            datetime.fromisoformat(req.timestamp)
            if req.timestamp
            else datetime.now(UTC)
        )
        record = CostRecord(
            timestamp=ts,
            model=req.model,
            tier=req.tier,
            task_id=req.task_id,
            input_tokens=req.input_tokens,
            output_tokens=req.output_tokens,
            cost_usd=req.cost_usd,
            daily_total_usd=req.daily_total_usd,
        )
        store.insert_cost(record)  # type: ignore[union-attr]
        return CostRecordResponse()

    @app.get("/metrics/summary", response_model=DailySummaryResponse)
    async def get_summary(
        days: int = Query(default=7, ge=1, le=365),
    ) -> DailySummaryResponse:
        summaries = []
        now = datetime.now(UTC)
        for i in range(days):
            day = now - timedelta(days=i)
            summary = aggregator.daily_summary(day)
            summaries.append(
                {
                    "date": summary.date,
                    "total_tasks": summary.total_tasks,
                    "successful_tasks": summary.successful_tasks,
                    "failed_tasks": summary.failed_tasks,
                    "total_cost_usd": summary.total_cost_usd,
                    "total_tokens": summary.total_tokens,
                    "success_rate": summary.success_rate,
                }
            )
        return DailySummaryResponse(summaries=summaries)

    @app.get("/metrics/models", response_model=ModelPerformanceResponse)
    async def get_models(
        days: int = Query(default=7, ge=1, le=365),
    ) -> ModelPerformanceResponse:
        since = datetime.now(UTC) - timedelta(days=days)
        perfs = aggregator.model_performance(since=since)
        models = [
            {
                "model": p.model,
                "call_count": p.call_count,
                "total_cost": p.total_cost,
                "avg_latency": p.avg_latency,
                "error_rate": p.error_rate,
            }
            for p in perfs
        ]
        return ModelPerformanceResponse(models=models)

    return app
