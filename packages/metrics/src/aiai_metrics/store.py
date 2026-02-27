"""DuckDB time-series storage for metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from aiai_core.types import CostRecord, TaskMetrics


def _to_naive(dt: datetime | None) -> datetime | None:
    """Strip timezone info for DuckDB TIMESTAMP columns (assumed UTC)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _to_naive_required(dt: datetime) -> datetime:
    """Strip timezone for required datetime fields."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


class MetricsStore:
    """Manages a DuckDB database for task metrics and cost records."""

    def __init__(self, db_path: str = "metrics.duckdb") -> None:
        self._conn = duckdb.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_metrics (
                task_id VARCHAR,
                complexity VARCHAR,
                status VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_cost_usd DOUBLE,
                total_tokens BIGINT,
                model_calls INTEGER,
                retries INTEGER,
                error VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_records (
                timestamp TIMESTAMP,
                model VARCHAR,
                tier VARCHAR,
                task_id VARCHAR,
                input_tokens BIGINT,
                output_tokens BIGINT,
                cost_usd DOUBLE,
                daily_total_usd DOUBLE
            )
        """)

    def insert_task(self, metrics: TaskMetrics) -> None:
        """Insert a task metrics record."""
        self._conn.execute(
            """
            INSERT INTO task_metrics
                (task_id, complexity, status, start_time, end_time,
                 total_cost_usd, total_tokens, model_calls, retries, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                metrics.task_id,
                metrics.complexity.value,
                metrics.status.value,
                _to_naive_required(metrics.start_time),
                _to_naive(metrics.end_time),
                metrics.total_cost_usd,
                metrics.total_tokens,
                metrics.model_calls,
                metrics.retries,
                metrics.error,
            ],
        )

    def insert_cost(self, record: CostRecord) -> None:
        """Insert a cost record."""
        self._conn.execute(
            """
            INSERT INTO cost_records
                (timestamp, model, tier, task_id, input_tokens,
                 output_tokens, cost_usd, daily_total_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _to_naive_required(record.timestamp),
                record.model,
                record.tier,
                record.task_id,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.daily_total_usd,
            ],
        )

    def query_tasks(
        self,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict[str, object]]:
        """Query task metrics within a time range."""
        if until is None:
            until = datetime.now(UTC)
        result = self._conn.execute(
            """
            SELECT task_id, complexity, status, start_time, end_time,
                   total_cost_usd, total_tokens, model_calls, retries, error
            FROM task_metrics
            WHERE start_time >= ? AND start_time <= ?
            ORDER BY start_time
            """,
            [_to_naive_required(since), _to_naive_required(until)],
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=False)) for row in result.fetchall()]

    def query_costs(
        self,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict[str, object]]:
        """Query cost records within a time range."""
        if until is None:
            until = datetime.now(UTC)
        result = self._conn.execute(
            """
            SELECT timestamp, model, tier, task_id, input_tokens,
                   output_tokens, cost_usd, daily_total_usd
            FROM cost_records
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            [_to_naive_required(since), _to_naive_required(until)],
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=False)) for row in result.fetchall()]

    def apply_retention(self, max_age_days: int) -> int:
        """Delete records older than max_age_days. Returns total rows deleted."""
        deleted = 0

        result = self._conn.execute(
            f"""
            DELETE FROM task_metrics
            WHERE start_time < CURRENT_TIMESTAMP - INTERVAL '{max_age_days} days'
            """,
        )
        deleted += result.fetchone()[0]  # type: ignore[index]

        result = self._conn.execute(
            f"""
            DELETE FROM cost_records
            WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '{max_age_days} days'
            """,
        )
        deleted += result.fetchone()[0]  # type: ignore[index]

        return int(deleted)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
