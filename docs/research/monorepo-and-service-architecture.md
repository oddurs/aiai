# Monorepo and Service Architecture for aiai

> Research document: Python monorepo patterns, custom analytics, dashboard design, and service architecture for an autonomous AI system that builds its own services.

---

## 1. Python Monorepo Patterns

### The Landscape

A monorepo puts all code in one repository. This is not the same as a monolith -- services can be independently deployable while sharing a single source-control home. Google, Meta, Microsoft, and Uber all use monorepos at massive scale. The question is not whether monorepos work, but which tooling fits a given project's size and complexity.

### Tooling Tiers

**Tier 1: Enterprise build systems (Bazel, Pants)**

Bazel (Google's open-source build system) and Pants (inspired by Google's internal Blaze) provide hermetic builds, fine-grained dependency graphs, remote caching, and distributed execution. Pants is often described as more approachable for Python projects, automatically inferring dependencies and supporting incremental builds. Bazel uses Starlark (a Python-like configuration language) for build files and excels at cross-language builds.

These are overkill for aiai. They require significant configuration overhead, introduce their own build language, and are designed for teams of 50+ engineers. When the primary developers are AI agents, build system complexity is wasted cognitive budget.

**Tier 2: Python-native workspace tools (uv, Hatch, PDM)**

uv workspaces (from Astral, the makers of ruff) are the modern standard. Inspired by Rust's Cargo workspaces, they manage multiple Python packages under a single root with a shared lockfile, automatic editable installs, and unified dependency resolution. Apache Airflow ships 120+ separate Python distributions from a single repo using uv workspaces.

Hatch provides multi-environment support and a build backend. PDM offers PEP 582 local packages and workspace support. Both are capable but have smaller ecosystems than uv.

**Tier 3: Manual approaches (pip -e ., simple scripts)**

A flat `src/` directory with `pip install -e .` and a single `pyproject.toml`. Works for small projects. Breaks down when services need different dependency sets or independent deployment.

### What Works for AI-Driven Development

AI agents are the primary developers of aiai. This changes the calculus:

- **Agents understand `pyproject.toml` natively.** Every model has seen thousands of them in training data. Custom build system configs (BUILD files, Starlark) are less familiar.
- **Agents need fast feedback loops.** `uv sync` resolves 900+ packages in seconds. Slow dependency resolution kills iteration speed.
- **Agents benefit from convention over configuration.** A predictable directory structure means less context needed per task.
- **Agents can read the lockfile.** `uv.lock` contains the full dependency graph in a parseable format, which agents can use to understand inter-package relationships.

**Recommendation: uv workspaces.** They provide the right balance of structure and simplicity. The single lockfile ensures consistency. Editable installs mean changes are immediately available. The `--package` flag lets agents target specific services.

### Import Path Design

In a uv workspace, each package has its own namespace:

```python
# From the router service
from aiai_core.config import load_config
from aiai_core.types import TaskComplexity

# From the analytics service
from aiai_analytics.collector import MetricsCollector
from aiai_analytics.storage import TimeSeriesStore

# From the dashboard
from aiai_dashboard.charts import render_chart
```

The `aiai_` prefix provides a clear namespace. Each package is independently importable. Shared code lives in `aiai_core`, not scattered across services.

**Avoid**: relative imports across package boundaries, `sys.path` manipulation, import hooks. These create invisible coupling that breaks when services are deployed independently.

---

## 2. Directory Structure Design

### The Concrete Layout

```
aiai/
├── pyproject.toml              # Workspace root -- defines members, shared dev deps
├── uv.lock                     # Single lockfile for the entire workspace
├── CLAUDE.md                   # Agent conventions
├── README.md
│
├── packages/
│   ├── core/                   # aiai-core: shared library
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── aiai_core/
│   │           ├── __init__.py
│   │           ├── config.py           # Configuration loading (models.yaml, etc.)
│   │           ├── types.py            # Shared types (TaskComplexity, AgentRole, etc.)
│   │           ├── openrouter.py       # OpenRouter client
│   │           ├── logging.py          # Structured logging
│   │           └── utils.py            # Small shared utilities
│   │
│   ├── router/                 # aiai-router: model routing service
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   │   ├── test_routing.py
│   │   │   └── test_fallback.py
│   │   └── src/
│   │       └── aiai_router/
│   │           ├── __init__.py
│   │           ├── router.py           # Route requests to models
│   │           ├── cost.py             # Cost tracking and budgets
│   │           └── fallback.py         # Fallback chain logic
│   │
│   ├── analytics/              # aiai-analytics: metrics collection and storage
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   │   ├── test_collector.py
│   │   │   ├── test_storage.py
│   │   │   └── test_queries.py
│   │   └── src/
│   │       └── aiai_analytics/
│   │           ├── __init__.py
│   │           ├── collector.py        # Metrics ingestion API
│   │           ├── storage.py          # DuckDB time-series storage
│   │           ├── queries.py          # Pre-built query functions
│   │           ├── aggregation.py      # Rollup and retention pipelines
│   │           └── models.py           # Data models (Metric, Event, etc.)
│   │
│   ├── dashboard/              # aiai-dashboard: web UI
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   │   └── test_routes.py
│   │   └── src/
│   │       └── aiai_dashboard/
│   │           ├── __init__.py
│   │           ├── app.py              # FastAPI application
│   │           ├── routes.py           # Dashboard routes
│   │           ├── sse.py              # Server-Sent Events for live updates
│   │           ├── templates/
│   │           │   ├── base.html
│   │           │   ├── dashboard.html
│   │           │   └── partials/       # htmx partial templates
│   │           │       ├── metrics_table.html
│   │           │       ├── chart.html
│   │           │       └── status.html
│   │           └── static/
│   │               ├── style.css
│   │               └── app.js          # Minimal JS (htmx + chart init)
│   │
│   └── cli/                    # aiai-cli: command-line interface
│       ├── pyproject.toml
│       ├── tests/
│       │   └── test_commands.py
│       └── src/
│           └── aiai_cli/
│               ├── __init__.py
│               ├── main.py             # CLI entry point
│               ├── commands/
│               │   ├── status.py       # System status
│               │   ├── metrics.py      # Query metrics
│               │   └── deploy.py       # Deployment commands
│               └── output.py           # Terminal formatting
│
├── config/
│   ├── models.yaml             # OpenRouter model routing config
│   └── settings.toml           # Environment-specific settings
│
├── deploy/
│   ├── Dockerfile              # Multi-stage build for all services
│   ├── docker-compose.yml      # Local development
│   ├── hetzner/
│   │   ├── setup.sh            # Server provisioning
│   │   └── deploy.sh           # Deployment script
│   └── systemd/
│       ├── aiai-analytics.service
│       └── aiai-dashboard.service
│
├── scripts/
│   ├── git-workflow.sh
│   ├── agent-git.sh
│   └── dev.sh                  # Start all services for local dev
│
├── docs/
│   ├── architecture.md
│   ├── vision.md
│   ├── concepts.md
│   └── research/
│
└── .github/
    └── workflows/
        ├── ci.yml              # Main CI pipeline
        └── deploy.yml          # Deployment pipeline
```

### Design Decisions in the Layout

**Tests colocated with packages, not in a top-level `tests/` directory.** Each package owns its tests. When an agent modifies `aiai_analytics/storage.py`, the relevant tests are right there in `packages/analytics/tests/`. This is better for AI developers because the agent does not need to map between two parallel directory trees. It also means `uv run --package aiai-analytics pytest` runs only the relevant tests.

**`packages/` rather than `services/`.** Not everything is a service. `core` is a library. `cli` is a command-line tool. "Packages" is the accurate term and matches uv workspace conventions.

**`deploy/` is separate from packages.** Deployment configuration cuts across all services. It does not belong inside any single package. Dockerfiles, systemd units, and Hetzner provisioning scripts live together.

**`config/` at the root.** Configuration that is shared across services (model routing, environment settings) lives at the project root. Service-specific config can live inside the service package if needed.

**Static files inside the dashboard package.** Templates and static assets are part of the dashboard, not a separate top-level directory. This keeps the dashboard self-contained and deployable as a unit.

### Where Shared Types Live

Shared types belong in `aiai_core.types`. This includes:

```python
# packages/core/src/aiai_core/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class TaskComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    timestamp: datetime
    tags: dict[str, str]
    source: str  # which service/agent produced this


@dataclass(frozen=True)
class ModelUsage:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    task_complexity: TaskComplexity
    timestamp: datetime
```

The rule: if two or more packages need the same type, it goes in `core`. If only one package uses it, it stays in that package.

---

## 3. Dependency Management in Monorepos

### Root pyproject.toml

The workspace root defines workspace membership and shared development dependencies:

```toml
[project]
name = "aiai"
version = "0.1.0"
requires-python = ">=3.11"
description = "AI that builds itself"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "pyright>=1.1",
    "pytest-cov>=6.0",
]

[build-system]
requires = ["uv_build>=0.10.6,<0.11.0"]
build-backend = "uv_build"
```

### Per-Package pyproject.toml

Each package declares its own runtime dependencies and workspace sources:

```toml
# packages/analytics/pyproject.toml
[project]
name = "aiai-analytics"
version = "0.1.0"
requires-python = ">=3.11"
description = "Metrics collection and time-series analytics for aiai"
dependencies = [
    "aiai-core",
    "duckdb>=1.1",
    "fastapi>=0.115",
    "uvicorn>=0.34",
]

[tool.uv.sources]
aiai-core = { workspace = true }

[build-system]
requires = ["uv_build>=0.10.6,<0.11.0"]
build-backend = "uv_build"
```

```toml
# packages/core/pyproject.toml
[project]
name = "aiai-core"
version = "0.1.0"
requires-python = ">=3.11"
description = "Shared core library for aiai"
dependencies = [
    "httpx>=0.28",
    "pyyaml>=6.0",
    "pydantic>=2.10",
]

[build-system]
requires = ["uv_build>=0.10.6,<0.11.0"]
build-backend = "uv_build"
```

```toml
# packages/dashboard/pyproject.toml
[project]
name = "aiai-dashboard"
version = "0.1.0"
requires-python = ">=3.11"
description = "Web dashboard for aiai metrics and system status"
dependencies = [
    "aiai-core",
    "aiai-analytics",
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "jinja2>=3.1",
    "sse-starlette>=2.0",
]

[tool.uv.sources]
aiai-core = { workspace = true }
aiai-analytics = { workspace = true }

[build-system]
requires = ["uv_build>=0.10.6,<0.11.0"]
build-backend = "uv_build"
```

### The Single Lockfile

`uv.lock` at the workspace root locks every dependency for every package. This means:

- All packages use the same version of `httpx`, `pydantic`, etc.
- No version conflicts between services at development time.
- One `uv lock` command resolves the entire workspace.
- The lockfile is the source of truth for the dependency graph.

This is a tradeoff. If `analytics` needs `duckdb>=1.1` and some future service needs `duckdb<1.0`, you have a conflict. In practice, this rarely happens for a small project. If it does, it is a signal to split into separate workspaces (see Section 8).

### Virtual Environments

uv creates a single virtual environment for the workspace at the root `.venv/`. All packages are installed as editable. This means:

- `import aiai_core` works from any package in the workspace.
- Changes to `core` are immediately visible to `analytics` without reinstalling.
- Python cannot isolate imports between workspace members -- if `analytics` has `duckdb` installed, `core` can accidentally import it too.

**Discipline required:** do not import a dependency you have not declared in your `pyproject.toml`. Agents should be instructed to check `pyproject.toml` before adding imports.

### Docker Implications

For deployment, each service gets its own Docker image built from a shared multi-stage Dockerfile:

```dockerfile
# deploy/Dockerfile
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Stage 1: Install dependencies without source code (cache layer)
FROM base AS deps
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/analytics/pyproject.toml packages/analytics/pyproject.toml

# Install deps without workspace packages for caching
RUN uv sync --frozen --no-install-workspace --package aiai-analytics

# Stage 2: Copy source and install workspace packages
FROM deps AS analytics
COPY packages/core/src packages/core/src
COPY packages/analytics/src packages/analytics/src
COPY config/ config/
RUN uv sync --frozen --package aiai-analytics

CMD ["uv", "run", "--package", "aiai-analytics", "uvicorn", "aiai_analytics.collector:app", "--host", "0.0.0.0", "--port", "8001"]
```

The key optimization: `--no-install-workspace` in the deps stage means third-party dependencies are cached. Source code changes do not trigger a full dependency reinstall.

---

## 4. Build-It-Ourselves Analytics

### Why Build It

Prometheus + Grafana is the industry standard. But "industry standard" means:

- Learning PromQL (a query language nobody enjoys).
- Configuring YAML-heavy scrape targets.
- Running two additional services (Prometheus server + Grafana).
- Managing Grafana dashboards through a web UI that does not version-control well.
- Storage that is designed for infrastructure metrics, not application-level AI analytics.

For aiai, the metrics are domain-specific: model costs, token usage, task completion rates, agent performance, self-improvement deltas. A custom system that stores exactly what we need, queries it with SQL, and renders it with Python is simpler to build, maintain, and extend than configuring Prometheus for a non-standard use case.

### Storage: DuckDB

DuckDB is an embedded columnar analytics database. It is to OLAP what SQLite is to OLTP: zero configuration, no server process, single-file storage, and blazingly fast analytical queries.

**Why DuckDB over alternatives:**

| Option | Pros | Cons |
|--------|------|------|
| **DuckDB** | Embedded, SQL, columnar, fast aggregations, Parquet export | Single-writer (fine for our scale) |
| SQLite | Embedded, ubiquitous | Row-oriented, slow for analytical queries over millions of rows |
| JSONL files | Simplest possible | No indexing, slow queries, manual parsing |
| ClickHouse | Extremely fast at scale | Requires a server process, operational overhead |
| Parquet files | Excellent for archival | No in-place updates, need a query engine on top |

DuckDB handles the sweet spot: it works as an embedded library (no server), supports standard SQL with window functions and time-series operations, and can query Parquet files directly for archival data.

### Data Model

```python
# packages/analytics/src/aiai_analytics/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"      # Monotonically increasing (total requests)
    GAUGE = "gauge"          # Point-in-time value (memory usage)
    HISTOGRAM = "histogram"  # Distribution of values (latency)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    source: str = "unknown"


@dataclass
class Event:
    """A discrete event (task completed, error occurred, deployment)."""
    name: str
    timestamp: datetime
    data: dict = field(default_factory=dict)
    source: str = "unknown"
```

### Storage Layer

```python
# packages/analytics/src/aiai_analytics/storage.py
from __future__ import annotations

import duckdb
from pathlib import Path
from datetime import datetime, timedelta

from aiai_analytics.models import MetricPoint, MetricType, Event


class TimeSeriesStore:
    """DuckDB-backed time-series storage for metrics and events."""

    def __init__(self, db_path: str | Path = "data/metrics.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                metric_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                tags JSON,
                source VARCHAR NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                name VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data JSON,
                source VARCHAR NOT NULL
            )
        """)
        # Indexes for common query patterns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_ts
            ON metrics (name, timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_name_ts
            ON events (name, timestamp)
        """)

    def record_metric(self, metric: MetricPoint) -> None:
        self.conn.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
            [
                metric.name,
                metric.value,
                metric.metric_type.value,
                metric.timestamp,
                metric.tags,
                metric.source,
            ],
        )

    def record_event(self, event: Event) -> None:
        self.conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            [event.name, event.timestamp, event.data, event.source],
        )

    def query_metrics(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[dict]:
        """Query metrics by name and optional time range."""
        query = "SELECT * FROM metrics WHERE name = ?"
        params: list = [name]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        if tags:
            for key, value in tags.items():
                query += f" AND json_extract_string(tags, '$.{key}') = ?"
                params.append(value)

        query += " ORDER BY timestamp"
        return self.conn.execute(query, params).fetchdf().to_dict("records")

    def aggregate(
        self,
        name: str,
        interval: str = "1 hour",
        agg_func: str = "avg",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict]:
        """Aggregate metrics over time intervals.

        Args:
            interval: DuckDB interval string ('1 hour', '5 minutes', '1 day')
            agg_func: Aggregation function ('avg', 'sum', 'min', 'max', 'count')
        """
        query = f"""
            SELECT
                time_bucket(INTERVAL '{interval}', timestamp) AS bucket,
                {agg_func}(value) AS value,
                count(*) AS sample_count
            FROM metrics
            WHERE name = ?
        """
        params: list = [name]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " GROUP BY bucket ORDER BY bucket"
        return self.conn.execute(query, params).fetchdf().to_dict("records")

    def rollup_and_purge(self, retention_days: int = 30, rollup_interval: str = "1 hour") -> int:
        """Roll up old fine-grained data into hourly aggregates, purge raw data."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        # Insert rolled-up data
        self.conn.execute(f"""
            INSERT INTO metrics
            SELECT
                name,
                avg(value) AS value,
                metric_type,
                time_bucket(INTERVAL '{rollup_interval}', timestamp) AS timestamp,
                NULL AS tags,
                'rollup' AS source
            FROM metrics
            WHERE timestamp < ? AND source != 'rollup'
            GROUP BY name, metric_type, time_bucket(INTERVAL '{rollup_interval}', timestamp)
        """, [cutoff])

        # Delete raw data older than retention
        result = self.conn.execute(
            "DELETE FROM metrics WHERE timestamp < ? AND source != 'rollup'",
            [cutoff],
        )
        return result.fetchone()[0]  # rows deleted

    def export_to_parquet(self, path: str) -> None:
        """Export metrics to Parquet for long-term archival."""
        self.conn.execute(f"COPY metrics TO '{path}' (FORMAT PARQUET)")

    def close(self) -> None:
        self.conn.close()
```

### Ingestion API

```python
# packages/analytics/src/aiai_analytics/collector.py
from __future__ import annotations

from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

from aiai_analytics.models import MetricPoint, MetricType, Event
from aiai_analytics.storage import TimeSeriesStore


app = FastAPI(title="aiai-analytics", version="0.1.0")
store = TimeSeriesStore()


class MetricIn(BaseModel):
    name: str
    value: float
    metric_type: str = "gauge"
    tags: dict[str, str] = {}
    source: str = "unknown"


class EventIn(BaseModel):
    name: str
    data: dict = {}
    source: str = "unknown"


class QueryParams(BaseModel):
    name: str
    start: datetime | None = None
    end: datetime | None = None
    interval: str = "1 hour"
    agg_func: str = "avg"
    tags: dict[str, str] = {}


@app.post("/v1/metrics")
async def ingest_metric(metric: MetricIn) -> dict:
    point = MetricPoint(
        name=metric.name,
        value=metric.value,
        metric_type=MetricType(metric.metric_type),
        timestamp=datetime.utcnow(),
        tags=metric.tags,
        source=metric.source,
    )
    store.record_metric(point)
    return {"status": "ok"}


@app.post("/v1/metrics/batch")
async def ingest_metrics_batch(metrics: list[MetricIn]) -> dict:
    for metric in metrics:
        point = MetricPoint(
            name=metric.name,
            value=metric.value,
            metric_type=MetricType(metric.metric_type),
            timestamp=datetime.utcnow(),
            tags=metric.tags,
            source=metric.source,
        )
        store.record_metric(point)
    return {"status": "ok", "count": len(metrics)}


@app.post("/v1/events")
async def ingest_event(event: EventIn) -> dict:
    evt = Event(
        name=event.name,
        timestamp=datetime.utcnow(),
        data=event.data,
        source=event.source,
    )
    store.record_event(evt)
    return {"status": "ok"}


@app.post("/v1/query")
async def query_metrics(params: QueryParams) -> dict:
    if params.interval:
        results = store.aggregate(
            name=params.name,
            interval=params.interval,
            agg_func=params.agg_func,
            start=params.start,
            end=params.end,
        )
    else:
        results = store.query_metrics(
            name=params.name,
            start=params.start,
            end=params.end,
            tags=params.tags,
        )
    return {"data": results}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "analytics"}
```

### Sending Metrics from Other Services

Any service in the monorepo can send metrics. A lightweight client in `core` makes this easy:

```python
# packages/core/src/aiai_core/metrics.py
from __future__ import annotations

import httpx
from datetime import datetime


class MetricsClient:
    """Lightweight client for sending metrics to the analytics service."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=5.0)

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None, source: str = "unknown") -> None:
        self._client.post("/v1/metrics", json={
            "name": name,
            "value": value,
            "metric_type": "gauge",
            "tags": tags or {},
            "source": source,
        })

    def counter(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None, source: str = "unknown") -> None:
        self._client.post("/v1/metrics", json={
            "name": name,
            "value": value,
            "metric_type": "counter",
            "tags": tags or {},
            "source": source,
        })

    def event(self, name: str, data: dict | None = None, source: str = "unknown") -> None:
        self._client.post("/v1/events", json={
            "name": name,
            "data": data or {},
            "source": source,
        })

    def close(self) -> None:
        self._client.close()
```

Usage from the router service:

```python
from aiai_core.metrics import MetricsClient

metrics = MetricsClient()
metrics.counter("router.requests", tags={"model": "claude-opus-4", "complexity": "complex"}, source="router")
metrics.gauge("router.cost_usd", 0.15, tags={"model": "claude-opus-4"}, source="router")
metrics.event("router.model_fallback", data={"from": "claude-opus-4", "to": "claude-sonnet-4", "reason": "rate_limit"}, source="router")
```

### Aggregation and Retention Strategy

Fine-grained data (per-second or per-request) is useful for debugging but expensive to store forever. The strategy:

1. **Raw data**: Keep for 7 days. Every metric point, every event.
2. **Hourly rollups**: Keep for 90 days. Average, min, max, count per hour.
3. **Daily rollups**: Keep for 1 year. Average, min, max, count per day.
4. **Parquet archive**: Export monthly snapshots for long-term analysis.

A scheduled task (cron or systemd timer) runs the rollup:

```python
# packages/analytics/src/aiai_analytics/maintenance.py
from aiai_analytics.storage import TimeSeriesStore
from datetime import datetime


def run_maintenance():
    store = TimeSeriesStore()

    # Roll up data older than 7 days into hourly buckets
    deleted = store.rollup_and_purge(retention_days=7, rollup_interval="1 hour")
    print(f"[{datetime.utcnow()}] Rolled up and purged {deleted} raw metric rows")

    # Export last month to Parquet for archival
    month = datetime.utcnow().strftime("%Y-%m")
    store.export_to_parquet(f"data/archive/metrics-{month}.parquet")

    store.close()


if __name__ == "__main__":
    run_maintenance()
```

---

## 5. Custom Dashboard Design

### The Stack: FastAPI + Jinja2 + htmx

This is the Python-native approach to dashboards. No React, no npm, no webpack, no node_modules. The server renders HTML. htmx makes it interactive. Chart.js draws the charts.

**Why this stack:**

- **FastAPI** is already used for the analytics API. Same framework, same patterns.
- **Jinja2** templates are just HTML with variables. AI agents generate HTML fluently.
- **htmx** adds interactivity with HTML attributes, not JavaScript. Partial page updates via AJAX, SSE for live data, all declarative.
- **Chart.js** is a lightweight (70KB) charting library with good defaults.

The result: a dashboard that loads in under 50ms, updates in real-time, and can be built and maintained entirely in Python with minimal JavaScript.

### Application Setup

```python
# packages/dashboard/src/aiai_dashboard/app.py
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aiai_dashboard.routes import router
from aiai_dashboard.sse import sse_router


PACKAGE_DIR = Path(__file__).parent

app = FastAPI(title="aiai-dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
app.include_router(router)
app.include_router(sse_router)

templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
```

### Dashboard Routes

```python
# packages/dashboard/src/aiai_dashboard/routes.py
from __future__ import annotations

from datetime import datetime, timedelta
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

from aiai_analytics.storage import TimeSeriesStore


PACKAGE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
router = APIRouter()
store = TimeSeriesStore()


@router.get("/")
async def dashboard(request: Request):
    """Main dashboard page."""
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)

    # Gather key metrics for the dashboard
    cost_data = store.aggregate("router.cost_usd", interval="1 hour", agg_func="sum", start=last_24h)
    request_data = store.aggregate("router.requests", interval="1 hour", agg_func="count", start=last_24h)

    total_cost = sum(row["value"] for row in cost_data) if cost_data else 0
    total_requests = sum(row["value"] for row in request_data) if request_data else 0

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_cost_24h": f"${total_cost:.2f}",
        "total_requests_24h": int(total_requests),
        "cost_chart_data": cost_data,
        "request_chart_data": request_data,
        "last_updated": now.isoformat(),
    })


@router.get("/partials/metrics-table")
async def metrics_table_partial(request: Request):
    """htmx partial: refreshable metrics table."""
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)

    recent = store.query_metrics("router.cost_usd", start=last_hour)
    return templates.TemplateResponse("partials/metrics_table.html", {
        "request": request,
        "metrics": recent[-50:],  # Last 50 data points
    })


@router.get("/partials/cost-chart")
async def cost_chart_partial(request: Request):
    """htmx partial: cost chart data."""
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)

    data = store.aggregate("router.cost_usd", interval="1 hour", agg_func="sum", start=last_24h)
    return templates.TemplateResponse("partials/chart.html", {
        "request": request,
        "chart_id": "cost-chart",
        "chart_label": "Cost (USD/hour)",
        "labels": [row["bucket"].isoformat() for row in data],
        "values": [round(row["value"], 4) for row in data],
    })
```

### Server-Sent Events for Live Updates

```python
# packages/dashboard/src/aiai_dashboard/sse.py
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from aiai_analytics.storage import TimeSeriesStore


sse_router = APIRouter()
store = TimeSeriesStore()


async def metrics_stream():
    """Generate SSE events with latest metrics every 5 seconds."""
    while True:
        now = datetime.utcnow()
        cost = store.query_metrics("router.cost_usd", start=now)
        requests = store.query_metrics("router.requests", start=now)

        data = {
            "timestamp": now.isoformat(),
            "cost_last_min": sum(r["value"] for r in cost) if cost else 0,
            "requests_last_min": len(requests),
        }
        yield {"event": "metrics", "data": json.dumps(data)}
        await asyncio.sleep(5)


@sse_router.get("/sse/metrics")
async def sse_metrics():
    """SSE endpoint for real-time metric updates."""
    return EventSourceResponse(metrics_stream())
```

### Base Template

```html
<!-- packages/dashboard/src/aiai_dashboard/templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>aiai dashboard</title>
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/htmx-ext-sse@2.3.0/sse.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4/dist/chart.umd.min.js"></script>
</head>
<body>
    <nav>
        <div class="nav-brand">aiai</div>
        <div class="nav-links">
            <a href="/" class="active">Dashboard</a>
            <a href="/agents">Agents</a>
            <a href="/costs">Costs</a>
        </div>
    </nav>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

### Dashboard Template

```html
<!-- packages/dashboard/src/aiai_dashboard/templates/dashboard.html -->
{% extends "base.html" %}

{% block content %}
<div class="dashboard-grid">

    <!-- Summary cards with live SSE updates -->
    <div class="card" hx-ext="sse" sse-connect="/sse/metrics">
        <h3>Cost (24h)</h3>
        <div class="metric-value" sse-swap="metrics"
             hx-swap="innerHTML">
            {{ total_cost_24h }}
        </div>
    </div>

    <div class="card">
        <h3>Requests (24h)</h3>
        <div class="metric-value">{{ total_requests_24h }}</div>
    </div>

    <!-- Cost chart - auto-refreshes every 60 seconds -->
    <div class="card wide"
         hx-get="/partials/cost-chart"
         hx-trigger="load, every 60s"
         hx-swap="innerHTML">
        <p>Loading chart...</p>
    </div>

    <!-- Recent metrics table - refreshes every 10 seconds -->
    <div class="card wide"
         hx-get="/partials/metrics-table"
         hx-trigger="load, every 10s"
         hx-swap="innerHTML">
        <p>Loading metrics...</p>
    </div>
</div>
{% endblock %}
```

### Chart Partial Template

```html
<!-- packages/dashboard/src/aiai_dashboard/templates/partials/chart.html -->
<h3>{{ chart_label }}</h3>
<canvas id="{{ chart_id }}" height="200"></canvas>
<script>
    (function() {
        const ctx = document.getElementById('{{ chart_id }}').getContext('2d');
        // Destroy existing chart if re-rendering via htmx
        if (Chart.getChart(ctx)) Chart.getChart(ctx).destroy();

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ labels | tojson }},
                datasets: [{
                    label: '{{ chart_label }}',
                    data: {{ values | tojson }},
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'time' in Chart.registry.scales ? 'time' : 'category',
                        ticks: { maxTicksToRotation: 0, maxRotation: 0 }
                    },
                    y: { beginAtZero: true }
                },
                plugins: { legend: { display: false } }
            }
        });
    })();
</script>
```

### Minimal CSS

```css
/* packages/dashboard/src/aiai_dashboard/static/style.css */
:root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --border: #334155;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
}

.nav-brand {
    font-size: 1.25rem;
    font-weight: 700;
    font-family: monospace;
}

.nav-links a {
    color: var(--muted);
    text-decoration: none;
    margin-left: 1.5rem;
}

.nav-links a.active { color: var(--accent); }

main { padding: 2rem; }

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

.card {
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
    border: 1px solid var(--border);
}

.card.wide { grid-column: 1 / -1; }

.card h3 {
    color: var(--muted);
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: monospace;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

th, td {
    text-align: left;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
}

th { color: var(--muted); font-weight: 500; }

/* Responsive: stack cards on mobile */
@media (max-width: 768px) {
    main { padding: 1rem; }
    .dashboard-grid { grid-template-columns: 1fr; }
    .card.wide { grid-column: 1; }
}
```

### Authentication for Internal Tools

For a Hetzner deployment accessed only by the team (or by AI agents), heavy authentication is unnecessary. Practical options:

1. **No auth + firewall rules.** Bind the dashboard to `127.0.0.1` or the private network. Access via SSH tunnel or WireGuard VPN. Simplest approach.
2. **Basic HTTP auth.** A single shared password via FastAPI middleware. Good enough for internal tools.
3. **Token in URL.** A random token in the URL path (`/dashboard/{token}/`). Not secure for production, but prevents casual access.

Start with option 1 (firewall). Add basic auth if the tool is exposed publicly.

---

## 6. Service Communication in a Monorepo

### The Spectrum

Services in a monorepo can communicate at different levels of coupling:

```
Tighter coupling                                          Looser coupling
       |                                                        |
  Direct import  →  Function call  →  Unix socket  →  HTTP API  →  Message queue
       |              (in-process)     (same host)    (any host)    (async, durable)
```

### Direct Import (Library Mode)

When two packages are in the same process, direct import is the simplest approach:

```python
# In the router service, directly using analytics storage
from aiai_analytics.storage import TimeSeriesStore
from aiai_analytics.models import MetricPoint, MetricType

store = TimeSeriesStore()
store.record_metric(MetricPoint(
    name="router.latency_ms",
    value=42.5,
    metric_type=MetricType.GAUGE,
    timestamp=datetime.utcnow(),
    source="router",
))
```

**When this works:** early development, single-process deployment, when services are really libraries that share a runtime. This is where aiai should start.

**When it breaks:** when you need independent scaling, when one service crashes and should not take down the other, when services need different resource profiles (CPU-bound vs I/O-bound).

### HTTP API (Service Mode)

When services run as separate processes, they communicate over HTTP:

```python
# Router service sends metrics to analytics service
import httpx

async def record_metric(name: str, value: float, tags: dict):
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:8001/v1/metrics", json={
            "name": name,
            "value": value,
            "tags": tags,
            "source": "router",
        })
```

**When to use HTTP:** when services run as separate processes on the same or different hosts. FastAPI makes this nearly free -- the analytics service already has an HTTP API.

### Unix Sockets (Same-Host Optimization)

When services run on the same Hetzner machine, Unix sockets avoid TCP overhead:

```python
# Analytics service listening on a Unix socket
uvicorn.run(app, uds="/tmp/aiai-analytics.sock")

# Router service connecting via Unix socket
client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds="/tmp/aiai-analytics.sock"))
await client.post("http://localhost/v1/metrics", json={...})
```

This is an optimization, not a starting point. Use TCP first. Switch to Unix sockets if latency matters.

### Shared State Through Files

For some patterns, a shared file is simpler than an API:

```python
# Analytics writes a summary file every minute
import json
from pathlib import Path

summary = {
    "total_cost_today": 12.34,
    "total_requests_today": 1567,
    "last_updated": datetime.utcnow().isoformat(),
}
Path("data/summary.json").write_text(json.dumps(summary))

# Dashboard reads the summary file
summary = json.loads(Path("data/summary.json").read_text())
```

This works for read-heavy data that updates infrequently. No API needed, no network calls, no connection management. The filesystem is the communication channel.

### Event-Driven Patterns (Without a Message Queue)

A full message queue (RabbitMQ, Redis Streams) is overkill at the start. For simple event-driven patterns, use an append-only file or SQLite table as a lightweight event log:

```python
# Simple file-based event bus
import json
from pathlib import Path
from datetime import datetime
from filelock import FileLock

EVENT_LOG = Path("data/events.jsonl")
LOCK = FileLock("data/events.jsonl.lock")


def publish(event_name: str, data: dict) -> None:
    event = {
        "name": event_name,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with LOCK:
        with EVENT_LOG.open("a") as f:
            f.write(json.dumps(event) + "\n")


def consume_since(last_offset: int) -> tuple[list[dict], int]:
    with LOCK:
        lines = EVENT_LOG.read_text().strip().split("\n")
    events = [json.loads(line) for line in lines[last_offset:]]
    return events, len(lines)
```

### Recommended Progression for aiai

1. **Phase 1 (now):** Direct imports. Everything in one process. `core`, `analytics`, `router` are libraries imported by a single `main.py`.
2. **Phase 2 (when needed):** Split analytics and dashboard into separate processes. Communicate over HTTP on localhost.
3. **Phase 3 (if needed):** Split across multiple Hetzner machines. HTTP over private network. Unix sockets for same-host services.
4. **Phase 4 (probably never at this scale):** Message queue for async event processing.

---

## 7. CI/CD for Monorepos

### Path-Based Triggers in GitHub Actions

GitHub Actions natively supports `paths` filters on `push` and `pull_request` triggers. Combined with the `dorny/paths-filter` action for job-level filtering, you get efficient CI that only tests what changed.

### Main CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      core: ${{ steps.filter.outputs.core }}
      analytics: ${{ steps.filter.outputs.analytics }}
      dashboard: ${{ steps.filter.outputs.dashboard }}
      router: ${{ steps.filter.outputs.router }}
      cli: ${{ steps.filter.outputs.cli }}
      any_package: ${{ steps.filter.outputs.any_package }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            core:
              - 'packages/core/**'
            analytics:
              - 'packages/analytics/**'
              - 'packages/core/**'
            dashboard:
              - 'packages/dashboard/**'
              - 'packages/core/**'
              - 'packages/analytics/**'
            router:
              - 'packages/router/**'
              - 'packages/core/**'
            cli:
              - 'packages/cli/**'
              - 'packages/core/**'
            any_package:
              - 'packages/**'

  lint:
    needs: detect-changes
    if: needs.detect-changes.outputs.any_package == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check packages/
      - run: uv run ruff format --check packages/
      - run: uv run pyright packages/

  test-core:
    needs: detect-changes
    if: needs.detect-changes.outputs.core == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --package aiai-core
      - run: uv run --package aiai-core pytest packages/core/tests/ -v

  test-analytics:
    needs: detect-changes
    if: needs.detect-changes.outputs.analytics == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --package aiai-analytics
      - run: uv run --package aiai-analytics pytest packages/analytics/tests/ -v

  test-dashboard:
    needs: detect-changes
    if: needs.detect-changes.outputs.dashboard == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --package aiai-dashboard
      - run: uv run --package aiai-dashboard pytest packages/dashboard/tests/ -v

  test-router:
    needs: detect-changes
    if: needs.detect-changes.outputs.router == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --package aiai-router
      - run: uv run --package aiai-router pytest packages/router/tests/ -v

  test-cli:
    needs: detect-changes
    if: needs.detect-changes.outputs.cli == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --package aiai-cli
      - run: uv run --package aiai-cli pytest packages/cli/tests/ -v

  # Safety net: full test suite on main
  test-all:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run pytest packages/ -v --tb=short
```

### Key Design Decisions

**Change detection includes dependencies.** When `core` changes, `analytics`, `dashboard`, `router`, and `cli` all re-test because they depend on `core`. The `dorny/paths-filter` config makes these dependency chains explicit.

**Lint everything if anything changed.** Linting is fast. Running ruff and pyright across all packages takes seconds. Not worth the complexity of per-package lint jobs.

**Full test suite on main.** Every push to main runs all tests regardless of what changed. This catches integration issues that per-package testing might miss. The per-package filtering is for PR speed.

**`uv sync --frozen` ensures reproducibility.** The `--frozen` flag uses the lockfile exactly as committed. No dependency resolution in CI.

### Self-Hosted Runners on Hetzner

GitHub-hosted runners work but cost money and add latency. A self-hosted runner on the Hetzner machine gives you:

- Free CI minutes.
- Pre-warmed caches (uv cache, Docker layers).
- Access to the production environment for integration tests.
- No data transfer costs.

```yaml
# .github/workflows/ci-self-hosted.yml
jobs:
  test:
    runs-on: self-hosted  # Runs on your Hetzner machine
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --frozen
      - run: uv run pytest packages/ -v
```

Setup is straightforward: install the GitHub Actions runner on Hetzner, register it with the repository, and use `runs-on: self-hosted` in workflows. The runner persists between jobs, so the uv cache is always warm.

### Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    paths:
      - 'packages/**'
      - 'deploy/**'
      - 'config/**'

jobs:
  deploy:
    runs-on: self-hosted
    needs: [test-all]  # Only deploy after full test suite passes
    steps:
      - uses: actions/checkout@v4

      - name: Build and deploy analytics
        if: contains(github.event.head_commit.modified, 'packages/analytics') || contains(github.event.head_commit.modified, 'packages/core')
        run: |
          cd deploy
          docker compose build analytics
          docker compose up -d analytics

      - name: Build and deploy dashboard
        if: contains(github.event.head_commit.modified, 'packages/dashboard') || contains(github.event.head_commit.modified, 'packages/core')
        run: |
          cd deploy
          docker compose build dashboard
          docker compose up -d dashboard
```

---

## 8. When to Split Services

### Signals That a Component Should Become Its Own Service

Not everything needs to be a service. Libraries should stay libraries. But when any of these signals appear, it is time to split:

**Resource divergence.** The analytics service is CPU-bound (running DuckDB aggregation queries) while the dashboard is I/O-bound (serving HTTP requests). Running them in one process means a heavy aggregation query blocks dashboard rendering.

**Deployment frequency mismatch.** The dashboard changes daily (UI tweaks, new charts) while the analytics storage layer changes monthly. Deploying the whole thing every time someone fixes a CSS typo is wasteful.

**Failure isolation.** If the analytics ingestion crashes due to a malformed metric, the dashboard should keep serving cached data. In a single process, one crash takes everything down.

**Scale requirements.** The dashboard gets 10 requests/minute. The metrics ingestion endpoint gets 1000 requests/minute. Scaling them together wastes resources.

**Code size.** When a package exceeds 2000-3000 lines of Python, it is probably doing too much. Split it into focused services or libraries.

### Signals to Keep Things Together

**Shared data access patterns.** If two components always read the same data from the same DuckDB file, putting them in separate processes means either sharing the file (DuckDB supports single-writer) or duplicating the data.

**Tight request coupling.** If every dashboard request requires a query to analytics, the HTTP round-trip overhead of splitting them might be worse than keeping them in-process.

**Operational overhead.** Every separate service needs its own process management, health checking, log collection, and deployment pipeline. At small scale, this overhead dominates.

### The Decision Framework

```
Q: Does this component need to run on a different schedule?
   Yes → Split
   No  → Continue

Q: Does this component need different resources (CPU/memory)?
   Yes → Split
   No  → Continue

Q: Would a crash in this component affect unrelated functionality?
   Yes → Split
   No  → Continue

Q: Is this component > 3000 lines and growing?
   Yes → Consider splitting
   No  → Keep together

Default: Keep together. Split when the pain is real, not theoretical.
```

### How to Split Cleanly

The monorepo structure makes splitting easy because packages already have clear boundaries:

1. The package already has its own `pyproject.toml` with declared dependencies.
2. The package already has its own test suite.
3. The package communicates through defined interfaces (imports or HTTP).

To promote a library to a service:

1. Add an entry point (FastAPI app or CLI script) to the package.
2. Add a Dockerfile stage for the package (or a systemd service file).
3. Replace direct imports in other packages with HTTP client calls.
4. Add health check and startup logic.
5. Update CI to deploy the new service independently.

The monorepo means all this happens in one commit, one PR, one deployment. No repository-splitting drama.

---

## 9. Recommended Architecture for aiai

### Phase 1: Everything is a Library (Start Here)

```
aiai/
├── pyproject.toml              # Workspace root
├── uv.lock
├── CLAUDE.md
│
├── packages/
│   ├── core/                   # Shared types, config, OpenRouter client
│   │   ├── pyproject.toml
│   │   └── src/aiai_core/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── types.py
│   │       ├── openrouter.py
│   │       ├── metrics.py      # MetricsClient for sending metrics
│   │       └── logging.py
│   │
│   ├── router/                 # Model routing logic (library, not service)
│   │   ├── pyproject.toml
│   │   └── src/aiai_router/
│   │       ├── __init__.py
│   │       ├── router.py
│   │       ├── cost.py
│   │       └── fallback.py
│   │
│   └── analytics/              # Metrics storage and queries (library)
│       ├── pyproject.toml
│       └── src/aiai_analytics/
│           ├── __init__.py
│           ├── collector.py    # FastAPI app (can run standalone or be mounted)
│           ├── storage.py
│           ├── queries.py
│           └── models.py
│
├── config/
│   └── models.yaml
├── deploy/
├── scripts/
├── docs/
└── .github/
```

In this phase, everything can run in a single process. The CLI or a single FastAPI app imports from all packages:

```python
# A single entry point that mounts everything
from fastapi import FastAPI
from aiai_analytics.collector import app as analytics_app

app = FastAPI(title="aiai")
app.mount("/analytics", analytics_app)

# Or simply use the libraries directly:
from aiai_router.router import route_request
from aiai_analytics.storage import TimeSeriesStore

store = TimeSeriesStore()
result = route_request(prompt="...", complexity="medium")
store.record_metric(...)
```

### Phase 2: Analytics and Dashboard Become Services

When the dashboard needs to serve requests while analytics runs heavy aggregations, split them into separate processes:

```
aiai/
├── packages/
│   ├── core/                   # Still a library
│   ├── router/                 # Still a library
│   ├── analytics/              # Now a service (runs as its own process)
│   ├── dashboard/              # New: web dashboard service
│   └── cli/                    # New: command-line interface
│
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml      # Runs analytics + dashboard
```

The `docker-compose.yml` for local development:

```yaml
# deploy/docker-compose.yml
services:
  analytics:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
      target: analytics
    ports:
      - "8001:8001"
    volumes:
      - ../data:/app/data

  dashboard:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
      target: dashboard
    ports:
      - "8000:8000"
    depends_on:
      - analytics
    environment:
      - ANALYTICS_URL=http://analytics:8001
```

### Phase 3: AI Builds New Services

This is where the architecture pays off. When the AI decides it needs a new service -- say, a prompt evaluation system or a code quality analyzer -- the pattern is established:

1. Create `packages/evaluator/` with a `pyproject.toml`.
2. Add `aiai-core` as a workspace dependency.
3. Implement the service following the established patterns.
4. Add tests in `packages/evaluator/tests/`.
5. Add a CI job in the workflow (or rely on the catch-all `test-all` job).
6. Add a Dockerfile stage if it needs independent deployment.

The AI agent does not need to make architectural decisions. The monorepo structure is the decision framework. New capabilities slot into `packages/` with predictable conventions.

### What the AI Builds Next (Likely Candidates)

Based on the project vision, these packages will likely emerge:

```
packages/
├── core/           # Exists
├── router/         # Exists
├── analytics/      # Exists
├── dashboard/      # Exists
├── cli/            # Exists
│
│   # Likely next:
├── evaluator/      # Evaluate agent outputs for quality
├── memory/         # Persistent memory system (beyond git)
├── evolution/      # Self-improvement engine
├── scheduler/      # Task scheduling and queue management
└── tools/          # Tool registry for agent capabilities
```

Each follows the same pattern: `pyproject.toml`, `src/aiai_{name}/`, `tests/`, workspace dependency on `core`.

### Configuration Management

All services read from a shared configuration system:

```python
# packages/core/src/aiai_core/config.py
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class Config:
    """Central configuration loaded from config/ and environment."""
    # Model routing
    models_path: Path = Path("config/models.yaml")

    # Analytics
    analytics_db: Path = Path("data/metrics.duckdb")
    analytics_url: str = "http://localhost:8001"

    # Dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    # Deployment
    environment: str = "development"

    @classmethod
    def load(cls) -> Config:
        config = cls()
        # Environment variables override defaults
        config.analytics_url = os.getenv("ANALYTICS_URL", config.analytics_url)
        config.dashboard_host = os.getenv("DASHBOARD_HOST", config.dashboard_host)
        config.dashboard_port = int(os.getenv("DASHBOARD_PORT", str(config.dashboard_port)))
        config.environment = os.getenv("AIAI_ENV", config.environment)
        return config

    def load_models(self) -> dict:
        with open(self.models_path) as f:
            return yaml.safe_load(f)
```

### The Migration Path Summarized

```
Phase 1: Library mode
  - All packages imported into one process
  - Single `uv run` starts everything
  - DuckDB file shared directly
  - Good for: development, MVP, < 100 req/min

Phase 2: Service mode
  - Analytics and dashboard run as separate processes
  - Communicate over HTTP (localhost or Unix socket)
  - Each service has its own Dockerfile stage
  - Good for: production, 100-10K req/min, independent scaling

Phase 3: Distributed mode
  - Services on different Hetzner machines
  - HTTP over private network
  - Centralized config service or environment variables
  - Good for: high scale, geographic distribution

Phase 4: Platform mode
  - AI creates and deploys new services autonomously
  - Service mesh for discovery and routing
  - The monorepo is the platform, packages are the units of deployment
  - Good for: when the AI has built 10+ services
```

The monorepo makes every phase transition a refactoring task, not an architectural migration. Packages already have clean boundaries. The only change is how they communicate and where they run.

---

## Sources

- [uv Workspaces Documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [Python Workspaces (Monorepos)](https://tomasrepcik.dev/blog/2025/2025-10-26-python-workspaces/)
- [Cracking the Python Monorepo](https://gafni.dev/blog/cracking-the-python-monorepo/)
- [uv Monorepo Best Practices Discussion](https://github.com/astral-sh/uv/issues/10960)
- [FOSDEM Talk: Modern Python Monorepo with uv](https://pydevtools.com/blog/fosdem-talk-modern-python-monorepo/)
- [DuckDB Time Series Analytics](https://medium.com/@Quaxel/time-series-crunching-with-duckdb-without-losing-your-mind-fd129ba7173f)
- [Observability and Log Analytics with DuckDB](https://neogeografia.wordpress.com/2023/08/02/observability-and-log-analytics-with-duckdb/)
- [Building Real-Time Dashboards with FastAPI and HTMX](https://medium.com/codex/building-real-time-dashboards-with-fastapi-and-htmx-01ea458673cb)
- [HTMX FastAPI Patterns 2025](https://johal.in/htmx-fastapi-patterns-hypermedia-driven-single-page-applications-2025/)
- [fasthx: Server-side rendering for FastAPI with HTMX](https://github.com/volfpeter/fasthx)
- [GitHub Actions Monorepo CI/CD Guide](https://dev.to/pockit_tools/github-actions-in-2026-the-complete-guide-to-monorepo-cicd-and-self-hosted-runners-1jop)
- [dorny/paths-filter for Monorepo CI](https://github.com/dorny/paths-filter)
- [Pants Build System](https://v1.pantsbuild.org/why_use_pants.html)
- [Monorepo Tooling Comparison (Bazel, Pants, Nx)](https://graphite.com/guides/monorepo-tooling-comparison)
- [Python Monorepo with uv and pex](https://chrismati.cz/posts/uv-pex-monorepo/)
- [Monorepo Guide: Managing Repositories and Microservices](https://www.aviator.co/blog/monorepo-a-hands-on-guide-for-managing-repositories-and-microservices/)
