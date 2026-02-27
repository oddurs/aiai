# Metrics Collection and Self-Improvement Engine Design

**Research compiled: 2026-02-26**

> How an autonomous AI system observes itself, detects patterns, generates hypotheses, and implements improvements -- without human gates. This document covers the full stack from raw metric collection through evolutionary self-improvement, with concrete Python implementations and architecture patterns.

---

## Table of Contents

1. [What to Measure in Autonomous AI](#1-what-to-measure-in-autonomous-ai)
2. [Metrics Collection Architecture](#2-metrics-collection-architecture)
3. [Pattern Detection in Agent Metrics](#3-pattern-detection-in-agent-metrics)
4. [Hypothesis Generation for Self-Improvement](#4-hypothesis-generation-for-self-improvement)
5. [A/B Testing for Autonomous Systems](#5-ab-testing-for-autonomous-systems)
6. [AlphaEvolve Deep Dive](#6-alphaevolve-deep-dive)
7. [Prompt Evolution Techniques](#7-prompt-evolution-techniques)
8. [Tool Creation by AI](#8-tool-creation-by-ai)
9. [Evolution Engine Architecture Patterns](#9-evolution-engine-architecture-patterns)
10. [Preventing Model Collapse in Self-Improvement](#10-preventing-model-collapse-in-self-improvement)
11. [References](#11-references)

---

## 1. What to Measure in Autonomous AI

The first question for any self-improving system is: improving *what*? Without clear metrics, the system has no gradient to follow. The metrics must be automatically collectible (no human annotation), causally linked to real outcomes, and resistant to Goodhart's Law (optimizing the metric should not diverge from the actual goal).

### 1.1 Core Metric Categories

#### Task Outcome Metrics (Lagging Indicators)

These measure whether the system actually accomplished its goals.

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Task Success Rate** | `completed_tasks / total_tasks` | The most fundamental measure -- did the agent do what was asked? |
| **Partial Success Score** | 0.0-1.0 score based on how much of a task was completed | Binary success/fail loses nuance; partial credit matters for complex tasks |
| **Test Pass Rate** | `passing_tests / total_tests` for code-producing tasks | Objective, automated measure of code correctness |
| **Error Rate** | `tasks_with_errors / total_tasks` | Tracks failure modes -- errors that crash the agent vs. silent failures |
| **Retry Rate** | `tasks_requiring_retry / total_tasks` | High retry rates indicate fragile strategies even if eventual success rate is high |
| **Regression Rate** | Tasks that previously succeeded but now fail | Critical for detecting self-improvement gone wrong |

#### Efficiency Metrics (Leading Indicators)

These predict future performance and resource consumption.

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Cost per Task** | Total LLM API spend for a single task | Directly impacts economic viability; should trend downward |
| **Time to Completion** | Wall-clock seconds from task start to finish | User-facing latency; includes API wait time, retries, and processing |
| **Tokens per Task** | Total input + output tokens consumed | Proxy for cost that is model-price-independent |
| **Model Efficiency** | `quality_score / cost` | Quality per dollar -- the metric to maximize |
| **Tool Calls per Task** | Number of tool invocations to complete a task | Measures how efficiently the agent uses its tools |
| **Steps per Task** | Number of agent loop iterations | Fewer steps with same success = better planning |

#### Code Quality Metrics

For a coding agent, the output quality matters beyond "does it pass tests."

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Lint Score** | Linter warnings/errors per file | Automated code style and correctness check |
| **Cyclomatic Complexity** | McCabe complexity of generated functions | High complexity = harder to maintain |
| **Test Coverage Delta** | Change in test coverage after agent modification | Did the agent leave the codebase better or worse? |
| **Lines Changed** | Total lines added/modified/deleted | Minimal changes that achieve the goal indicate precision |
| **Churn Rate** | How often agent-written code gets re-modified | High churn = low-quality initial solutions |

### 1.2 Defining "Success" by Task Type

Not all tasks are equal. A metric framework must define success differently per task type.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    TEST_WRITING = "test_writing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEVOPS = "devops"

@dataclass
class SuccessCriteria:
    """Defines what 'success' means for a task type."""
    task_type: TaskType
    primary_metric: str           # The metric that must pass
    primary_threshold: float       # Minimum value for primary metric
    secondary_metrics: dict[str, float]  # Additional metrics and thresholds
    timeout_seconds: int           # Max allowed time
    max_cost_usd: float           # Max allowed cost

SUCCESS_DEFINITIONS: dict[TaskType, SuccessCriteria] = {
    TaskType.CODE_GENERATION: SuccessCriteria(
        task_type=TaskType.CODE_GENERATION,
        primary_metric="test_pass_rate",
        primary_threshold=1.0,          # All tests must pass
        secondary_metrics={
            "lint_score": 0.9,          # 90%+ lint clean
            "complexity_per_function": 10.0,  # McCabe < 10
        },
        timeout_seconds=300,
        max_cost_usd=0.50,
    ),
    TaskType.BUG_FIX: SuccessCriteria(
        task_type=TaskType.BUG_FIX,
        primary_metric="regression_test_pass_rate",
        primary_threshold=1.0,
        secondary_metrics={
            "lines_changed": 50,        # Bug fixes should be minimal
            "original_test_pass": True,  # The failing test now passes
        },
        timeout_seconds=180,
        max_cost_usd=0.30,
    ),
    TaskType.REFACTOR: SuccessCriteria(
        task_type=TaskType.REFACTOR,
        primary_metric="test_pass_rate",
        primary_threshold=1.0,          # All existing tests still pass
        secondary_metrics={
            "complexity_delta": 0.0,    # Complexity must not increase
            "coverage_delta": 0.0,      # Coverage must not decrease
        },
        timeout_seconds=600,
        max_cost_usd=1.00,
    ),
}
```

### 1.3 Leading vs Lagging Indicators

**Lagging indicators** tell you what already happened. Task success rate, error rate, and cost per task are all lagging -- they are only available after a task completes.

**Leading indicators** predict future outcomes and enable proactive intervention:

- **Token velocity** (tokens/second during generation): Sudden drops may indicate the model is "stuck" on a difficult problem and will likely fail.
- **Tool call density in early steps**: If an agent makes many tool calls in the first 3 steps, it correlates with eventual success (it is exploring effectively). If it makes zero, it may be hallucinating a solution.
- **Error recovery rate in first retry**: If the agent successfully recovers from the first error, it usually completes the task. If it fails the first retry, success probability drops below 20%.
- **Prompt-response token ratio**: If the response is very short relative to the prompt, the model may have given up or produced a minimal answer.

Leading indicators enable **early termination** of tasks that are likely to fail, saving cost and time:

```python
def should_abort_task(metrics: dict) -> tuple[bool, str]:
    """Check leading indicators and abort if task is likely to fail."""
    # Stuck detection: no progress for 3+ iterations
    if metrics.get("consecutive_no_progress_steps", 0) >= 3:
        return True, "Agent stuck: 3 consecutive steps with no progress"

    # Cost runaway: already spent 80% of budget with < 50% completion
    if (metrics.get("cost_so_far", 0) > metrics.get("max_cost", 1.0) * 0.8
            and metrics.get("estimated_completion", 0) < 0.5):
        return True, "Cost runaway: high spend with low estimated completion"

    # Retry spiral: more than 5 retries on the same step
    if metrics.get("same_step_retry_count", 0) > 5:
        return True, "Retry spiral: too many retries on the same step"

    return False, ""
```

---

## 2. Metrics Collection Architecture

### 2.1 Structured Logging with JSONL

JSONL (JSON Lines) is the right format for agent metrics: one JSON object per line, append-only, trivially parseable, and naturally time-series.

```python
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

@dataclass
class AgentEvent:
    """A single event emitted by the agent system."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = ""            # "task_started", "task_completed", "llm_call", etc.
    task_id: str = ""
    agent_id: str = ""
    model: str = ""                 # "claude-sonnet-4-20250514", "gpt-4o", etc.
    data: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), default=str)


class MetricsLogger:
    """Append-only JSONL metrics logger with rotation support."""

    def __init__(self, base_dir: Path, max_file_size_mb: int = 100):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self._current_file: Optional[Path] = None

    def _get_log_file(self) -> Path:
        """Get current log file, rotating if needed."""
        if self._current_file and self._current_file.exists():
            if self._current_file.stat().st_size < self.max_file_size:
                return self._current_file

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Find the next available file number for today
        existing = list(self.base_dir.glob(f"metrics-{date_str}-*.jsonl"))
        file_num = len(existing)
        self._current_file = self.base_dir / f"metrics-{date_str}-{file_num:04d}.jsonl"
        return self._current_file

    def emit(self, event: AgentEvent) -> None:
        """Write a single event to the log."""
        log_file = self._get_log_file()
        with open(log_file, "a") as f:
            f.write(event.to_jsonl() + "\n")

    def emit_task_start(self, task_id: str, agent_id: str, task_type: str,
                        metadata: dict | None = None) -> None:
        self.emit(AgentEvent(
            event_type="task_started",
            task_id=task_id,
            agent_id=agent_id,
            data={"task_type": task_type, **(metadata or {})},
        ))

    def emit_llm_call(self, task_id: str, agent_id: str, model: str,
                      input_tokens: int, output_tokens: int,
                      cost_usd: float, latency_ms: float) -> None:
        self.emit(AgentEvent(
            event_type="llm_call",
            task_id=task_id,
            agent_id=agent_id,
            model=model,
            data={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
            },
        ))

    def emit_task_complete(self, task_id: str, agent_id: str,
                           success: bool, metrics: dict) -> None:
        self.emit(AgentEvent(
            event_type="task_completed",
            task_id=task_id,
            agent_id=agent_id,
            data={"success": success, **metrics},
        ))
```

### 2.2 Event Sourcing Pattern

The key insight from event sourcing: **store raw events, derive aggregates**. Never store only aggregates -- you cannot disaggregate, but you can always re-aggregate from raw events.

```
Raw Events (immutable, append-only)
    |
    v
Event Store (JSONL files, partitioned by date)
    |
    +---> Task-level aggregation  (success rate, cost, duration per task)
    +---> Agent-level aggregation (which agent performs best?)
    +---> Model-level aggregation (which model is most efficient?)
    +---> Time-series aggregation (hourly/daily/weekly trends)
    +---> Session aggregation    (multi-task session patterns)
```

### 2.3 Metric Schemas

A schema definition ensures consistent data across the system:

```python
METRIC_SCHEMAS = {
    "task_started": {
        "required": ["task_id", "agent_id", "task_type"],
        "optional": ["description", "parent_task_id", "priority"],
    },
    "llm_call": {
        "required": ["task_id", "model", "input_tokens", "output_tokens",
                      "cost_usd", "latency_ms"],
        "optional": ["cache_hit", "prompt_hash", "temperature", "tool_calls"],
    },
    "task_completed": {
        "required": ["task_id", "success", "duration_seconds", "total_cost_usd"],
        "optional": ["test_pass_rate", "error_message", "retry_count",
                      "lines_changed", "files_modified"],
    },
    "tool_call": {
        "required": ["task_id", "tool_name", "success"],
        "optional": ["duration_ms", "error", "input_size", "output_size"],
    },
    "error": {
        "required": ["task_id", "error_type", "error_message"],
        "optional": ["stack_trace", "recoverable", "retry_number"],
    },
}
```

### 2.4 Aggregation Engine

```python
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator
import json
import statistics

@dataclass
class TaskSummary:
    task_id: str
    task_type: str
    success: bool
    duration_seconds: float
    total_cost_usd: float
    total_tokens: int
    llm_calls: int
    tool_calls: int
    retries: int
    models_used: list[str]
    error_types: list[str]

class MetricsAggregator:
    """Aggregates raw events into queryable summaries."""

    def __init__(self, events: list[dict]):
        self.events = events
        self._task_events: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            task_id = event.get("task_id", "unknown")
            self._task_events[task_id].append(event)

    def task_summaries(self) -> list[TaskSummary]:
        """Produce one summary per task."""
        summaries = []
        for task_id, events in self._task_events.items():
            llm_calls = [e for e in events if e["event_type"] == "llm_call"]
            completions = [e for e in events if e["event_type"] == "task_completed"]
            errors = [e for e in events if e["event_type"] == "error"]
            tool_calls = [e for e in events if e["event_type"] == "tool_call"]

            if not completions:
                continue

            completion = completions[-1]  # Use last completion event
            summaries.append(TaskSummary(
                task_id=task_id,
                task_type=completion["data"].get("task_type", "unknown"),
                success=completion["data"]["success"],
                duration_seconds=completion["data"].get("duration_seconds", 0),
                total_cost_usd=sum(e["data"]["cost_usd"] for e in llm_calls),
                total_tokens=sum(
                    e["data"]["input_tokens"] + e["data"]["output_tokens"]
                    for e in llm_calls
                ),
                llm_calls=len(llm_calls),
                tool_calls=len(tool_calls),
                retries=completion["data"].get("retry_count", 0),
                models_used=list({e["model"] for e in llm_calls if e.get("model")}),
                error_types=[e["data"]["error_type"] for e in errors],
            ))
        return summaries

    def model_efficiency(self) -> dict[str, dict]:
        """Compute efficiency metrics per model."""
        model_data: dict[str, list] = defaultdict(list)
        for event in self.events:
            if event["event_type"] == "llm_call":
                model_data[event["model"]].append(event["data"])

        results = {}
        for model, calls in model_data.items():
            costs = [c["cost_usd"] for c in calls]
            latencies = [c["latency_ms"] for c in calls]
            results[model] = {
                "total_calls": len(calls),
                "total_cost": sum(costs),
                "avg_cost": statistics.mean(costs),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
                    if len(latencies) >= 20 else max(latencies),
                "total_tokens": sum(
                    c["input_tokens"] + c["output_tokens"] for c in calls
                ),
            }
        return results

    def hourly_trends(self) -> dict[str, dict]:
        """Aggregate metrics by hour for trend detection."""
        hourly: dict[str, list] = defaultdict(list)
        for event in self.events:
            if event["event_type"] == "task_completed":
                hour = event["timestamp"][:13]  # "2026-02-26T14"
                hourly[hour].append(event["data"])

        trends = {}
        for hour, tasks in sorted(hourly.items()):
            successes = sum(1 for t in tasks if t.get("success"))
            trends[hour] = {
                "total_tasks": len(tasks),
                "success_rate": successes / len(tasks) if tasks else 0,
                "avg_cost": statistics.mean(
                    [t.get("total_cost_usd", 0) for t in tasks]
                ),
                "avg_duration": statistics.mean(
                    [t.get("duration_seconds", 0) for t in tasks]
                ),
            }
        return trends
```

### 2.5 Storage Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **JSONL files** | Zero dependencies, append-only, easy to ship | No indexing, slow queries on large datasets | Early stage, < 100K events/day |
| **SQLite** | SQL queries, single file, no server | Write contention with multiple agents | Single-node, < 1M events/day |
| **PostgreSQL + TimescaleDB** | Time-series optimized, SQL, mature ecosystem | Operational overhead, needs a server | Production, > 1M events/day |
| **ClickHouse** | Extremely fast analytical queries, column-oriented | Overkill for small scale, learning curve | Large-scale analytics |
| **Prometheus + Grafana** | Industry standard for metrics, great dashboards | Not designed for event-level data | Real-time monitoring layer |

**Recommendation for aiai**: Start with JSONL files for raw event storage. Add SQLite for aggregated summaries and query access. Migrate to TimescaleDB when event volume exceeds what file-based processing handles comfortably. Use JSONL as the canonical format throughout -- even after adding a database, keep writing JSONL for audit trails and reprocessing.

---

## 3. Pattern Detection in Agent Metrics

### 3.1 Cost Outlier Detection

Detecting when a task costs significantly more than expected enables both real-time intervention and post-hoc analysis.

**Z-Score Method**: Simple, assumes roughly normal distribution.

```python
import statistics
import math

def detect_cost_outliers_zscore(
    costs: list[float],
    threshold: float = 2.0,
) -> list[tuple[int, float, float]]:
    """
    Detect cost outliers using z-scores.

    Returns: list of (index, cost, z_score) for outliers.
    """
    if len(costs) < 10:
        return []  # Not enough data for meaningful z-scores

    mean = statistics.mean(costs)
    stdev = statistics.stdev(costs)

    if stdev == 0:
        return []

    outliers = []
    for i, cost in enumerate(costs):
        z = (cost - mean) / stdev
        if abs(z) > threshold:
            outliers.append((i, cost, z))

    return outliers
```

**IQR Method**: More robust to skewed distributions (which cost data usually is).

```python
def detect_cost_outliers_iqr(
    costs: list[float],
    multiplier: float = 1.5,
) -> list[tuple[int, float]]:
    """
    Detect cost outliers using Interquartile Range.
    More robust than z-scores for skewed distributions.

    Returns: list of (index, cost) for outliers.
    """
    if len(costs) < 10:
        return []

    sorted_costs = sorted(costs)
    n = len(sorted_costs)
    q1 = sorted_costs[n // 4]
    q3 = sorted_costs[3 * n // 4]
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return [
        (i, cost) for i, cost in enumerate(costs)
        if cost < lower_bound or cost > upper_bound
    ]
```

### 3.2 Failure Pattern Clustering

When tasks fail, understanding *why* they fail enables targeted improvement. Failures often cluster into a small number of root causes.

```python
from collections import Counter
from difflib import SequenceMatcher

def cluster_failures(
    error_messages: list[str],
    similarity_threshold: float = 0.6,
) -> dict[str, list[int]]:
    """
    Cluster error messages by similarity.
    Returns: dict mapping cluster representative -> list of indices.
    """
    clusters: dict[str, list[int]] = {}

    for i, msg in enumerate(error_messages):
        # Find the most similar existing cluster
        best_match = None
        best_score = 0.0

        for representative in clusters:
            score = SequenceMatcher(None, msg, representative).ratio()
            if score > best_score:
                best_score = score
                best_match = representative

        if best_match and best_score >= similarity_threshold:
            clusters[best_match].append(i)
        else:
            clusters[msg] = [i]

    return clusters


def failure_report(task_summaries: list[dict]) -> dict:
    """
    Analyze failure patterns across tasks.
    Returns structured report of failure categories and frequencies.
    """
    failed = [t for t in task_summaries if not t["success"]]
    if not failed:
        return {"total_failures": 0, "patterns": []}

    # Cluster by error type
    error_type_counts = Counter(
        t.get("error_type", "unknown") for t in failed
    )

    # Cluster by error message similarity
    messages = [t.get("error_message", "") for t in failed if t.get("error_message")]
    message_clusters = cluster_failures(messages)

    # Analyze by task type
    failures_by_type = Counter(t.get("task_type", "unknown") for t in failed)

    return {
        "total_failures": len(failed),
        "failure_rate": len(failed) / len(task_summaries),
        "by_error_type": dict(error_type_counts.most_common(10)),
        "by_task_type": dict(failures_by_type.most_common(10)),
        "error_clusters": {
            rep: {"count": len(indices), "sample_indices": indices[:3]}
            for rep, indices in sorted(
                message_clusters.items(), key=lambda x: -len(x[1])
            )[:10]
        },
    }
```

### 3.3 Trend Detection

Is the system getting better or worse over time? Trend detection answers this.

```python
def linear_trend(values: list[float]) -> dict:
    """
    Compute linear trend using least-squares regression.
    Returns slope, direction, and statistical confidence.
    """
    n = len(values)
    if n < 5:
        return {"slope": 0, "direction": "insufficient_data", "r_squared": 0}

    # Simple least-squares linear regression
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return {"slope": 0, "direction": "flat", "r_squared": 0}

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # R-squared (coefficient of determination)
    ss_res = sum((y - (slope * i + intercept)) ** 2 for i, y in enumerate(values))
    ss_tot = sum((y - y_mean) ** 2 for y in values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Determine direction with confidence
    if r_squared < 0.1:
        direction = "no_clear_trend"
    elif slope > 0:
        direction = "improving" if slope > 0 else "degrading"
    else:
        direction = "degrading"

    return {
        "slope": slope,
        "direction": direction,
        "r_squared": r_squared,
        "trend_strength": "strong" if r_squared > 0.5 else
                          "moderate" if r_squared > 0.25 else "weak",
        "predicted_next": slope * n + intercept,
    }


def detect_regime_change(
    values: list[float],
    window_size: int = 10,
    threshold_stdevs: float = 2.0,
) -> list[dict]:
    """
    Detect points where behavior changes significantly.
    Uses a sliding window comparison.
    """
    if len(values) < window_size * 3:
        return []

    changes = []
    for i in range(window_size, len(values) - window_size):
        before = values[i - window_size:i]
        after = values[i:i + window_size]

        before_mean = statistics.mean(before)
        before_std = statistics.stdev(before) if len(before) > 1 else 1.0
        after_mean = statistics.mean(after)

        if before_std == 0:
            continue

        z = abs(after_mean - before_mean) / before_std
        if z > threshold_stdevs:
            changes.append({
                "index": i,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "z_score": z,
                "direction": "increase" if after_mean > before_mean else "decrease",
                "magnitude_pct": ((after_mean - before_mean) / before_mean * 100)
                    if before_mean != 0 else float("inf"),
            })

    return changes
```

### 3.4 Correlation Analysis

Which configuration changes correlate with metric changes?

```python
def pearson_correlation(x: list[float], y: list[float]) -> dict:
    """
    Compute Pearson correlation between two metric series.
    Useful for finding which factors correlate with success.
    """
    n = len(x)
    if n != len(y) or n < 5:
        return {"r": 0, "strength": "insufficient_data"}

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))

    if std_x == 0 or std_y == 0:
        return {"r": 0, "strength": "no_variance"}

    r = cov / (std_x * std_y)

    abs_r = abs(r)
    strength = (
        "strong" if abs_r > 0.7 else
        "moderate" if abs_r > 0.4 else
        "weak" if abs_r > 0.2 else
        "negligible"
    )

    return {
        "r": r,
        "r_squared": r ** 2,
        "strength": strength,
        "direction": "positive" if r > 0 else "negative",
    }


def find_metric_correlations(
    task_data: list[dict],
    target_metric: str,
    candidate_metrics: list[str],
) -> list[dict]:
    """
    Find which metrics correlate most strongly with a target metric.
    E.g., which factors correlate with task success?
    """
    target_values = [t[target_metric] for t in task_data if target_metric in t]

    correlations = []
    for metric in candidate_metrics:
        metric_values = [t[metric] for t in task_data if metric in t]
        # Align arrays (only use tasks that have both metrics)
        paired = [
            (t[target_metric], t[metric])
            for t in task_data
            if target_metric in t and metric in t
        ]
        if len(paired) < 10:
            continue

        x_vals, y_vals = zip(*paired)
        corr = pearson_correlation(list(x_vals), list(y_vals))
        corr["metric"] = metric
        correlations.append(corr)

    # Sort by absolute correlation strength
    correlations.sort(key=lambda c: abs(c.get("r", 0)), reverse=True)
    return correlations
```

---

## 4. Hypothesis Generation for Self-Improvement

### 4.1 From Observation to Hypothesis

The critical intellectual leap in a self-improving system is going from "we observe X" to "we hypothesize that changing Y will improve X." This is where LLMs are uniquely valuable -- they can reason about causal mechanisms, not just statistical patterns.

The pipeline:

```
Metrics Analysis  -->  Observations  -->  LLM Reasoning  -->  Hypotheses  -->  Ranking  -->  Implementation
```

### 4.2 Structured Observation Format

```python
@dataclass
class Observation:
    """A structured observation derived from metric analysis."""
    observation_id: str
    category: str           # "cost", "success_rate", "latency", "error_pattern"
    severity: str           # "critical", "high", "medium", "low"
    description: str        # Human-readable description
    evidence: dict          # Supporting data
    affected_metric: str    # Which metric is impacted
    trend_direction: str    # "improving", "degrading", "volatile", "stable"
    confidence: float       # 0.0-1.0 confidence in the observation


@dataclass
class Hypothesis:
    """A structured hypothesis for improvement."""
    hypothesis_id: str
    observation_id: str          # Which observation triggered this
    description: str             # What to change
    mechanism: str               # Why we think this will work
    expected_improvement: float  # Expected % improvement
    confidence: float            # 0.0-1.0 confidence it will work
    cost_to_test: float          # Cost in USD to run the experiment
    risk_level: str              # "safe", "moderate", "risky"
    implementation: str          # Concrete change specification
    rollback_plan: str           # How to undo if it fails
    expected_value: float = 0.0  # Computed: improvement * confidence / cost

    def compute_expected_value(self) -> float:
        """Rank hypotheses by expected value."""
        self.expected_value = (
            self.expected_improvement * self.confidence
            / max(self.cost_to_test, 0.01)
        )
        return self.expected_value
```

### 4.3 Using LLMs for Hypothesis Generation

```python
HYPOTHESIS_GENERATION_PROMPT = """You are analyzing metrics from an autonomous AI coding agent system.

## Current Observations

{observations}

## Recent Metric Summaries

Success rate (7d): {success_rate_7d}
Avg cost per task: ${avg_cost}
Avg duration: {avg_duration}s
Top error types: {top_errors}
Model distribution: {model_distribution}

## System Configuration

Current prompt version: {prompt_version}
Current model routing: {model_routing}
Current tool set: {tool_set}

## Task

Generate 3-5 specific, actionable hypotheses for improving the system.
For each hypothesis, provide:

1. **What to change**: A specific, implementable change
2. **Why it should work**: The causal mechanism
3. **Expected improvement**: Quantified estimate (e.g., "reduce cost by 15%")
4. **Confidence**: How confident you are (0.0-1.0)
5. **Risk**: What could go wrong
6. **How to test**: Concrete experiment design

Focus on high-leverage changes. Prefer changes that are:
- Low risk (easily reversible)
- High expected improvement
- Cheap to test
- Supported by evidence in the observations

Output as JSON array of hypothesis objects.
"""

async def generate_hypotheses(
    observations: list[Observation],
    metrics_summary: dict,
    system_config: dict,
    llm_client,  # Your LLM client
) -> list[Hypothesis]:
    """Use an LLM to generate improvement hypotheses from observations."""
    prompt = HYPOTHESIS_GENERATION_PROMPT.format(
        observations="\n".join(
            f"- [{o.severity}] {o.description} (confidence: {o.confidence})"
            for o in observations
        ),
        success_rate_7d=metrics_summary.get("success_rate_7d", "N/A"),
        avg_cost=metrics_summary.get("avg_cost", "N/A"),
        avg_duration=metrics_summary.get("avg_duration", "N/A"),
        top_errors=metrics_summary.get("top_errors", []),
        model_distribution=metrics_summary.get("model_distribution", {}),
        prompt_version=system_config.get("prompt_version", "unknown"),
        model_routing=system_config.get("model_routing", "unknown"),
        tool_set=system_config.get("tool_set", []),
    )

    response = await llm_client.generate(prompt, response_format="json")
    raw_hypotheses = json.loads(response)

    hypotheses = []
    for h in raw_hypotheses:
        hypothesis = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            observation_id=observations[0].observation_id if observations else "",
            description=h["what_to_change"],
            mechanism=h["why_it_should_work"],
            expected_improvement=h["expected_improvement_pct"] / 100,
            confidence=h["confidence"],
            cost_to_test=h.get("test_cost_usd", 1.0),
            risk_level=h.get("risk", "moderate"),
            implementation=h.get("implementation", ""),
            rollback_plan=h.get("rollback", "Revert to previous configuration"),
        )
        hypothesis.compute_expected_value()
        hypotheses.append(hypothesis)

    # Rank by expected value
    hypotheses.sort(key=lambda h: h.expected_value, reverse=True)
    return hypotheses
```

### 4.4 Hypothesis Validation Before Implementation

Not every hypothesis should be implemented. Pre-implementation validation filters out bad ideas:

```python
class HypothesisValidator:
    """Validates hypotheses before they are implemented."""

    def __init__(self, history: list[Hypothesis], config: dict):
        self.history = history
        self.config = config

    def validate(self, hypothesis: Hypothesis) -> tuple[bool, list[str]]:
        """
        Validate a hypothesis. Returns (is_valid, list_of_reasons).
        """
        reasons = []

        # 1. Check for duplicate/similar past hypotheses that failed
        for past in self.history:
            if (self._similarity(hypothesis.description, past.description) > 0.8
                    and past.expected_value < 0):
                reasons.append(
                    f"Similar hypothesis '{past.description}' was tried and failed"
                )

        # 2. Check risk level against current stability
        if hypothesis.risk_level == "risky" and self.config.get("stability_mode"):
            reasons.append("System is in stability mode; risky changes are blocked")

        # 3. Check cost to test against budget
        test_budget = self.config.get("experiment_budget_usd", 10.0)
        if hypothesis.cost_to_test > test_budget:
            reasons.append(
                f"Test cost ${hypothesis.cost_to_test} exceeds budget ${test_budget}"
            )

        # 4. Check confidence threshold
        min_confidence = self.config.get("min_hypothesis_confidence", 0.3)
        if hypothesis.confidence < min_confidence:
            reasons.append(
                f"Confidence {hypothesis.confidence} below minimum {min_confidence}"
            )

        # 5. Check expected value threshold
        min_ev = self.config.get("min_expected_value", 0.1)
        if hypothesis.expected_value < min_ev:
            reasons.append(
                f"Expected value {hypothesis.expected_value} below minimum {min_ev}"
            )

        return len(reasons) == 0, reasons

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
```

---

## 5. A/B Testing for Autonomous Systems

### 5.1 The Challenge

Traditional A/B testing assumes you can randomly assign users to treatment and control groups and run both simultaneously. Autonomous agent systems break this assumption:

- Tasks are not identical -- each task has different difficulty, so direct comparison is noisy.
- You often cannot run two agent configurations simultaneously on the same task.
- Sample sizes are small -- you might only run 50-100 tasks per day.
- The system is non-stationary -- external factors (API latency, model updates) change over time.

### 5.2 Sequential Testing

Instead of fixed-sample-size tests, use sequential testing that can reach conclusions with fewer samples.

```python
import math

class SequentialABTest:
    """
    Sequential Probability Ratio Test (SPRT) for comparing
    two agent configurations.

    Allows early stopping when there is enough evidence,
    requiring 20-80% fewer samples than fixed-sample tests.
    """

    def __init__(
        self,
        alpha: float = 0.05,    # Type I error rate (false positive)
        beta: float = 0.20,     # Type II error rate (false negative)
        delta: float = 0.05,    # Minimum detectable effect (5%)
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.log_a = math.log(beta / (1 - alpha))          # Lower boundary
        self.log_b = math.log((1 - beta) / alpha)           # Upper boundary
        self.log_likelihood_ratio = 0.0
        self.n_a = 0  # Successes in A
        self.n_b = 0  # Successes in B
        self.total_a = 0  # Total trials in A
        self.total_b = 0  # Total trials in B

    def add_observation(self, variant: str, success: bool) -> dict:
        """
        Add a single observation and check if we can conclude.
        variant: 'A' (control) or 'B' (treatment)
        """
        if variant == "A":
            self.total_a += 1
            if success:
                self.n_a += 1
        else:
            self.total_b += 1
            if success:
                self.n_b += 1

        # Need minimum observations in each group
        if self.total_a < 5 or self.total_b < 5:
            return {"decision": "continue", "reason": "insufficient_data"}

        p_a = self.n_a / self.total_a
        p_b = self.n_b / self.total_b

        # Compute log-likelihood ratio for this observation
        # Under H1 (B is better by delta), vs H0 (A and B are equal)
        p_0 = (self.n_a + self.n_b) / (self.total_a + self.total_b)
        p_1 = p_0 + self.delta

        if 0 < p_0 < 1 and 0 < p_1 < 1:
            if success and variant == "B":
                self.log_likelihood_ratio += math.log(p_1 / p_0)
            elif not success and variant == "B":
                self.log_likelihood_ratio += math.log((1 - p_1) / (1 - p_0))

        # Check boundaries
        if self.log_likelihood_ratio >= self.log_b:
            return {
                "decision": "B_wins",
                "p_a": p_a,
                "p_b": p_b,
                "improvement": (p_b - p_a) / p_a if p_a > 0 else float("inf"),
                "total_samples": self.total_a + self.total_b,
            }
        elif self.log_likelihood_ratio <= self.log_a:
            return {
                "decision": "A_wins",
                "p_a": p_a,
                "p_b": p_b,
                "total_samples": self.total_a + self.total_b,
            }
        else:
            return {
                "decision": "continue",
                "log_likelihood_ratio": self.log_likelihood_ratio,
                "boundaries": (self.log_a, self.log_b),
                "p_a": p_a,
                "p_b": p_b,
                "total_samples": self.total_a + self.total_b,
            }
```

### 5.3 Bayesian Comparison

When sample sizes are small, Bayesian methods provide richer information than frequentist tests.

```python
import random

class BayesianABComparison:
    """
    Bayesian comparison of two agent configurations.
    Uses Beta-Binomial model for success rates.

    Provides probability that B is better than A,
    rather than just a binary significant/not-significant answer.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        prior_alpha, prior_beta: Beta distribution prior parameters.
        (1, 1) = uniform prior (no prior knowledge).
        Use historical data to set informative priors.
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def probability_b_better(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        n_simulations: int = 100_000,
    ) -> dict:
        """
        Compute probability that B has a higher success rate than A
        using Monte Carlo simulation from posterior distributions.
        """
        alpha_a = self.prior_alpha + successes_a
        beta_a = self.prior_beta + (trials_a - successes_a)
        alpha_b = self.prior_alpha + successes_b
        beta_b = self.prior_beta + (trials_b - successes_b)

        # Monte Carlo: sample from both posteriors and compare
        b_wins = 0
        improvements = []

        for _ in range(n_simulations):
            sample_a = random.betavariate(alpha_a, beta_a)
            sample_b = random.betavariate(alpha_b, beta_b)
            if sample_b > sample_a:
                b_wins += 1
                if sample_a > 0:
                    improvements.append((sample_b - sample_a) / sample_a)

        prob_b_better = b_wins / n_simulations

        return {
            "prob_b_better": prob_b_better,
            "prob_a_better": 1 - prob_b_better,
            "posterior_mean_a": alpha_a / (alpha_a + beta_a),
            "posterior_mean_b": alpha_b / (alpha_b + beta_b),
            "expected_improvement": (
                statistics.mean(improvements) if improvements else 0
            ),
            "recommendation": (
                "adopt_b" if prob_b_better > 0.95 else
                "reject_b" if prob_b_better < 0.05 else
                "continue_testing"
            ),
        }

    def cost_comparison(
        self,
        costs_a: list[float],
        costs_b: list[float],
        n_simulations: int = 50_000,
    ) -> dict:
        """
        Compare cost distributions using bootstrap resampling.
        Appropriate for continuous metrics like cost and latency.
        """
        if len(costs_a) < 5 or len(costs_b) < 5:
            return {"decision": "insufficient_data"}

        b_cheaper = 0
        savings = []

        for _ in range(n_simulations):
            # Bootstrap sample means
            sample_a = statistics.mean(random.choices(costs_a, k=len(costs_a)))
            sample_b = statistics.mean(random.choices(costs_b, k=len(costs_b)))

            if sample_b < sample_a:
                b_cheaper += 1
                savings.append(sample_a - sample_b)

        prob_b_cheaper = b_cheaper / n_simulations

        return {
            "prob_b_cheaper": prob_b_cheaper,
            "mean_cost_a": statistics.mean(costs_a),
            "mean_cost_b": statistics.mean(costs_b),
            "expected_savings_per_task": (
                statistics.mean(savings) if savings else 0
            ),
            "recommendation": (
                "adopt_b" if prob_b_cheaper > 0.90 else
                "reject_b" if prob_b_cheaper < 0.10 else
                "continue_testing"
            ),
        }
```

### 5.4 Practical Experiment Framework

```python
@dataclass
class Experiment:
    """An A/B experiment comparing two agent configurations."""
    experiment_id: str
    name: str
    hypothesis_id: str
    config_a: dict              # Control configuration
    config_b: dict              # Treatment configuration
    target_metric: str          # Primary metric to compare
    secondary_metrics: list[str]
    min_samples: int = 20       # Minimum before any conclusion
    max_samples: int = 200      # Maximum before forced conclusion
    status: str = "running"     # "running", "concluded", "aborted"
    results_a: list[float] = field(default_factory=list)
    results_b: list[float] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""


class ExperimentRunner:
    """Manages the lifecycle of A/B experiments."""

    def __init__(self):
        self.active_experiments: dict[str, Experiment] = {}
        self.completed_experiments: list[Experiment] = []
        self.bayesian = BayesianABComparison()

    def assign_variant(self, experiment_id: str, task_id: str) -> str:
        """Assign a task to A or B using alternating assignment."""
        exp = self.active_experiments[experiment_id]
        total = len(exp.results_a) + len(exp.results_b)
        return "A" if total % 2 == 0 else "B"

    def record_result(self, experiment_id: str, variant: str,
                      metric_value: float) -> dict | None:
        """Record a result and check if experiment can conclude."""
        exp = self.active_experiments[experiment_id]

        if variant == "A":
            exp.results_a.append(metric_value)
        else:
            exp.results_b.append(metric_value)

        total = len(exp.results_a) + len(exp.results_b)

        # Check if we have enough data to conclude
        if total >= exp.min_samples:
            comparison = self.bayesian.cost_comparison(
                exp.results_a, exp.results_b
            ) if exp.target_metric in ("cost", "duration", "latency") else {
                "recommendation": "continue_testing"
            }

            if comparison["recommendation"] != "continue_testing":
                exp.status = "concluded"
                exp.end_time = datetime.now(timezone.utc).isoformat()
                self.completed_experiments.append(exp)
                del self.active_experiments[experiment_id]
                return comparison

        if total >= exp.max_samples:
            exp.status = "concluded"
            exp.end_time = datetime.now(timezone.utc).isoformat()
            self.completed_experiments.append(exp)
            del self.active_experiments[experiment_id]
            return {"recommendation": "inconclusive", "reason": "max_samples_reached"}

        return None  # Experiment continues
```

---

## 6. AlphaEvolve Deep Dive

### 6.1 Architecture Overview

[AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, May 2025) is an evolutionary coding agent that pairs LLM-based code generation with automated evaluation in a closed loop. It represents the most successful application of evolutionary search with LLMs to date.

The system has four tightly coupled components:

```
+-------------------+       +-------------------+
|  Prompt Sampler   | ----> |  LLM Ensemble     |
|  (context-rich    |       |  Gemini Flash:     |
|   prompts from    |       |    breadth         |
|   top programs)   |       |  Gemini Pro:       |
+--------+----------+       |    depth           |
         ^                  +--------+----------+
         |                           |
         |                           v
+--------+----------+       +-------------------+
|  Programs         |       |  Evaluator Pool   |
|  Database         | <---- |  (automated        |
|  (evolutionary    |       |   scoring)         |
|   selection)      |       +-------------------+
+-------------------+
```

#### Prompt Sampler

Constructs context-rich prompts by combining the problem specification (natural language + code skeleton), a selection of high-performing programs from the database, and their evaluation scores. The sampler biases toward higher-scoring programs but maintains diversity to avoid premature convergence.

#### LLM Ensemble (Mutation Operators)

AlphaEvolve uses two Gemini models as complementary mutation operators:

- **Gemini Flash**: High throughput generation of diverse algorithmic variants. Maximizes breadth of exploration.
- **Gemini Pro**: Fewer but deeper, more insightful mutations. Provides critical depth.

The LLMs propose **diff-based code changes** rather than rewriting entire programs. This targeted mutation approach is more sample-efficient.

#### Evaluator Pool

Verifies, executes, and scores proposed programs using automated metrics. This is the critical constraint: AlphaEvolve requires a programmatically evaluable fitness function. The evaluator provides objective, quantifiable scores that drive selection pressure.

#### Programs Database

Stores all evaluated programs and their scores. Implements the evolutionary algorithm's selection mechanism. High-scoring programs are preferentially sampled for future prompts, but diversity is maintained.

### 6.2 Key Results

| Achievement | Details |
|------------|---------|
| **Gemini training speedup** | 23% speedup of a vital kernel, leading to 1% overall Gemini training time reduction |
| **FlashAttention optimization** | Up to 32.5% speedup for FlashAttention kernel implementation |
| **Matrix multiplication** | Found algorithm to multiply 4x4 complex matrices with 48 scalar multiplications, improving on Strassen's 1969 algorithm |
| **Data center scheduling** | Developed more efficient scheduling algorithm for Google data centers |
| **Hardware design** | Found functionally equivalent simplification in circuit design for hardware accelerators |

The recursive aspect is notable: AlphaEvolve optimized a kernel used in training the Gemini model that powers AlphaEvolve itself.

### 6.3 What aiai Can Learn from AlphaEvolve

The key lessons for aiai:

1. **Evaluation is everything.** AlphaEvolve works because it has automated evaluators that provide unambiguous fitness signals. For aiai, this means investing heavily in automated evaluation: test suites, linters, benchmarks, cost tracking.

2. **Diff-based mutations are better than full rewrites.** Small, targeted changes are easier to evaluate, easier to roll back, and produce more interpretable improvements.

3. **Ensemble models for exploration vs exploitation.** Using a cheap/fast model for breadth and an expensive/smart model for depth is more effective than using one model for everything. This maps directly to aiai's model routing.

4. **Maintain population diversity.** Do not just keep the single best solution -- keep a diverse population. The second-best solution might be one mutation away from the global optimum.

5. **The evolutionary loop is the right abstraction.** Generate, evaluate, select, repeat. This is the core loop for self-improvement.

### 6.4 OpenEvolve: Open-Source Implementation

[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) is an open-source implementation of AlphaEvolve's architecture. Key implementation details:

- **Island-based evolution**: Multiple isolated populations prevent premature convergence. Each island evolves independently with periodic migration of top performers.
- **Quality-diversity evolution**: Maintains diverse populations across feature dimensions, not just fitness.
- **LLM-agnostic**: Uses the OpenAI SDK, compatible with any provider (OpenAI, Anthropic, Google, local models via Ollama/vLLM).
- **Asynchronous pipeline**: Controller coordinates LLMs, evaluators, and databases in a non-blocking pipeline.

---

## 7. Prompt Evolution Techniques

### 7.1 The Problem

Prompts are the most leverage point in an AI agent system. A 10% improvement in a system prompt can cascade into 30-50% improvements in downstream task success. But optimizing prompts manually is slow, subjective, and does not scale.

### 7.2 DSPy: Programming, Not Prompting

[DSPy](https://dspy.ai/) (Stanford NLP) reframes prompt engineering as programming. Instead of writing prompts, you define input/output signatures and let the framework optimize the prompt automatically.

Core concept: specify what you want, let the optimizer figure out how to ask for it.

Key optimizers in DSPy (2025):

| Optimizer | Approach | Best For |
|-----------|----------|----------|
| **MIPROv2** | Generates instructions + few-shot examples using Bayesian Optimization | Production systems, general purpose |
| **COPRO** | Coordinate ascent over instruction space | Simple pipelines |
| **SIMBA** | Stochastic mini-batch sampling, identifies hard examples, generates self-reflective improvement rules | Improving on difficult edge cases |
| **GEPA** | LLM reflects on program trajectory, identifies what worked and what failed | Complex multi-step pipelines |

**Lesson for aiai**: DSPy's key insight is that prompt optimization should be automated and metric-driven. aiai should not be editing its own prompts based on vibes -- it should have an automated optimization loop with measurable outcomes.

### 7.3 APE: Automatic Prompt Engineer

[APE](https://arxiv.org/abs/2211.01910) (Zhou et al., ICLR 2023) treats prompt optimization as black-box optimization:

1. **Generate**: Use an LLM to propose many candidate instructions for a task.
2. **Evaluate**: Score each candidate on a held-out evaluation set.
3. **Select**: Keep the best-performing instruction.
4. **Refine**: Use iterative Monte Carlo search to further improve.

Key result: On 24 NLP tasks, APE-generated instructions outperformed human-written instructions on 19/24 tasks.

### 7.4 EvoPrompt: Evolutionary Prompt Optimization

[EvoPrompt](https://arxiv.org/abs/2309.08532) (ICLR 2024) explicitly applies evolutionary algorithms to prompt optimization:

```
Population of prompts
    |
    v
Evaluate each prompt on dev set --> Fitness scores
    |
    v
Selection: Keep top-k prompts
    |
    v
Crossover: Combine parts of two good prompts  (LLM-mediated)
    |
    v
Mutation: LLM modifies a prompt to create variant
    |
    v
New population --> repeat
```

EvoPrompt achieves up to 25% improvement over human-written prompts on BIG-Bench Hard tasks. The key innovation is using LLMs as the crossover and mutation operators -- the LLM understands natural language, so it can combine and modify prompts intelligently rather than doing random character-level mutations.

### 7.5 Evolving CLAUDE.md and Agent Instructions

For aiai, the most impactful prompt evolution target is the system instructions (CLAUDE.md and equivalent agent prompts). Here is a concrete approach:

```python
@dataclass
class PromptVariant:
    """A versioned prompt variant with its performance data."""
    variant_id: str
    parent_id: str | None       # Which variant this was derived from
    prompt_text: str
    generation: int             # Evolutionary generation number
    mutation_type: str          # "crossover", "mutation", "manual"
    fitness_score: float = 0.0
    tasks_evaluated: int = 0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    created_at: str = ""


class PromptEvolver:
    """
    Evolves agent system prompts using evolutionary optimization.
    """

    def __init__(self, population_size: int = 8, elite_count: int = 2):
        self.population_size = population_size
        self.elite_count = elite_count
        self.population: list[PromptVariant] = []
        self.generation = 0

    async def initialize(self, seed_prompt: str, llm_client) -> None:
        """Create initial population from a seed prompt."""
        self.population = [
            PromptVariant(
                variant_id=str(uuid.uuid4()),
                parent_id=None,
                prompt_text=seed_prompt,
                generation=0,
                mutation_type="seed",
            )
        ]

        # Generate initial variants
        for i in range(self.population_size - 1):
            mutated = await self._mutate(seed_prompt, llm_client)
            self.population.append(PromptVariant(
                variant_id=str(uuid.uuid4()),
                parent_id=self.population[0].variant_id,
                prompt_text=mutated,
                generation=0,
                mutation_type="initial_mutation",
            ))

    async def _mutate(self, prompt: str, llm_client) -> str:
        """Use LLM to create a mutation of a prompt."""
        mutation_prompt = f"""You are optimizing a system prompt for an AI coding agent.

Current prompt:
---
{prompt}
---

Generate a variation of this prompt that might perform better. You can:
- Reword instructions for clarity
- Add specific examples or constraints
- Reorder sections by priority
- Add error-handling instructions
- Remove redundant or confusing instructions

Maintain the same overall intent but try to improve effectiveness.
Output only the new prompt text, nothing else."""

        return await llm_client.generate(mutation_prompt)

    async def _crossover(self, prompt_a: str, prompt_b: str,
                         llm_client) -> str:
        """Combine two prompts using LLM-mediated crossover."""
        crossover_prompt = f"""You are combining two system prompts for an AI coding agent.
Take the best elements from each and create a combined prompt.

Prompt A:
---
{prompt_a}
---

Prompt B:
---
{prompt_b}
---

Create a new prompt that combines the strongest elements of both.
Output only the new prompt text, nothing else."""

        return await llm_client.generate(crossover_prompt)

    async def evolve_generation(self, llm_client) -> list[PromptVariant]:
        """Run one generation of evolution."""
        self.generation += 1

        # Sort by fitness
        self.population.sort(key=lambda p: p.fitness_score, reverse=True)

        # Keep elites
        new_population = self.population[:self.elite_count]

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < 0.3:
                # Crossover: combine two parents
                parent_a = random.choice(self.population[:4])
                parent_b = random.choice(self.population[:4])
                child_text = await self._crossover(
                    parent_a.prompt_text, parent_b.prompt_text, llm_client
                )
                mutation_type = "crossover"
                parent_id = parent_a.variant_id
            else:
                # Mutation: modify a single parent
                parent = random.choice(self.population[:4])
                child_text = await self._mutate(parent.prompt_text, llm_client)
                mutation_type = "mutation"
                parent_id = parent.variant_id

            new_population.append(PromptVariant(
                variant_id=str(uuid.uuid4()),
                parent_id=parent_id,
                prompt_text=child_text,
                generation=self.generation,
                mutation_type=mutation_type,
            ))

        self.population = new_population
        return self.population
```

### 7.6 Prompt Versioning

Every prompt change must be versioned, traceable, and reversible:

```python
@dataclass
class PromptVersion:
    version: str                  # Semantic version "1.2.3"
    prompt_hash: str              # SHA-256 of prompt text
    prompt_text: str
    parent_version: str | None
    change_description: str
    performance_metrics: dict     # Metrics at time of promotion
    created_at: str
    promoted_at: str | None = None  # When it became the active prompt
    retired_at: str | None = None   # When it was replaced


class PromptRegistry:
    """Version control for prompts."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.versions: list[PromptVersion] = []
        self.active_version: PromptVersion | None = None

    def register(self, prompt_text: str, change_description: str,
                 parent_version: str | None = None) -> PromptVersion:
        """Register a new prompt version."""
        import hashlib
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        # Auto-increment version
        if self.versions:
            last = self.versions[-1].version.split(".")
            new_version = f"{last[0]}.{last[1]}.{int(last[2]) + 1}"
        else:
            new_version = "1.0.0"

        version = PromptVersion(
            version=new_version,
            prompt_hash=prompt_hash,
            prompt_text=prompt_text,
            parent_version=parent_version,
            change_description=change_description,
            performance_metrics={},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.versions.append(version)
        return version

    def promote(self, version_str: str, metrics: dict) -> None:
        """Promote a version to active (after passing evaluation)."""
        for v in self.versions:
            if v.version == version_str:
                if self.active_version:
                    self.active_version.retired_at = (
                        datetime.now(timezone.utc).isoformat()
                    )
                v.promoted_at = datetime.now(timezone.utc).isoformat()
                v.performance_metrics = metrics
                self.active_version = v
                return
        raise ValueError(f"Version {version_str} not found")

    def rollback(self) -> PromptVersion | None:
        """Rollback to the previous active version."""
        promoted = [v for v in self.versions if v.promoted_at and v != self.active_version]
        if not promoted:
            return None
        previous = promoted[-1]
        if self.active_version:
            self.active_version.retired_at = datetime.now(timezone.utc).isoformat()
        previous.retired_at = None
        self.active_version = previous
        return previous
```

---

## 8. Tool Creation by AI

### 8.1 The Problem

An autonomous agent will encounter situations where its existing tools are insufficient. A human developer would write a new utility function. An autonomous agent should be able to do the same.

Two complementary approaches exist: **tool discovery** (finding an existing tool that solves the problem) and **tool creation** (writing a new tool when none exists).

### 8.2 FunSearch's Approach

[FunSearch](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) (DeepMind, published in Nature) searches for new solutions by evolving functions. The key insight: LLMs can be used as mutation operators in evolutionary search over the space of programs.

FunSearch's loop:
1. Show the LLM a selection of best programs found so far.
2. Ask it to generate an even better one.
3. Automatically execute and evaluate the proposed program.
4. Add the best programs to the database for future selection.

What makes FunSearch powerful is that it outputs **programs** rather than just answers. The programs reveal *how* the solutions are constructed, making them interpretable and reusable.

### 8.3 Voyager's Skill Library

[Voyager](https://voyager.minedojo.org/) (NVIDIA Research) maintains an ever-growing skill library of executable code. Each skill is:

- **Temporally extended**: A skill is a complete, reusable function, not a single action.
- **Interpretable**: Skills are readable code with docstrings.
- **Compositional**: Complex skills are built from simpler ones, compounding capability over time.

The skill library architecture:

```
Task requirement
    |
    v
Skill retrieval (embedding similarity search)
    |
    +--> Found matching skill? --> Execute it
    |
    +--> No matching skill? --> Generate new skill
                                    |
                                    v
                              Execute and verify
                                    |
                              +--> Success? --> Add to skill library
                              |
                              +--> Failure? --> Refine with error feedback
```

Key result: Voyager obtains 3.3x more unique items and unlocks milestones up to 15.3x faster than baselines, demonstrating that a growing skill library dramatically accelerates capability acquisition.

### 8.4 Safe Tool Creation for aiai

Creating new tools at runtime introduces risk. A poorly written tool can corrupt data, consume resources, or introduce security vulnerabilities. Here is a safe tool creation pipeline:

```python
@dataclass
class ToolSpec:
    """Specification for a new tool."""
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    code: str
    test_cases: list[dict]
    safety_constraints: list[str]
    created_by: str             # agent_id that created it
    version: str = "0.1.0"
    status: str = "proposed"    # "proposed", "testing", "approved", "active", "retired"


class ToolFactory:
    """Creates, validates, and registers new tools safely."""

    def __init__(self, tool_dir: Path, sandbox_timeout: int = 30):
        self.tool_dir = tool_dir
        self.sandbox_timeout = sandbox_timeout
        self.registry: dict[str, ToolSpec] = {}

    async def propose_tool(self, need_description: str,
                           llm_client) -> ToolSpec:
        """Use LLM to generate a tool specification from a need."""
        prompt = f"""An autonomous AI coding agent needs a new tool.

Need: {need_description}

Generate a Python tool with:
1. A clear function signature with type hints
2. A docstring explaining what it does
3. Input validation
4. Error handling (no bare except clauses)
5. At least 3 test cases

The tool must:
- Be a pure function (no side effects outside its explicit outputs)
- Not access the network unless explicitly needed
- Not modify the filesystem outside of designated directories
- Not execute arbitrary shell commands
- Have a timeout mechanism for long-running operations

Output as JSON with keys: name, description, input_schema, output_schema,
code, test_cases, safety_constraints"""

        response = await llm_client.generate(prompt, response_format="json")
        spec_data = json.loads(response)
        return ToolSpec(**spec_data, created_by="tool_factory")

    def validate_tool(self, spec: ToolSpec) -> tuple[bool, list[str]]:
        """
        Static validation of a tool before execution.
        Checks for safety violations and code quality.
        """
        issues = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("import subprocess", "Direct subprocess access not allowed"),
            ("os.system(", "os.system calls not allowed"),
            ("eval(", "eval() not allowed"),
            ("exec(", "exec() not allowed"),
            ("__import__", "Dynamic imports not allowed"),
            ("open(", "Direct file access -- must use approved I/O helpers"),
            ("requests.", "Direct HTTP requests -- must use approved HTTP client"),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in spec.code:
                issues.append(f"Safety violation: {reason}")

        # Check for test cases
        if len(spec.test_cases) < 3:
            issues.append("Insufficient test cases (minimum 3 required)")

        # Check for type hints
        if "def " in spec.code and "->" not in spec.code:
            issues.append("Missing return type annotation")

        # Check for docstring
        if '"""' not in spec.code and "'''" not in spec.code:
            issues.append("Missing docstring")

        return len(issues) == 0, issues

    async def test_tool(self, spec: ToolSpec) -> dict:
        """
        Execute tool's test cases in a sandboxed environment.
        """
        results = {
            "total_tests": len(spec.test_cases),
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        for i, test in enumerate(spec.test_cases):
            try:
                # Create isolated namespace for execution
                namespace: dict = {}
                exec(spec.code, namespace)

                # Find the main function
                func_name = spec.name
                if func_name not in namespace:
                    results["errors"].append(
                        f"Test {i}: Function '{func_name}' not found"
                    )
                    results["failed"] += 1
                    continue

                func = namespace[func_name]
                result = func(**test["input"])

                if result == test["expected_output"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(
                        f"Test {i}: Expected {test['expected_output']}, "
                        f"got {result}"
                    )

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Test {i}: {type(e).__name__}: {e}")

        results["pass_rate"] = (
            results["passed"] / results["total_tests"]
            if results["total_tests"] > 0 else 0
        )
        return results

    def register_tool(self, spec: ToolSpec) -> bool:
        """Register an approved tool in the registry."""
        if spec.status != "approved":
            return False
        self.registry[spec.name] = spec
        # Write tool to disk
        tool_file = self.tool_dir / f"{spec.name}.py"
        tool_file.write_text(spec.code)
        return True
```

### 8.5 Tool Evolution

Tools, like prompts, can be evolved over time:

```
Tool v1 (initial creation)
    |
    v
Observe: tool takes 2s, could be faster
    |
    v
Generate: LLM creates optimized version
    |
    v
Test: Run same test suite on v2
    |
    v
Compare: v2 passes all tests and is 3x faster
    |
    v
Promote: v2 replaces v1, v1 is kept for rollback
```

---

## 9. Evolution Engine Architecture Patterns

### 9.1 OODA Loop (Observe-Orient-Decide-Act)

The [OODA loop](https://developer.nvidia.com/blog/optimizing-data-center-performance-with-ai-agents-and-the-ooda-loop-strategy/), originally developed by military strategist John Boyd, maps naturally to self-improving systems:

```
OBSERVE     Collect metrics, read logs, monitor outcomes
    |
    v
ORIENT      Analyze patterns, compare to baselines, contextualize
    |        (This is the critical step -- understanding what the data means)
    v
DECIDE      Generate hypotheses, rank by expected value, select experiment
    |
    v
ACT         Implement change, run experiment, collect new data
    |
    +-----> back to OBSERVE
```

**Strengths**: Fast iteration, emphasis on orientation (the hardest part), well-suited for continuous improvement.

**Weaknesses**: Can get stuck in local optima if orientation is biased, no explicit mechanism for diversity.

```python
class OODALoop:
    """OODA-based self-improvement engine."""

    def __init__(self, metrics_store, hypothesis_generator, experiment_runner):
        self.metrics = metrics_store
        self.generator = hypothesis_generator
        self.runner = experiment_runner
        self.cycle_count = 0

    async def observe(self) -> dict:
        """Collect current system state and recent metrics."""
        return {
            "success_rate": self.metrics.success_rate(window="24h"),
            "cost_trend": self.metrics.cost_trend(window="7d"),
            "error_patterns": self.metrics.error_clusters(window="24h"),
            "model_efficiency": self.metrics.model_efficiency(window="7d"),
            "active_experiments": self.runner.active_count(),
        }

    async def orient(self, observations: dict) -> list:
        """Analyze observations and identify opportunities."""
        opportunities = []

        # Check for degradation
        if observations["success_rate"] < 0.85:
            opportunities.append({
                "type": "degradation",
                "severity": "high",
                "metric": "success_rate",
                "value": observations["success_rate"],
            })

        # Check for cost anomalies
        cost_trend = observations["cost_trend"]
        if cost_trend.get("direction") == "increasing":
            opportunities.append({
                "type": "cost_increase",
                "severity": "medium",
                "metric": "cost_per_task",
                "slope": cost_trend["slope"],
            })

        # Check for recurring errors
        for cluster, count in observations["error_patterns"].items():
            if count > 5:
                opportunities.append({
                    "type": "recurring_error",
                    "severity": "high",
                    "error_pattern": cluster,
                    "count": count,
                })

        return opportunities

    async def decide(self, opportunities: list) -> Hypothesis | None:
        """Generate and select the best hypothesis to test."""
        if not opportunities:
            return None

        hypotheses = await self.generator.generate(opportunities)
        if not hypotheses:
            return None

        # Select highest expected value hypothesis
        return max(hypotheses, key=lambda h: h.expected_value)

    async def act(self, hypothesis: Hypothesis) -> str:
        """Implement the hypothesis as an experiment."""
        experiment_id = await self.runner.start_experiment(hypothesis)
        return experiment_id

    async def run_cycle(self) -> dict:
        """Run one complete OODA cycle."""
        self.cycle_count += 1
        observations = await self.observe()
        opportunities = await self.orient(observations)
        hypothesis = await self.decide(opportunities)

        result = {
            "cycle": self.cycle_count,
            "observations": observations,
            "opportunities_found": len(opportunities),
            "hypothesis": hypothesis.description if hypothesis else None,
        }

        if hypothesis:
            experiment_id = await self.act(hypothesis)
            result["experiment_id"] = experiment_id

        return result
```

### 9.2 Scientific Method Loop

More rigorous than OODA, the scientific method loop adds explicit hypothesis testing and knowledge accumulation:

```
OBSERVE       Collect data, identify anomalies
    |
    v
HYPOTHESIZE   Generate testable predictions
    |
    v
PREDICT       Specify expected outcomes if hypothesis is true
    |
    v
EXPERIMENT    Run controlled test
    |
    v
ANALYZE       Compare results to predictions
    |
    v
CONCLUDE      Accept/reject hypothesis, update knowledge base
    |
    +-------> back to OBSERVE (informed by conclusions)
```

The key difference from OODA is the explicit **prediction** step. Before running an experiment, the system must state what it expects to see. This prevents post-hoc rationalization ("we got a 3% improvement, that must be what we wanted") and forces the system to think precisely about mechanisms.

```python
@dataclass
class ExperimentPrediction:
    """What we expect to see if the hypothesis is correct."""
    target_metric: str
    expected_direction: str       # "increase" or "decrease"
    expected_magnitude: float     # Expected % change
    confidence_interval: tuple[float, float]  # 95% CI of expected effect
    minimum_sample_size: int
    maximum_duration_hours: int

    def evaluate(self, actual_change: float) -> dict:
        """Evaluate whether the prediction was confirmed."""
        within_ci = (
            self.confidence_interval[0] <= actual_change
            <= self.confidence_interval[1]
        )
        correct_direction = (
            (self.expected_direction == "increase" and actual_change > 0) or
            (self.expected_direction == "decrease" and actual_change < 0)
        )

        return {
            "prediction_confirmed": within_ci and correct_direction,
            "direction_correct": correct_direction,
            "within_confidence_interval": within_ci,
            "actual_change": actual_change,
            "expected_change": self.expected_magnitude,
            "prediction_error": abs(actual_change - self.expected_magnitude),
        }
```

### 9.3 Genetic Algorithm Approach

For problems with many interacting parameters (e.g., optimizing a complex configuration), evolutionary search outperforms sequential hypothesis testing.

```python
@dataclass
class ConfigGenome:
    """A configuration represented as an evolvable genome."""
    genes: dict[str, Any]        # Configuration parameters
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class GeneticOptimizer:
    """Genetic algorithm for optimizing agent configurations."""

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.1,
        gene_ranges: dict[str, tuple] = None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.gene_ranges = gene_ranges or {}
        self.population: list[ConfigGenome] = []
        self.generation = 0
        self.best_ever: ConfigGenome | None = None

    def initialize_population(self, base_config: dict) -> list[ConfigGenome]:
        """Create initial population with random variations around base config."""
        self.population = []
        for _ in range(self.population_size):
            genes = {}
            for key, value in base_config.items():
                if key in self.gene_ranges:
                    low, high = self.gene_ranges[key]
                    if isinstance(value, int):
                        genes[key] = random.randint(low, high)
                    elif isinstance(value, float):
                        genes[key] = random.uniform(low, high)
                    else:
                        genes[key] = value
                else:
                    genes[key] = value

            self.population.append(ConfigGenome(genes=genes, generation=0))
        return self.population

    def select_parents(self) -> tuple[ConfigGenome, ConfigGenome]:
        """Tournament selection."""
        tournament_size = 3
        def tournament():
            candidates = random.sample(self.population, tournament_size)
            return max(candidates, key=lambda g: g.fitness)

        return tournament(), tournament()

    def crossover(self, parent_a: ConfigGenome,
                  parent_b: ConfigGenome) -> ConfigGenome:
        """Uniform crossover between two parents."""
        child_genes = {}
        for key in parent_a.genes:
            if random.random() < 0.5:
                child_genes[key] = parent_a.genes[key]
            else:
                child_genes[key] = parent_b.genes.get(key, parent_a.genes[key])

        return ConfigGenome(
            genes=child_genes,
            generation=self.generation + 1,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
        )

    def mutate(self, genome: ConfigGenome) -> ConfigGenome:
        """Mutate individual genes with probability mutation_rate."""
        new_genes = dict(genome.genes)
        for key, value in new_genes.items():
            if random.random() < self.mutation_rate:
                if key in self.gene_ranges:
                    low, high = self.gene_ranges[key]
                    if isinstance(value, int):
                        new_genes[key] = random.randint(low, high)
                    elif isinstance(value, float):
                        # Gaussian perturbation
                        new_genes[key] = max(low, min(high,
                            value + random.gauss(0, (high - low) * 0.1)
                        ))

        return ConfigGenome(
            genes=new_genes,
            generation=self.generation + 1,
            parent_ids=[genome.genome_id],
        )

    def evolve(self) -> list[ConfigGenome]:
        """Run one generation of evolution."""
        self.generation += 1

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best ever
        if not self.best_ever or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0]

        # Keep elites
        new_pop = self.population[:self.elite_count]

        # Generate offspring
        while len(new_pop) < self.population_size:
            parent_a, parent_b = self.select_parents()

            if random.random() < self.crossover_rate:
                child = self.crossover(parent_a, parent_b)
            else:
                child = ConfigGenome(
                    genes=dict(parent_a.genes),
                    generation=self.generation,
                    parent_ids=[parent_a.genome_id],
                )

            child = self.mutate(child)
            new_pop.append(child)

        self.population = new_pop
        return self.population
```

### 9.4 Multi-Armed Bandit Approach

When you have multiple strategies and want to balance exploration vs exploitation in real-time:

```python
class ThompsonSamplingBandit:
    """
    Thompson Sampling for choosing between agent configurations.
    Balances exploration (trying uncertain options) with
    exploitation (using the best known option).
    """

    def __init__(self, arms: list[str]):
        """
        arms: list of configuration identifiers.
        """
        self.arms = arms
        # Beta distribution parameters for each arm
        self.alpha = {arm: 1.0 for arm in arms}  # Successes + 1
        self.beta = {arm: 1.0 for arm in arms}   # Failures + 1

    def select_arm(self) -> str:
        """Select an arm by sampling from each arm's posterior."""
        samples = {
            arm: random.betavariate(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float) -> None:
        """Update beliefs after observing a reward (0 or 1)."""
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

    def get_estimates(self) -> dict[str, dict]:
        """Get current estimates for each arm."""
        return {
            arm: {
                "estimated_rate": self.alpha[arm] / (self.alpha[arm] + self.beta[arm]),
                "total_trials": int(self.alpha[arm] + self.beta[arm] - 2),
                "confidence_width": 1.96 * math.sqrt(
                    (self.alpha[arm] * self.beta[arm])
                    / ((self.alpha[arm] + self.beta[arm]) ** 2
                       * (self.alpha[arm] + self.beta[arm] + 1))
                ),
            }
            for arm in self.arms
        }
```

### 9.5 Combining Approaches

In practice, a self-improving system benefits from multiple approaches working at different scales:

| Scale | Time Horizon | Approach | What It Optimizes |
|-------|-------------|----------|-------------------|
| **Per-task** | Real-time | Multi-armed bandit | Which model/prompt to use right now |
| **Daily** | Hours | OODA loop | Operational parameters, error responses |
| **Weekly** | Days | Scientific method | Prompt evolution, tool creation |
| **Monthly** | Weeks | Genetic algorithm | Architecture-level configurations |

```python
class CompositeEvolutionEngine:
    """
    Combines multiple improvement strategies at different time scales.
    """

    def __init__(self):
        self.bandit = ThompsonSamplingBandit(arms=[])
        self.ooda = OODALoop(...)
        self.prompt_evolver = PromptEvolver()
        self.genetic = GeneticOptimizer()

    async def on_task_start(self, task) -> dict:
        """Real-time decision: use bandit to select configuration."""
        config_id = self.bandit.select_arm()
        return self._get_config(config_id)

    async def on_task_complete(self, task, result) -> None:
        """Update bandit with task outcome."""
        self.bandit.update(task.config_id, float(result.success))

    async def daily_cycle(self) -> dict:
        """Daily OODA cycle for operational improvements."""
        return await self.ooda.run_cycle()

    async def weekly_cycle(self) -> dict:
        """Weekly prompt evolution cycle."""
        new_generation = await self.prompt_evolver.evolve_generation(
            self.llm_client
        )
        # Evaluate new prompts over the next week
        return {"new_prompt_variants": len(new_generation)}

    async def monthly_cycle(self) -> dict:
        """Monthly genetic optimization of architecture parameters."""
        new_population = self.genetic.evolve()
        return {"new_configs": len(new_population)}
```

---

## 10. Preventing Model Collapse in Self-Improvement

### 10.1 The Central Risk

Model collapse occurs when a machine learning system trained on its own outputs progressively degrades. In the context of self-improving AI agents, this manifests differently than in traditional ML, but the core dynamic is the same: the system optimizes for proxies of quality rather than quality itself, and over iterations, the proxies diverge from reality.

Specific risks for autonomous coding agents:

- **Prompt evolution converges to prompt hacks**: Prompts evolve to game evaluation metrics rather than produce genuinely better code.
- **Tool proliferation with declining quality**: New tools are created that pass tests but are brittle, poorly designed, or redundant.
- **Configuration overfitting**: The system optimizes for the specific distribution of recent tasks rather than general capability.
- **Metric gaming**: The system learns to achieve high metric scores through shortcuts (e.g., generating trivially passing tests instead of meaningful ones).
- **Diversity collapse**: All variants converge to a single strategy, losing the ability to handle edge cases.

### 10.2 Golden Test Suites as Anchors

The most reliable defense against model collapse is a **fixed, curated evaluation set** that never changes and never enters the training/optimization loop.

```python
class GoldenTestSuite:
    """
    An immutable test suite that anchors evaluation.
    These tests are NEVER used for optimization --
    only for validation that the system has not degraded.
    """

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.tests = self._load_tests()
        self._baseline_scores: dict[str, float] = {}

    def _load_tests(self) -> list[dict]:
        """Load golden test cases from disk."""
        tests = []
        for test_file in sorted(self.test_dir.glob("*.json")):
            with open(test_file) as f:
                tests.append(json.load(f))
        return tests

    def set_baseline(self, scores: dict[str, float]) -> None:
        """Set the baseline scores that future versions must meet."""
        self._baseline_scores = scores

    async def evaluate(self, agent_config: dict, agent_runner) -> dict:
        """
        Run the golden suite against a configuration.
        Returns pass/fail and comparison to baseline.
        """
        results = []
        for test in self.tests:
            result = await agent_runner.run_task(
                task=test["task"],
                config=agent_config,
                timeout=test.get("timeout", 300),
            )
            results.append({
                "test_id": test["id"],
                "category": test["category"],
                "passed": result.success,
                "score": result.quality_score,
                "cost": result.cost,
                "duration": result.duration,
            })

        # Compute aggregate scores
        scores = {
            "overall_pass_rate": sum(r["passed"] for r in results) / len(results),
            "avg_quality": statistics.mean(r["score"] for r in results),
            "total_cost": sum(r["cost"] for r in results),
        }

        # Category-level scores
        categories = set(r["category"] for r in results)
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            scores[f"pass_rate_{cat}"] = (
                sum(r["passed"] for r in cat_results) / len(cat_results)
            )

        # Compare to baseline
        regression_detected = False
        regressions = []
        for metric, baseline_value in self._baseline_scores.items():
            current = scores.get(metric, 0)
            if metric.startswith("pass_rate") or metric == "avg_quality":
                # Higher is better -- regression if current < baseline
                if current < baseline_value * 0.95:  # 5% tolerance
                    regression_detected = True
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current,
                        "degradation_pct": (
                            (baseline_value - current) / baseline_value * 100
                        ),
                    })

        return {
            "scores": scores,
            "regression_detected": regression_detected,
            "regressions": regressions,
            "individual_results": results,
        }
```

### 10.3 Independent Validation Models

Use a different LLM to validate improvements. This prevents the system from converging on artifacts of a single model's biases.

```python
class IndependentValidator:
    """
    Uses a separate LLM to validate that improvements are genuine.
    Prevents single-model bias from accumulating across generations.
    """

    def __init__(self, validation_model: str, primary_model: str):
        self.validation_model = validation_model
        self.primary_model = primary_model

    async def validate_code_improvement(
        self,
        original_code: str,
        improved_code: str,
        task_description: str,
        llm_client,
    ) -> dict:
        """
        Ask an independent model whether the 'improvement' is genuine.
        """
        prompt = f"""You are reviewing a code change made by an automated system.
The system claims this is an improvement. Your job is to independently assess
whether it is genuinely better.

## Task Description
{task_description}

## Original Code
```python
{original_code}
```

## Proposed "Improvement"
```python
{improved_code}
```

## Assessment Criteria
1. Does the improved code correctly handle all cases the original handled?
2. Does it introduce any new bugs or edge case failures?
3. Is the improvement genuine (better readability, performance, correctness)?
4. Is the change minimal and focused, or does it add unnecessary complexity?

Respond with JSON:
{{
    "is_genuine_improvement": true/false,
    "confidence": 0.0-1.0,
    "risks_identified": ["list of risks"],
    "assessment": "brief explanation"
}}"""

        response = await llm_client.generate(
            prompt,
            model=self.validation_model,  # Different model than the one
            response_format="json",       # that made the improvement
        )
        return json.loads(response)
```

### 10.4 Diversity Maintenance

Preventing diversity collapse requires active measures:

```python
class DiversityMaintainer:
    """
    Ensures the system maintains diverse strategies
    rather than converging to a single approach.
    """

    def __init__(self, min_diversity: float = 0.3):
        self.min_diversity = min_diversity

    def measure_prompt_diversity(self, prompts: list[str]) -> float:
        """
        Measure diversity of a prompt population using
        pairwise distance. Returns 0.0 (identical) to 1.0 (maximally diverse).
        """
        if len(prompts) < 2:
            return 0.0

        from difflib import SequenceMatcher
        distances = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                similarity = SequenceMatcher(
                    None, prompts[i], prompts[j]
                ).ratio()
                distances.append(1 - similarity)

        return statistics.mean(distances)

    def measure_config_diversity(self, configs: list[dict]) -> float:
        """
        Measure diversity of configuration population.
        Uses normalized Hamming distance for discrete params,
        normalized Euclidean distance for continuous params.
        """
        if len(configs) < 2:
            return 0.0

        distances = []
        all_keys = set()
        for c in configs:
            all_keys.update(c.keys())

        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                diff_count = 0
                for key in all_keys:
                    v1 = configs[i].get(key)
                    v2 = configs[j].get(key)
                    if v1 != v2:
                        diff_count += 1
                distances.append(diff_count / len(all_keys) if all_keys else 0)

        return statistics.mean(distances)

    def needs_diversity_injection(self, population_diversity: float) -> bool:
        """Check if the population needs more diversity."""
        return population_diversity < self.min_diversity

    def inject_diversity(
        self,
        population: list[dict],
        base_config: dict,
        gene_ranges: dict,
        n_random: int = 3,
    ) -> list[dict]:
        """
        Inject random individuals into a converging population.
        Replaces the worst-performing individuals.
        """
        for _ in range(n_random):
            random_config = {}
            for key, value in base_config.items():
                if key in gene_ranges:
                    low, high = gene_ranges[key]
                    if isinstance(value, int):
                        random_config[key] = random.randint(low, high)
                    elif isinstance(value, float):
                        random_config[key] = random.uniform(low, high)
                    else:
                        random_config[key] = value
                else:
                    random_config[key] = value
            population.append(random_config)

        return population
```

### 10.5 The Ratchet Mechanism

Improvements should only go forward, never backward. Every accepted change must be validated against the golden suite, and if it regresses on any dimension, it is rejected.

```python
class ImprovementRatchet:
    """
    Ensures improvements are monotonic -- the system never gets worse.
    Acts as a gate between proposed improvements and production.
    """

    def __init__(self, golden_suite: GoldenTestSuite,
                 validator: IndependentValidator):
        self.golden_suite = golden_suite
        self.validator = validator
        self.accepted_versions: list[dict] = []
        self.rejected_versions: list[dict] = []

    async def evaluate_improvement(
        self,
        current_config: dict,
        proposed_config: dict,
        agent_runner,
    ) -> dict:
        """
        Decide whether to accept a proposed improvement.
        Requires:
        1. No regression on golden test suite
        2. Statistically significant improvement on target metric
        3. Independent model validation (optional but recommended)
        """
        # Step 1: Run golden suite on proposed config
        golden_results = await self.golden_suite.evaluate(
            proposed_config, agent_runner
        )

        if golden_results["regression_detected"]:
            decision = {
                "accepted": False,
                "reason": "Golden suite regression detected",
                "regressions": golden_results["regressions"],
            }
            self.rejected_versions.append({
                "config": proposed_config,
                "decision": decision,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return decision

        # Step 2: Compare metrics to current version
        current_results = await self.golden_suite.evaluate(
            current_config, agent_runner
        )

        improvement = (
            golden_results["scores"]["overall_pass_rate"]
            - current_results["scores"]["overall_pass_rate"]
        )

        if improvement < -0.01:  # More than 1% worse
            decision = {
                "accepted": False,
                "reason": f"Net negative improvement: {improvement:.2%}",
            }
            self.rejected_versions.append({
                "config": proposed_config,
                "decision": decision,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return decision

        # Step 3: Accept
        decision = {
            "accepted": True,
            "improvement": improvement,
            "golden_scores": golden_results["scores"],
        }
        self.accepted_versions.append({
            "config": proposed_config,
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return decision
```

### 10.6 When to Stop Improving

Continuous improvement is not always desirable. The system should recognize when further optimization yields diminishing returns:

```python
def should_stop_optimizing(
    improvement_history: list[float],
    min_improvements: int = 10,
    plateau_threshold: float = 0.001,
    plateau_window: int = 5,
) -> tuple[bool, str]:
    """
    Determine if optimization should stop.

    Reasons to stop:
    1. Improvements have plateaued (last N changes < threshold)
    2. The system is already at the practical ceiling
    3. Cost of optimization exceeds expected benefit
    """
    if len(improvement_history) < min_improvements:
        return False, "Not enough improvement history"

    # Check for plateau
    recent = improvement_history[-plateau_window:]
    avg_recent_improvement = statistics.mean(abs(x) for x in recent)

    if avg_recent_improvement < plateau_threshold:
        return True, (
            f"Plateau detected: average improvement in last {plateau_window} "
            f"changes is {avg_recent_improvement:.4f} (threshold: {plateau_threshold})"
        )

    # Check for oscillation (improvements followed by regressions)
    sign_changes = sum(
        1 for i in range(1, len(recent))
        if (recent[i] > 0) != (recent[i-1] > 0)
    )
    if sign_changes >= plateau_window - 1:
        return True, "Oscillation detected: improvements alternate with regressions"

    # Check for diminishing returns
    if len(improvement_history) >= 20:
        first_half = improvement_history[:len(improvement_history)//2]
        second_half = improvement_history[len(improvement_history)//2:]
        if (statistics.mean(abs(x) for x in second_half)
                < 0.25 * statistics.mean(abs(x) for x in first_half)):
            return True, "Diminishing returns: recent improvements are <25% of early ones"

    return False, "Optimization should continue"
```

---

## 11. References

### Research Papers

- **AlphaEvolve**: Novikov et al., "AlphaEvolve: A coding agent for scientific and algorithmic discovery," DeepMind, 2025. [arXiv:2506.13131](https://arxiv.org/abs/2506.13131)
- **FunSearch**: Romera-Paredes et al., "Mathematical discoveries from program search with large language models," Nature, 2024. [Nature article](https://www.nature.com/articles/s41586-023-06924-6)
- **Voyager**: Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- **APE**: Zhou et al., "Large Language Models Are Human-Level Prompt Engineers," ICLR 2023. [arXiv:2211.01910](https://arxiv.org/abs/2211.01910)
- **EvoPrompt**: Guo et al., "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers," ICLR 2024. [arXiv:2309.08532](https://arxiv.org/abs/2309.08532)
- **Model Collapse**: Shumailov et al., "AI models collapse when trained on recursively generated data," Nature, 2024.
- **RouteLLM**: Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," 2024. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

### Frameworks and Tools

- **DSPy**: Stanford NLP framework for programming with language models. [dspy.ai](https://dspy.ai/)
- **OpenEvolve**: Open-source implementation of AlphaEvolve. [GitHub](https://github.com/algorithmicsuperintelligence/openevolve)
- **OpenAlpha_Evolve**: Alternative open-source AlphaEvolve implementation. [GitHub](https://github.com/shyamsaktawat/OpenAlpha_Evolve)

### Architecture References

- **OODA Loop for AI Agents**: NVIDIA Technical Blog on optimizing data center performance with AI agents and the OODA loop strategy. [NVIDIA Blog](https://developer.nvidia.com/blog/optimizing-data-center-performance-with-ai-agents-and-the-ooda-loop-strategy/)
- **Self-Evolving Agents Cookbook**: OpenAI's cookbook for autonomous agent retraining. [OpenAI Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining/)
- **Agentic AI Design Patterns**: Google Cloud documentation on choosing design patterns for agentic AI systems. [Google Cloud](https://docs.google.com/architecture/choose-design-pattern-agentic-ai-system)

### Statistical Methods

- **Sequential A/B Testing**: Evan Miller, "Simple Sequential A/B Testing." [evanmiller.org](https://www.evanmiller.org/sequential-ab-testing.html)
- **Bayesian A/B Testing**: Statsig, "Bayesian A/B testing: Beyond frequentist methods." [Statsig](https://www.statsig.com/perspectives/bayesian-ab-testing-beyond)
- **Thompson Sampling**: Russo et al., "A Tutorial on Thompson Sampling," Foundations and Trends in Machine Learning, 2018.
