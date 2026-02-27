"""Core types shared across all aiai packages."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime


class Complexity(enum.StrEnum):
    """Task complexity levels for model routing."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"


class ModelTier(enum.StrEnum):
    """Model tier names matching config/models.yaml."""

    NANO = "nano"
    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"
    MAX = "max"


class TaskStatus(enum.StrEnum):
    """Lifecycle status of an agent task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(frozen=True)
class RouteRequest:
    """Request to route a prompt to an appropriate model."""

    prompt: str
    complexity: Complexity
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteResponse:
    """Response from a routed model call."""

    content: str
    model: str
    tier: ModelTier
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    request_id: str = ""


@dataclass
class CostRecord:
    """A single cost event for logging and budget tracking."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    model: str = ""
    tier: str = ""
    task_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    daily_total_usd: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for JSONL logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "tier": self.tier,
            "task_id": self.task_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "daily_total_usd": self.daily_total_usd,
        }


@dataclass
class TaskMetrics:
    """Metrics collected during task execution."""

    task_id: str
    complexity: Complexity
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    model_calls: int = 0
    retries: int = 0
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Task duration in milliseconds, or None if not finished."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000
