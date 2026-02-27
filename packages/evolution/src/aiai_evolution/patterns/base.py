"""Base class for pluggable pattern detectors."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatternDetection:
    """Result of a pattern detection run."""

    pattern_name: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    data: dict[str, object] = field(default_factory=dict)


class BasePattern(abc.ABC):
    """Abstract base for pattern detectors.

    Each detector examines a list of metric dicts (rows from MetricsStore queries)
    and returns a PatternDetection if the pattern is found, None otherwise.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique name for this pattern detector."""

    @abc.abstractmethod
    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        """Analyze data and return a detection if the pattern is found."""
