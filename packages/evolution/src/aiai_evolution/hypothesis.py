"""Hypothesis generation from detected patterns."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from aiai_core.config import EvolutionConfig
from aiai_core.logging import get_logger

if TYPE_CHECKING:
    from aiai_evolution.patterns.base import PatternDetection

# Mapping from pattern names to hypothesis templates.
_HYPOTHESIS_TEMPLATES: dict[str, dict[str, str]] = {
    "cost_spike": {
        "title": "Reduce model costs for recent spike",
        "description": (
            "Recent costs have spiked above baseline. Consider routing more tasks "
            "to cheaper model tiers or optimizing prompt lengths."
        ),
    },
    "error_rate_increase": {
        "title": "Investigate and reduce task error rate",
        "description": (
            "Task failure rate has exceeded the threshold. Review failing tasks "
            "for common error patterns and add retries or fallbacks."
        ),
    },
    "latency_degradation": {
        "title": "Address latency regression in task execution",
        "description": (
            "Task latency has regressed compared to baseline. Consider using faster "
            "model tiers or reducing prompt complexity for affected tasks."
        ),
    },
    "token_waste": {
        "title": "Optimize token usage for wasteful tasks",
        "description": (
            "Tasks are using more tokens than expected for their complexity. "
            "Review and compress prompts, or reclassify task complexity levels."
        ),
    },
}


@dataclass
class Hypothesis:
    """An improvement proposal generated from a detected pattern."""

    id: str
    title: str
    description: str
    pattern_source: str
    expected_improvement: float
    budget_usd: float
    status: str = "proposed"  # proposed, executing, succeeded, failed, reverted


class HypothesisGenerator:
    """Generates improvement hypotheses from detected patterns.

    Enforces a daily rate limit on the number of hypotheses generated,
    and assigns a per-hypothesis cost budget from configuration.
    """

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        if config is None:
            config = EvolutionConfig()
        self._config = config
        self._logger = get_logger()
        self._daily_counts: dict[str, int] = {}

    def _today_key(self) -> str:
        return date.today().isoformat()

    @property
    def daily_count(self) -> int:
        """Number of hypotheses generated today."""
        return self._daily_counts.get(self._today_key(), 0)

    def _increment_daily_count(self, n: int = 1) -> None:
        key = self._today_key()
        self._daily_counts[key] = self._daily_counts.get(key, 0) + n

    @property
    def remaining_today(self) -> int:
        """Number of hypotheses that can still be generated today."""
        return max(0, self._config.max_hypotheses_per_day - self.daily_count)

    def generate(self, patterns: list[PatternDetection]) -> list[Hypothesis]:
        """Generate hypotheses from detected patterns.

        Args:
            patterns: List of pattern detections from MetricsAnalyzer.

        Returns:
            List of hypotheses, limited by daily rate limit.
        """
        remaining = self.remaining_today
        if remaining <= 0:
            self._logger.warn(
                "hypothesis.rate_limited",
                daily_count=self.daily_count,
                max_per_day=self._config.max_hypotheses_per_day,
            )
            return []

        hypotheses: list[Hypothesis] = []
        for detection in patterns:
            if remaining <= 0:
                break

            template = _HYPOTHESIS_TEMPLATES.get(detection.pattern_name)
            if template is None:
                title = f"Address {detection.pattern_name} pattern"
                description = detection.description
            else:
                title = template["title"]
                description = template["description"]

            # Estimate expected improvement from severity.
            expected = _severity_to_improvement(detection.severity)

            hypothesis = Hypothesis(
                id=str(uuid.uuid4())[:8],
                title=title,
                description=description,
                pattern_source=detection.pattern_name,
                expected_improvement=expected,
                budget_usd=self._config.hypothesis_budget_usd,
            )
            hypotheses.append(hypothesis)
            remaining -= 1

        self._increment_daily_count(len(hypotheses))
        self._logger.info(
            "hypothesis.generated",
            count=len(hypotheses),
            daily_total=self.daily_count,
        )
        return hypotheses


def _severity_to_improvement(severity: str) -> float:
    """Map severity to an expected improvement percentage."""
    return {
        "low": 0.05,
        "medium": 0.10,
        "high": 0.20,
        "critical": 0.30,
    }.get(severity, 0.10)
