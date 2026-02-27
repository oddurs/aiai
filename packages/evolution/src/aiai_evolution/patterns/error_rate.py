"""Detect rising error rates in task metrics."""

from __future__ import annotations

from aiai_evolution.patterns.base import BasePattern, PatternDetection


class ErrorRatePattern(BasePattern):
    """Detects when the error rate exceeds a threshold.

    Examines task status fields and calculates the ratio of failed tasks
    to total tasks. Triggers when this ratio exceeds the threshold.
    """

    def __init__(self, threshold: float = 0.10) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "error_rate_increase"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        if len(data) < 3:
            return None

        total = len(data)
        failed = sum(1 for row in data if str(row.get("status", "")).lower() == "failed")
        error_rate = failed / total

        if error_rate >= self._threshold:
            severity = "critical" if error_rate >= self._threshold * 3 else (
                "high" if error_rate >= self._threshold * 2 else "medium"
            )
            return PatternDetection(
                pattern_name=self.name,
                severity=severity,
                description=(
                    f"Error rate {error_rate:.1%} exceeds threshold {self._threshold:.1%} "
                    f"({failed}/{total} tasks failed)"
                ),
                data={
                    "error_rate": error_rate,
                    "failed_count": failed,
                    "total_count": total,
                    "threshold": self._threshold,
                },
            )
        return None
