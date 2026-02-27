"""Detect latency regression in task metrics."""

from __future__ import annotations

from aiai_evolution.patterns.base import BasePattern, PatternDetection


class LatencyPattern(BasePattern):
    """Detects latency regression by comparing recent tasks to baseline.

    Compares the average latency (derived from start_time/end_time) of
    the most recent 20% of records against the earlier 80%.
    Triggers when the ratio exceeds the threshold.
    """

    def __init__(self, threshold: float = 1.5) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "latency_degradation"

    def _extract_duration_ms(self, row: dict[str, object]) -> float | None:
        """Extract duration in ms from a task metric row."""
        start = row.get("start_time")
        end = row.get("end_time")
        if start is None or end is None:
            return None
        try:
            delta = end - start  # type: ignore[operator]
            return float(delta.total_seconds() * 1000)
        except (TypeError, AttributeError):
            return None

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        durations = []
        for row in data:
            d = self._extract_duration_ms(row)
            if d is not None and d > 0:
                durations.append(d)

        if len(durations) < 5:
            return None

        split = max(1, len(durations) * 4 // 5)
        baseline = durations[:split]
        recent = durations[split:]

        baseline_avg = sum(baseline) / len(baseline) if baseline else 0
        recent_avg = sum(recent) / len(recent) if recent else 0

        if baseline_avg <= 0:
            return None

        ratio = recent_avg / baseline_avg
        if ratio >= self._threshold:
            severity = "critical" if ratio >= self._threshold * 2 else "high"
            return PatternDetection(
                pattern_name=self.name,
                severity=severity,
                description=(
                    f"Latency regression: recent avg {recent_avg:.0f}ms is "
                    f"{ratio:.1f}x the baseline avg {baseline_avg:.0f}ms "
                    f"(threshold: {self._threshold}x)"
                ),
                data={
                    "baseline_avg_ms": baseline_avg,
                    "recent_avg_ms": recent_avg,
                    "ratio": ratio,
                    "threshold": self._threshold,
                },
            )
        return None
