"""Detect sudden cost increases in metrics data."""

from __future__ import annotations

from aiai_evolution.patterns.base import BasePattern, PatternDetection


class CostSpikePattern(BasePattern):
    """Detects when recent costs spike above the historical average.

    Compares the average cost of the most recent 20% of records against
    the average of the earlier 80%. Triggers when the ratio exceeds the threshold.
    """

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "cost_spike"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        if len(data) < 5:
            return None

        costs = [
            float(row.get("cost_usd", 0) or row.get("total_cost_usd", 0))  # type: ignore[arg-type]
            for row in data
        ]
        if not any(c > 0 for c in costs):
            return None

        split = max(1, len(costs) * 4 // 5)
        baseline_costs = costs[:split]
        recent_costs = costs[split:]

        baseline_avg = sum(baseline_costs) / len(baseline_costs) if baseline_costs else 0
        recent_avg = sum(recent_costs) / len(recent_costs) if recent_costs else 0

        if baseline_avg <= 0:
            return None

        ratio = recent_avg / baseline_avg
        if ratio >= self._threshold:
            severity = "critical" if ratio >= self._threshold * 2 else "high"
            return PatternDetection(
                pattern_name=self.name,
                severity=severity,
                description=(
                    f"Cost spike detected: recent avg ${recent_avg:.4f} is "
                    f"{ratio:.1f}x the baseline avg ${baseline_avg:.4f} "
                    f"(threshold: {self._threshold}x)"
                ),
                data={
                    "baseline_avg": baseline_avg,
                    "recent_avg": recent_avg,
                    "ratio": ratio,
                    "threshold": self._threshold,
                },
            )
        return None
