"""Detect unnecessarily high token usage relative to task complexity."""

from __future__ import annotations

from aiai_evolution.patterns.base import BasePattern, PatternDetection

# Expected token budgets per complexity level (rough heuristics).
_EXPECTED_TOKENS: dict[str, int] = {
    "trivial": 500,
    "simple": 2000,
    "medium": 5000,
    "complex": 15000,
    "critical": 30000,
}


class TokenWastePattern(BasePattern):
    """Detects tasks that use significantly more tokens than expected for their complexity.

    Compares actual total_tokens against expected budgets per complexity level.
    Triggers when the average ratio across tasks exceeds the threshold.
    """

    def __init__(self, threshold: float = 1.5) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "token_waste"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        if len(data) < 3:
            return None

        ratios: list[float] = []
        for row in data:
            tokens = row.get("total_tokens")
            complexity = str(row.get("complexity", "")).lower()
            if tokens is None or not isinstance(tokens, (int, float)):
                continue
            expected = _EXPECTED_TOKENS.get(complexity)
            if expected is None or expected <= 0:
                continue
            ratios.append(float(tokens) / expected)

        if len(ratios) < 3:
            return None

        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio >= self._threshold:
            worst_ratio = max(ratios)
            severity = "high" if avg_ratio >= self._threshold * 2 else "medium"
            return PatternDetection(
                pattern_name=self.name,
                severity=severity,
                description=(
                    f"Token waste detected: avg usage is {avg_ratio:.1f}x expected "
                    f"for task complexity (threshold: {self._threshold}x, "
                    f"worst: {worst_ratio:.1f}x)"
                ),
                data={
                    "avg_ratio": avg_ratio,
                    "worst_ratio": worst_ratio,
                    "sample_count": len(ratios),
                    "threshold": self._threshold,
                },
            )
        return None
