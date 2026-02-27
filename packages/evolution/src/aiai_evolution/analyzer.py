"""Metrics analyzer that runs pattern detectors over metric data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiai_core.logging import get_logger

from aiai_evolution.patterns import ALL_PATTERNS

if TYPE_CHECKING:
    from aiai_evolution.patterns.base import BasePattern, PatternDetection


class MetricsAnalyzer:
    """Runs pluggable pattern detectors over metric data.

    Accepts data as a parameter rather than coupling to MetricsStore directly,
    making it easy to test with synthetic data.
    """

    def __init__(self, patterns: list[BasePattern] | None = None) -> None:
        """Initialize with optional custom patterns.

        If no patterns are provided, all built-in detectors are loaded
        with default thresholds.
        """
        if patterns is not None:
            self._patterns = patterns
        else:
            self._patterns = [cls() for cls in ALL_PATTERNS]
        self._logger = get_logger()

    @property
    def patterns(self) -> list[BasePattern]:
        """Return the registered patterns."""
        return list(self._patterns)

    def analyze(self, data: list[dict[str, object]]) -> list[PatternDetection]:
        """Run all pattern detectors over the provided data.

        Args:
            data: List of metric dicts (rows from MetricsStore or synthetic).

        Returns:
            List of PatternDetection results for any patterns found.
        """
        detections: list[PatternDetection] = []
        for pattern in self._patterns:
            try:
                result = pattern.detect(data)
                if result is not None:
                    self._logger.info(
                        "pattern.detected",
                        pattern=result.pattern_name,
                        severity=result.severity,
                    )
                    detections.append(result)
            except Exception as exc:
                self._logger.error(
                    "pattern.error",
                    pattern=pattern.name,
                    error=str(exc),
                )
        return detections
