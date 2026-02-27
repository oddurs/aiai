"""Tests for MetricsAnalyzer."""

from aiai_evolution.analyzer import MetricsAnalyzer
from aiai_evolution.patterns.base import BasePattern, PatternDetection
from aiai_evolution.patterns.cost_spike import CostSpikePattern
from aiai_evolution.patterns.error_rate import ErrorRatePattern


class _AlwaysDetectPattern(BasePattern):
    """Test pattern that always returns a detection."""

    @property
    def name(self) -> str:
        return "always_detect"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        return PatternDetection(
            pattern_name=self.name,
            severity="medium",
            description="Always detected",
            data={"count": len(data)},
        )


class _NeverDetectPattern(BasePattern):
    """Test pattern that never returns a detection."""

    @property
    def name(self) -> str:
        return "never_detect"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        return None


class _ErrorPattern(BasePattern):
    """Test pattern that raises an exception."""

    @property
    def name(self) -> str:
        return "error_pattern"

    def detect(self, data: list[dict[str, object]]) -> PatternDetection | None:
        raise ValueError("Pattern detector error")


class TestMetricsAnalyzer:
    def test_no_data_returns_empty(self) -> None:
        analyzer = MetricsAnalyzer(patterns=[_AlwaysDetectPattern()])
        # Even "always detect" returns something for empty list
        results = analyzer.analyze([])
        assert len(results) == 1

    def test_detects_pattern(self) -> None:
        analyzer = MetricsAnalyzer(patterns=[_AlwaysDetectPattern()])
        results = analyzer.analyze([{"x": 1}])
        assert len(results) == 1
        assert results[0].pattern_name == "always_detect"

    def test_no_detection(self) -> None:
        analyzer = MetricsAnalyzer(patterns=[_NeverDetectPattern()])
        results = analyzer.analyze([{"x": 1}])
        assert len(results) == 0

    def test_multiple_patterns(self) -> None:
        analyzer = MetricsAnalyzer(
            patterns=[_AlwaysDetectPattern(), _NeverDetectPattern()]
        )
        results = analyzer.analyze([{"x": 1}])
        assert len(results) == 1
        assert results[0].pattern_name == "always_detect"

    def test_error_in_pattern_does_not_crash(self) -> None:
        analyzer = MetricsAnalyzer(
            patterns=[_ErrorPattern(), _AlwaysDetectPattern()]
        )
        results = analyzer.analyze([{"x": 1}])
        # Error pattern is skipped, always_detect still runs
        assert len(results) == 1
        assert results[0].pattern_name == "always_detect"

    def test_default_patterns_loaded(self) -> None:
        analyzer = MetricsAnalyzer()
        assert len(analyzer.patterns) == 4

    def test_with_real_cost_spike_data(self) -> None:
        analyzer = MetricsAnalyzer(patterns=[CostSpikePattern(threshold=2.0)])
        data = [{"cost_usd": 0.01} for _ in range(8)]
        data += [{"cost_usd": 0.05} for _ in range(2)]
        results = analyzer.analyze(data)
        assert len(results) == 1
        assert results[0].pattern_name == "cost_spike"

    def test_with_real_error_rate_data(self) -> None:
        analyzer = MetricsAnalyzer(patterns=[ErrorRatePattern(threshold=0.10)])
        data = [{"status": "success"} for _ in range(5)]
        data += [{"status": "failed"} for _ in range(5)]
        results = analyzer.analyze(data)
        assert len(results) == 1
        assert results[0].pattern_name == "error_rate_increase"

    def test_patterns_property_returns_copy(self) -> None:
        patterns = [_AlwaysDetectPattern()]
        analyzer = MetricsAnalyzer(patterns=patterns)
        returned = analyzer.patterns
        returned.append(_NeverDetectPattern())
        assert len(analyzer.patterns) == 1
