"""Tests for individual pattern detectors."""

from datetime import UTC, datetime, timedelta

from aiai_evolution.patterns.cost_spike import CostSpikePattern
from aiai_evolution.patterns.error_rate import ErrorRatePattern
from aiai_evolution.patterns.latency import LatencyPattern
from aiai_evolution.patterns.token_waste import TokenWastePattern


class TestCostSpikePattern:
    def test_no_spike_returns_none(self) -> None:
        pattern = CostSpikePattern(threshold=2.0)
        data = [{"cost_usd": 0.01} for _ in range(10)]
        assert pattern.detect(data) is None

    def test_spike_detected(self) -> None:
        pattern = CostSpikePattern(threshold=2.0)
        # 8 low-cost records, then 2 high-cost records (>2x)
        data = [{"cost_usd": 0.01} for _ in range(8)]
        data += [{"cost_usd": 0.05} for _ in range(2)]
        result = pattern.detect(data)
        assert result is not None
        assert result.pattern_name == "cost_spike"
        assert result.severity in ("high", "critical")
        assert result.data["ratio"] >= 2.0

    def test_too_few_records_returns_none(self) -> None:
        pattern = CostSpikePattern()
        data = [{"cost_usd": 0.01} for _ in range(3)]
        assert pattern.detect(data) is None

    def test_zero_baseline_returns_none(self) -> None:
        pattern = CostSpikePattern()
        data = [{"cost_usd": 0.0} for _ in range(8)]
        data += [{"cost_usd": 0.05} for _ in range(2)]
        assert pattern.detect(data) is None

    def test_uses_total_cost_usd_field(self) -> None:
        pattern = CostSpikePattern(threshold=2.0)
        data = [{"total_cost_usd": 0.01} for _ in range(8)]
        data += [{"total_cost_usd": 0.05} for _ in range(2)]
        result = pattern.detect(data)
        assert result is not None
        assert result.pattern_name == "cost_spike"

    def test_custom_threshold(self) -> None:
        pattern = CostSpikePattern(threshold=5.0)
        # 2x spike should NOT trigger at 5x threshold
        data = [{"cost_usd": 0.01} for _ in range(8)]
        data += [{"cost_usd": 0.02} for _ in range(2)]
        assert pattern.detect(data) is None

    def test_name(self) -> None:
        assert CostSpikePattern().name == "cost_spike"

    def test_critical_severity_for_extreme_spike(self) -> None:
        pattern = CostSpikePattern(threshold=2.0)
        # 4x+ spike should be critical (threshold * 2)
        data = [{"cost_usd": 0.01} for _ in range(8)]
        data += [{"cost_usd": 0.10} for _ in range(2)]
        result = pattern.detect(data)
        assert result is not None
        assert result.severity == "critical"


class TestErrorRatePattern:
    def test_no_errors_returns_none(self) -> None:
        pattern = ErrorRatePattern(threshold=0.10)
        data = [{"status": "success"} for _ in range(10)]
        assert pattern.detect(data) is None

    def test_high_error_rate_detected(self) -> None:
        pattern = ErrorRatePattern(threshold=0.10)
        data = [{"status": "success"} for _ in range(7)]
        data += [{"status": "failed"} for _ in range(3)]
        result = pattern.detect(data)
        assert result is not None
        assert result.pattern_name == "error_rate_increase"
        assert result.data["error_rate"] == 0.3

    def test_below_threshold_returns_none(self) -> None:
        pattern = ErrorRatePattern(threshold=0.50)
        data = [{"status": "success"} for _ in range(8)]
        data += [{"status": "failed"} for _ in range(2)]
        assert pattern.detect(data) is None

    def test_too_few_records(self) -> None:
        pattern = ErrorRatePattern()
        data = [{"status": "failed"} for _ in range(2)]
        assert pattern.detect(data) is None

    def test_all_failed(self) -> None:
        pattern = ErrorRatePattern(threshold=0.10)
        data = [{"status": "failed"} for _ in range(5)]
        result = pattern.detect(data)
        assert result is not None
        assert result.severity == "critical"

    def test_name(self) -> None:
        assert ErrorRatePattern().name == "error_rate_increase"


class TestLatencyPattern:
    def _make_task(self, duration_ms: float, base_time: datetime) -> dict[str, object]:
        start = base_time
        end = start + timedelta(milliseconds=duration_ms)
        return {"start_time": start, "end_time": end}

    def test_no_regression_returns_none(self) -> None:
        pattern = LatencyPattern(threshold=1.5)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        data = [self._make_task(100, base + timedelta(minutes=i)) for i in range(10)]
        assert pattern.detect(data) is None

    def test_latency_regression_detected(self) -> None:
        pattern = LatencyPattern(threshold=1.5)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        # 8 normal tasks, 2 slow tasks (3x latency)
        data = [self._make_task(100, base + timedelta(minutes=i)) for i in range(8)]
        data += [self._make_task(300, base + timedelta(minutes=i + 8)) for i in range(2)]
        result = pattern.detect(data)
        assert result is not None
        assert result.pattern_name == "latency_degradation"
        assert result.data["ratio"] >= 1.5

    def test_too_few_records(self) -> None:
        pattern = LatencyPattern()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        data = [self._make_task(100, base) for _ in range(3)]
        assert pattern.detect(data) is None

    def test_missing_end_time_skipped(self) -> None:
        pattern = LatencyPattern()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        data = [{"start_time": base, "end_time": None} for _ in range(10)]
        assert pattern.detect(data) is None

    def test_name(self) -> None:
        assert LatencyPattern().name == "latency_degradation"


class TestTokenWastePattern:
    def test_no_waste_returns_none(self) -> None:
        pattern = TokenWastePattern(threshold=1.5)
        data = [{"total_tokens": 500, "complexity": "trivial"} for _ in range(5)]
        assert pattern.detect(data) is None

    def test_waste_detected(self) -> None:
        pattern = TokenWastePattern(threshold=1.5)
        # trivial tasks using 2x expected tokens (expected: 500)
        data = [{"total_tokens": 1200, "complexity": "trivial"} for _ in range(5)]
        result = pattern.detect(data)
        assert result is not None
        assert result.pattern_name == "token_waste"
        assert result.data["avg_ratio"] >= 1.5

    def test_below_threshold_returns_none(self) -> None:
        pattern = TokenWastePattern(threshold=3.0)
        data = [{"total_tokens": 1000, "complexity": "trivial"} for _ in range(5)]
        assert pattern.detect(data) is None

    def test_too_few_records(self) -> None:
        pattern = TokenWastePattern()
        data = [{"total_tokens": 5000, "complexity": "trivial"} for _ in range(2)]
        assert pattern.detect(data) is None

    def test_unknown_complexity_skipped(self) -> None:
        pattern = TokenWastePattern()
        data = [{"total_tokens": 5000, "complexity": "unknown"} for _ in range(5)]
        assert pattern.detect(data) is None

    def test_name(self) -> None:
        assert TokenWastePattern().name == "token_waste"

    def test_high_severity_for_extreme_waste(self) -> None:
        pattern = TokenWastePattern(threshold=1.5)
        # 3x+ waste should be high severity
        data = [{"total_tokens": 2000, "complexity": "trivial"} for _ in range(5)]
        result = pattern.detect(data)
        assert result is not None
        assert result.severity == "high"
