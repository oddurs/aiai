"""Tests for HypothesisGenerator."""

from aiai_core.config import EvolutionConfig
from aiai_evolution.hypothesis import Hypothesis, HypothesisGenerator
from aiai_evolution.patterns.base import PatternDetection


def _make_detection(name: str = "cost_spike", severity: str = "high") -> PatternDetection:
    return PatternDetection(
        pattern_name=name,
        severity=severity,
        description=f"Test detection for {name}",
        data={"test": True},
    )


class TestHypothesis:
    def test_creation(self) -> None:
        h = Hypothesis(
            id="abc123",
            title="Test hypothesis",
            description="A test",
            pattern_source="cost_spike",
            expected_improvement=0.15,
            budget_usd=5.0,
        )
        assert h.id == "abc123"
        assert h.status == "proposed"
        assert h.budget_usd == 5.0

    def test_mutable_status(self) -> None:
        h = Hypothesis(
            id="abc123",
            title="Test",
            description="Test",
            pattern_source="test",
            expected_improvement=0.1,
            budget_usd=1.0,
        )
        h.status = "executing"
        assert h.status == "executing"


class TestHypothesisGenerator:
    def test_generates_from_patterns(self) -> None:
        config = EvolutionConfig(max_hypotheses_per_day=10, hypothesis_budget_usd=3.0)
        gen = HypothesisGenerator(config=config)
        patterns = [_make_detection("cost_spike"), _make_detection("error_rate_increase")]
        results = gen.generate(patterns)
        assert len(results) == 2
        assert all(isinstance(h, Hypothesis) for h in results)
        assert all(h.budget_usd == 3.0 for h in results)
        assert results[0].pattern_source == "cost_spike"
        assert results[1].pattern_source == "error_rate_increase"

    def test_rate_limiting(self) -> None:
        config = EvolutionConfig(max_hypotheses_per_day=2)
        gen = HypothesisGenerator(config=config)
        patterns = [
            _make_detection("cost_spike"),
            _make_detection("error_rate_increase"),
            _make_detection("latency_degradation"),
        ]
        results = gen.generate(patterns)
        assert len(results) == 2
        assert gen.daily_count == 2
        assert gen.remaining_today == 0

    def test_rate_limit_across_calls(self) -> None:
        config = EvolutionConfig(max_hypotheses_per_day=3)
        gen = HypothesisGenerator(config=config)
        r1 = gen.generate([_make_detection("cost_spike")])
        assert len(r1) == 1
        assert gen.daily_count == 1
        r2 = gen.generate([_make_detection("error_rate_increase"), _make_detection("token_waste")])
        assert len(r2) == 2
        assert gen.daily_count == 3
        # Now at limit
        r3 = gen.generate([_make_detection("latency_degradation")])
        assert len(r3) == 0
        assert gen.remaining_today == 0

    def test_empty_patterns(self) -> None:
        gen = HypothesisGenerator()
        results = gen.generate([])
        assert results == []
        assert gen.daily_count == 0

    def test_budget_from_config(self) -> None:
        config = EvolutionConfig(hypothesis_budget_usd=10.0)
        gen = HypothesisGenerator(config=config)
        results = gen.generate([_make_detection()])
        assert results[0].budget_usd == 10.0

    def test_unknown_pattern_gets_generic_title(self) -> None:
        gen = HypothesisGenerator()
        results = gen.generate([_make_detection("unknown_pattern")])
        assert len(results) == 1
        assert "unknown_pattern" in results[0].title

    def test_severity_maps_to_expected_improvement(self) -> None:
        gen = HypothesisGenerator()
        low = gen.generate([_make_detection(severity="low")])[0]
        # Reset by creating new generator
        gen2 = HypothesisGenerator()
        critical = gen2.generate([_make_detection(severity="critical")])[0]
        assert low.expected_improvement < critical.expected_improvement

    def test_default_config(self) -> None:
        gen = HypothesisGenerator()
        assert gen.remaining_today == 3  # default max_hypotheses_per_day

    def test_hypothesis_ids_are_unique(self) -> None:
        gen = HypothesisGenerator(config=EvolutionConfig(max_hypotheses_per_day=100))
        patterns = [_make_detection() for _ in range(10)]
        results = gen.generate(patterns)
        ids = [h.id for h in results]
        assert len(ids) == len(set(ids))
