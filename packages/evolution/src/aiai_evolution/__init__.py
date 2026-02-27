"""aiai-evolution: self-improvement engine with pattern detection and hypothesis testing."""

from aiai_evolution.analyzer import MetricsAnalyzer
from aiai_evolution.executor import ExecutionResult, HypothesisExecutor
from aiai_evolution.hypothesis import Hypothesis, HypothesisGenerator
from aiai_evolution.patterns import (
    ALL_PATTERNS,
    BasePattern,
    CostSpikePattern,
    ErrorRatePattern,
    LatencyPattern,
    PatternDetection,
    TokenWastePattern,
)

__all__ = [
    "MetricsAnalyzer",
    "HypothesisExecutor",
    "ExecutionResult",
    "Hypothesis",
    "HypothesisGenerator",
    "BasePattern",
    "PatternDetection",
    "CostSpikePattern",
    "ErrorRatePattern",
    "LatencyPattern",
    "TokenWastePattern",
    "ALL_PATTERNS",
]
