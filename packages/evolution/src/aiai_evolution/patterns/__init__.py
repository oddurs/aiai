"""Pluggable pattern detectors for the evolution engine."""

from aiai_evolution.patterns.base import BasePattern, PatternDetection
from aiai_evolution.patterns.cost_spike import CostSpikePattern
from aiai_evolution.patterns.error_rate import ErrorRatePattern
from aiai_evolution.patterns.latency import LatencyPattern
from aiai_evolution.patterns.token_waste import TokenWastePattern

ALL_PATTERNS: list[type[BasePattern]] = [
    CostSpikePattern,
    ErrorRatePattern,
    LatencyPattern,
    TokenWastePattern,
]

__all__ = [
    "BasePattern",
    "PatternDetection",
    "CostSpikePattern",
    "ErrorRatePattern",
    "LatencyPattern",
    "TokenWastePattern",
    "ALL_PATTERNS",
]
