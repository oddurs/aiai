"""aiai-metrics: custom analytics with DuckDB time-series storage."""

from aiai_metrics.aggregator import DailySummary, MetricsAggregator, ModelPerformance
from aiai_metrics.api import create_app
from aiai_metrics.collector import MetricsCollector
from aiai_metrics.store import MetricsStore

__all__ = [
    "DailySummary",
    "MetricsAggregator",
    "MetricsCollector",
    "MetricsStore",
    "ModelPerformance",
    "create_app",
]
