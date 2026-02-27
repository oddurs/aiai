"""Config loading for aiai system configuration files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiai_core.types import Complexity, ModelTier

# Default config directory relative to repo root
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[4] / "config"


def _config_dir() -> Path:
    """Return the config directory, respecting AIAI_CONFIG_DIR env var."""
    env = os.environ.get("AIAI_CONFIG_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CONFIG_DIR


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single model tier."""

    name: ModelTier
    models: list[str]
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class CostConfig:
    """Cost tracking configuration."""

    log_file: str = "logs/model-costs.jsonl"
    warn_threshold_usd: float = 1.0
    daily_budget_usd: float = 50.0


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed models.yaml configuration."""

    provider: str
    tiers: dict[ModelTier, TierConfig]
    routing: dict[Complexity, ModelTier]
    cost: CostConfig

    def tier_for_complexity(self, complexity: Complexity) -> TierConfig:
        """Get the tier config for a given complexity level."""
        model_tier = self.routing[complexity]
        return self.tiers[model_tier]


@dataclass(frozen=True)
class SafetyConfig:
    """Parsed safety.yaml configuration."""

    protected_files: list[str] = field(default_factory=list)
    max_failures_per_task: int = 5
    max_cost_per_task_usd: float = 5.0
    max_retries_per_task: int = 3
    max_tasks_per_hour: int = 100
    max_hypotheses_per_day: int = 3
    max_active_evolution_branches: int = 1
    hypothesis_budget_usd: float = 5.0
    quality_regression_threshold: float = 0.10
    quality_regression_window_days: int = 7


@dataclass(frozen=True)
class EvolutionConfig:
    """Parsed evolution.yaml configuration."""

    enabled: bool = False
    max_hypotheses_per_day: int = 3
    hypothesis_budget_usd: float = 5.0
    max_active_branches: int = 1
    metrics_window_days: int = 7
    min_improvement_threshold: float = 0.05
    auto_revert_on_regression: bool = True
    checkpoint_tag_prefix: str = "pre-evolution"


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    path = _config_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def load_models_config(filename: str = "models.yaml") -> ModelsConfig:
    """Load and parse the model routing configuration."""
    raw = load_yaml(filename)

    tiers: dict[ModelTier, TierConfig] = {}
    for tier_name, tier_data in raw.get("tiers", {}).items():
        mt = ModelTier(tier_name)
        tiers[mt] = TierConfig(
            name=mt,
            models=tier_data["models"],
            max_tokens=tier_data["max_tokens"],
            temperature=tier_data["temperature"],
        )

    routing: dict[Complexity, ModelTier] = {}
    for complexity_name, tier_name in raw.get("routing", {}).items():
        routing[Complexity(complexity_name)] = ModelTier(tier_name)

    cost_data = raw.get("cost", {})
    cost = CostConfig(
        log_file=cost_data.get("log_file", "logs/model-costs.jsonl"),
        warn_threshold_usd=cost_data.get("warn_threshold_usd", 1.0),
        daily_budget_usd=cost_data.get("daily_budget_usd", 50.0),
    )

    return ModelsConfig(
        provider=raw.get("provider", "openrouter"),
        tiers=tiers,
        routing=routing,
        cost=cost,
    )


def load_safety_config(filename: str = "safety.yaml") -> SafetyConfig:
    """Load and parse the safety configuration."""
    try:
        raw = load_yaml(filename)
    except FileNotFoundError:
        return SafetyConfig()

    cb = raw.get("circuit_breaker", {})
    evo = raw.get("evolution", {})
    quality = raw.get("quality", {})

    return SafetyConfig(
        protected_files=raw.get("protected_files", []),
        max_failures_per_task=cb.get("max_failures_per_task", 5),
        max_cost_per_task_usd=cb.get("max_cost_per_task_usd", 5.0),
        max_retries_per_task=cb.get("max_retries_per_task", 3),
        max_tasks_per_hour=cb.get("max_tasks_per_hour", 100),
        max_hypotheses_per_day=evo.get("max_hypotheses_per_day", 3),
        max_active_evolution_branches=evo.get("max_active_branches", 1),
        hypothesis_budget_usd=evo.get("hypothesis_budget_usd", 5.0),
        quality_regression_threshold=quality.get("regression_threshold", 0.10),
        quality_regression_window_days=quality.get("regression_window_days", 7),
    )


def load_evolution_config(filename: str = "evolution.yaml") -> EvolutionConfig:
    """Load and parse the evolution engine configuration."""
    try:
        raw = load_yaml(filename)
    except FileNotFoundError:
        return EvolutionConfig()

    return EvolutionConfig(
        enabled=raw.get("enabled", False),
        max_hypotheses_per_day=raw.get("max_hypotheses_per_day", 3),
        hypothesis_budget_usd=raw.get("hypothesis_budget_usd", 5.0),
        max_active_branches=raw.get("max_active_branches", 1),
        metrics_window_days=raw.get("metrics_window_days", 7),
        min_improvement_threshold=raw.get("min_improvement_threshold", 0.05),
        auto_revert_on_regression=raw.get("auto_revert_on_regression", True),
        checkpoint_tag_prefix=raw.get("checkpoint_tag_prefix", "pre-evolution"),
    )
