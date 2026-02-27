"""Shared fixtures for router tests."""

from __future__ import annotations

import pytest
from aiai_core.config import CostConfig, ModelsConfig, TierConfig
from aiai_core.types import Complexity, ModelTier


@pytest.fixture()
def models_config() -> ModelsConfig:
    """A minimal ModelsConfig for testing."""
    return ModelsConfig(
        provider="openrouter",
        tiers={
            ModelTier.NANO: TierConfig(
                name=ModelTier.NANO,
                models=["test/nano-model-a", "test/nano-model-b"],
                max_tokens=1024,
                temperature=0.0,
            ),
            ModelTier.FAST: TierConfig(
                name=ModelTier.FAST,
                models=["test/fast-model-a", "test/fast-model-b"],
                max_tokens=4096,
                temperature=0.0,
            ),
            ModelTier.BALANCED: TierConfig(
                name=ModelTier.BALANCED,
                models=["test/balanced-model-a"],
                max_tokens=8192,
                temperature=0.0,
            ),
            ModelTier.POWERFUL: TierConfig(
                name=ModelTier.POWERFUL,
                models=["test/powerful-model-a"],
                max_tokens=16384,
                temperature=0.0,
            ),
            ModelTier.MAX: TierConfig(
                name=ModelTier.MAX,
                models=["test/max-model-a"],
                max_tokens=32768,
                temperature=0.0,
            ),
        },
        routing={
            Complexity.TRIVIAL: ModelTier.NANO,
            Complexity.SIMPLE: ModelTier.FAST,
            Complexity.MEDIUM: ModelTier.BALANCED,
            Complexity.COMPLEX: ModelTier.POWERFUL,
            Complexity.CRITICAL: ModelTier.MAX,
        },
        cost=CostConfig(
            log_file="logs/test-costs.jsonl",
            warn_threshold_usd=1.0,
            daily_budget_usd=50.0,
        ),
    )
