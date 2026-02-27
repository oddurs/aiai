"""Tests for ModelSelector."""

from __future__ import annotations

import pytest
from aiai_core.config import ModelsConfig, TierConfig
from aiai_core.types import Complexity, ModelTier
from aiai_router.selector import ModelSelector


class TestModelSelector:
    def test_select_trivial(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        tier, model = selector.select(Complexity.TRIVIAL)
        assert tier == ModelTier.NANO
        assert model == "test/nano-model-a"

    def test_select_simple(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        tier, model = selector.select(Complexity.SIMPLE)
        assert tier == ModelTier.FAST
        assert model == "test/fast-model-a"

    def test_select_medium(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        tier, model = selector.select(Complexity.MEDIUM)
        assert tier == ModelTier.BALANCED
        assert model == "test/balanced-model-a"

    def test_select_complex(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        tier, model = selector.select(Complexity.COMPLEX)
        assert tier == ModelTier.POWERFUL
        assert model == "test/powerful-model-a"

    def test_select_critical(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        tier, model = selector.select(Complexity.CRITICAL)
        assert tier == ModelTier.MAX
        assert model == "test/max-model-a"

    def test_models_for_complexity(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        models = selector.models_for_complexity(Complexity.TRIVIAL)
        assert models == ["test/nano-model-a", "test/nano-model-b"]

    def test_tier_defaults(self, models_config: ModelsConfig) -> None:
        selector = ModelSelector(models_config)
        max_tokens, temperature = selector.tier_defaults(Complexity.TRIVIAL)
        assert max_tokens == 1024
        assert temperature == 0.0

    def test_select_empty_tier_raises(self) -> None:
        config = ModelsConfig(
            provider="openrouter",
            tiers={
                ModelTier.NANO: TierConfig(
                    name=ModelTier.NANO,
                    models=[],
                    max_tokens=1024,
                    temperature=0.0,
                ),
            },
            routing={Complexity.TRIVIAL: ModelTier.NANO},
            cost=models_config_cost(),
        )
        selector = ModelSelector(config)
        with pytest.raises(ValueError, match="No models configured"):
            selector.select(Complexity.TRIVIAL)


def models_config_cost():  # noqa: ANN201
    from aiai_core.config import CostConfig
    return CostConfig()
