"""Model selection based on task complexity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiai_core.config import ModelsConfig
    from aiai_core.types import Complexity, ModelTier


class ModelSelector:
    """Selects a model tier and model ID based on task complexity.

    Uses the routing table from ModelsConfig to map complexity levels
    to tiers, then picks the first available model in that tier.
    """

    def __init__(self, config: ModelsConfig) -> None:
        self._config = config

    def select(self, complexity: Complexity) -> tuple[ModelTier, str]:
        """Pick the best tier and model for a given complexity.

        Args:
            complexity: The declared task complexity.

        Returns:
            Tuple of (ModelTier, model_id) for the first model in the tier.

        Raises:
            ValueError: If no models are configured for the mapped tier.
        """
        tier_config = self._config.tier_for_complexity(complexity)
        if not tier_config.models:
            raise ValueError(f"No models configured for tier {tier_config.name.value}")
        return tier_config.name, tier_config.models[0]

    def models_for_complexity(self, complexity: Complexity) -> list[str]:
        """Return all models in the tier mapped to this complexity.

        Args:
            complexity: The declared task complexity.

        Returns:
            List of model IDs in priority order.
        """
        tier_config = self._config.tier_for_complexity(complexity)
        return list(tier_config.models)

    def tier_defaults(self, complexity: Complexity) -> tuple[int, float]:
        """Return the default max_tokens and temperature for a complexity level.

        Args:
            complexity: The declared task complexity.

        Returns:
            Tuple of (max_tokens, temperature).
        """
        tier_config = self._config.tier_for_complexity(complexity)
        return tier_config.max_tokens, tier_config.temperature
