"""aiai-router: OpenRouter client with model routing, fallback, and cost tracking."""

from aiai_router.client import OpenRouterClient, OpenRouterError
from aiai_router.cost import BudgetExceededError, CostTracker
from aiai_router.fallback import AllModelsFailedError, FallbackChain
from aiai_router.selector import ModelSelector

__all__ = [
    "OpenRouterClient",
    "OpenRouterError",
    "ModelSelector",
    "FallbackChain",
    "AllModelsFailedError",
    "CostTracker",
    "BudgetExceededError",
]
