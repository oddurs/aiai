"""Fallback chain with circuit breaker for model routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiai_router.client import OpenRouterClient, OpenRouterError
from aiai_router.selector import ModelSelector

if TYPE_CHECKING:
    from aiai_core.config import ModelsConfig
    from aiai_core.types import RouteRequest, RouteResponse


class AllModelsFailedError(Exception):
    """Raised when every model in the fallback chain has failed."""


@dataclass
class _CircuitState:
    """Tracks consecutive failures for a single model."""

    consecutive_failures: int = 0
    open_until: float = 0.0


class FallbackChain:
    """Tries models in priority order, skipping circuit-broken ones.

    When a model fails, the chain moves to the next model in the tier.
    After `max_failures` consecutive failures for a model, that model
    is circuit-broken (skipped) for `cooldown_seconds`.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        config: ModelsConfig,
        max_failures: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._client = client
        self._selector = ModelSelector(config)
        self._config = config
        self._max_failures = max_failures
        self._cooldown_seconds = cooldown_seconds
        self._circuits: dict[str, _CircuitState] = {}

    def _get_circuit(self, model: str) -> _CircuitState:
        if model not in self._circuits:
            self._circuits[model] = _CircuitState()
        return self._circuits[model]

    def _is_open(self, model: str) -> bool:
        """Check if a model's circuit breaker is open (should be skipped)."""
        circuit = self._get_circuit(model)
        if circuit.consecutive_failures < self._max_failures:
            return False
        return time.monotonic() < circuit.open_until

    def _record_failure(self, model: str) -> None:
        circuit = self._get_circuit(model)
        circuit.consecutive_failures += 1
        if circuit.consecutive_failures >= self._max_failures:
            circuit.open_until = time.monotonic() + self._cooldown_seconds

    def _record_success(self, model: str) -> None:
        circuit = self._get_circuit(model)
        circuit.consecutive_failures = 0
        circuit.open_until = 0.0

    async def route(self, request: RouteRequest) -> RouteResponse:
        """Route a request through the fallback chain.

        Tries each model in the tier for the request's complexity. Skips
        circuit-broken models. On success, resets the model's failure count.

        Args:
            request: The route request to send.

        Returns:
            RouteResponse from the first model that succeeds.

        Raises:
            AllModelsFailedError: If every model in the chain fails or is open.
        """
        models = self._selector.models_for_complexity(request.complexity)
        tier, _ = self._selector.select(request.complexity)
        max_tokens, temperature = self._selector.tier_defaults(request.complexity)

        errors: list[str] = []

        for model in models:
            if self._is_open(model):
                errors.append(f"{model}: circuit breaker open")
                continue

            try:
                response = await self._client.complete(
                    request=request,
                    model=model,
                    tier=tier,
                    max_tokens=request.max_tokens or max_tokens,
                    temperature=(
                        request.temperature if request.temperature is not None else temperature
                    ),
                )
                self._record_success(model)
                return response
            except OpenRouterError as exc:
                self._record_failure(model)
                errors.append(f"{model}: {exc}")

        raise AllModelsFailedError(
            f"All models failed for complexity={request.complexity.value}: "
            + "; ".join(errors)
        )
