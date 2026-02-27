"""Tests for FallbackChain with circuit breaker."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from aiai_core.types import Complexity, ModelTier, RouteRequest, RouteResponse
from aiai_router.client import OpenRouterClient, OpenRouterError
from aiai_router.fallback import AllModelsFailedError, FallbackChain

if TYPE_CHECKING:
    from aiai_core.config import ModelsConfig


def _make_response(model: str = "test/model", tier: ModelTier = ModelTier.NANO) -> RouteResponse:
    return RouteResponse(
        content="result",
        model=model,
        tier=tier,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=50.0,
    )


def _make_request(complexity: Complexity = Complexity.TRIVIAL) -> RouteRequest:
    return RouteRequest(prompt="test", complexity=complexity)


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_first_model_succeeds(self, models_config: ModelsConfig) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-a"))

        chain = FallbackChain(client, models_config)
        response = await chain.route(_make_request())

        assert response.model == "test/nano-model-a"
        assert client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_to_second_model(self, models_config: ModelsConfig) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        client.complete = AsyncMock(
            side_effect=[
                OpenRouterError("model A down"),
                _make_response("test/nano-model-b"),
            ]
        )

        chain = FallbackChain(client, models_config)
        response = await chain.route(_make_request())

        assert response.model == "test/nano-model-b"
        assert client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_all_models_fail(self, models_config: ModelsConfig) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        client.complete = AsyncMock(side_effect=OpenRouterError("all down"))

        chain = FallbackChain(client, models_config)
        with pytest.raises(AllModelsFailedError, match="All models failed"):
            await chain.route(_make_request())

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_max_failures(
        self, models_config: ModelsConfig
    ) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        chain = FallbackChain(client, models_config, max_failures=2, cooldown_seconds=60.0)

        # Fail model A twice to trip the circuit breaker
        client.complete = AsyncMock(side_effect=OpenRouterError("fail"))
        with pytest.raises(AllModelsFailedError):
            await chain.route(_make_request())
        # That was 1 failure each for A and B (2 models, each failed once)

        # Fail again to get model A to 2 failures
        client.complete = AsyncMock(
            side_effect=[
                OpenRouterError("fail A again"),
                _make_response("test/nano-model-b"),
            ]
        )
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-b"

        # Now model A has 2 consecutive failures -> circuit open
        # Next call should skip A and go straight to B
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-b"))
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-b"
        # Only 1 call because A was skipped
        assert client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, models_config: ModelsConfig) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        chain = FallbackChain(client, models_config, max_failures=3)

        # Fail model A twice (not enough to trip breaker at max_failures=3)
        client.complete = AsyncMock(
            side_effect=[
                OpenRouterError("fail 1"),
                _make_response("test/nano-model-b"),
            ]
        )
        await chain.route(_make_request())

        client.complete = AsyncMock(
            side_effect=[
                OpenRouterError("fail 2"),
                _make_response("test/nano-model-b"),
            ]
        )
        await chain.route(_make_request())

        # Now succeed with model A — should reset counter
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-a"))
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-a"

        # Model A should still be available (counter was reset)
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-a"))
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-a"
        assert client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_cooldown_expires(self, models_config: ModelsConfig) -> None:
        client = AsyncMock(spec=OpenRouterClient)
        chain = FallbackChain(client, models_config, max_failures=1, cooldown_seconds=0.1)

        # Trip the circuit breaker for model A
        client.complete = AsyncMock(
            side_effect=[
                OpenRouterError("fail"),
                _make_response("test/nano-model-b"),
            ]
        )
        await chain.route(_make_request())

        # Circuit is open — should skip A
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-b"))
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-b"

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Now A should be tried again
        client.complete = AsyncMock(return_value=_make_response("test/nano-model-a"))
        response = await chain.route(_make_request())
        assert response.model == "test/nano-model-a"
