"""Tests for OpenRouterClient with mocked httpx responses."""

from __future__ import annotations

import json

import httpx
import pytest
from aiai_core.types import Complexity, ModelTier, RouteRequest
from aiai_router.client import OpenRouterClient, OpenRouterError


def _mock_response(
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost: float = 0.001,
    status_code: int = 200,
    request_id: str = "req-123",
) -> httpx.Response:
    """Build a mock httpx.Response matching OpenRouter's format."""
    body = {
        "id": request_id,
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cost": cost,
        },
    }
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def _make_request(prompt: str = "Hi", complexity: Complexity = Complexity.SIMPLE) -> RouteRequest:
    return RouteRequest(prompt=prompt, complexity=complexity)


class TestOpenRouterClient:
    @pytest.mark.asyncio
    async def test_complete_success(self) -> None:
        transport = httpx.MockTransport(
            lambda request: _mock_response(
                content="test response", input_tokens=15, output_tokens=8
            )
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="test-key", http_client=http_client)
            response = await client.complete(
                request=_make_request("Hello"),
                model="test/model",
                tier=ModelTier.FAST,
            )

        assert response.content == "test response"
        assert response.model == "test/model"
        assert response.tier == ModelTier.FAST
        assert response.input_tokens == 15
        assert response.output_tokens == 8
        assert response.latency_ms > 0
        assert response.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self) -> None:
        captured_body: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return _mock_response()

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="test-key", http_client=http_client)
            req = RouteRequest(
                prompt="Hello",
                complexity=Complexity.SIMPLE,
                system="You are helpful.",
            )
            await client.complete(request=req, model="test/model", tier=ModelTier.FAST)

        assert len(captured_body["messages"]) == 2
        assert captured_body["messages"][0]["role"] == "system"
        assert captured_body["messages"][0]["content"] == "You are helpful."
        assert captured_body["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_complete_sends_auth_header(self) -> None:
        captured_headers: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return _mock_response()

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="my-secret-key", http_client=http_client)
            await client.complete(
                request=_make_request(),
                model="test/model",
                tier=ModelTier.FAST,
            )

        assert captured_headers["authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_complete_api_error(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                status_code=500,
                text="Internal Server Error",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            )
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="test-key", http_client=http_client)
            with pytest.raises(OpenRouterError, match="500"):
                await client.complete(
                    request=_make_request(),
                    model="test/model",
                    tier=ModelTier.FAST,
                )

    @pytest.mark.asyncio
    async def test_complete_malformed_response(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                status_code=200,
                json={"choices": []},
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            )
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="test-key", http_client=http_client)
            with pytest.raises(OpenRouterError, match="Unexpected response"):
                await client.complete(
                    request=_make_request(),
                    model="test/model",
                    tier=ModelTier.FAST,
                )

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens_and_temperature(self) -> None:
        captured_body: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return _mock_response()

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenRouterClient(api_key="test-key", http_client=http_client)
            req = RouteRequest(
                prompt="Hello",
                complexity=Complexity.SIMPLE,
                max_tokens=500,
                temperature=0.7,
            )
            await client.complete(request=req, model="test/model", tier=ModelTier.FAST)

        assert captured_body["max_tokens"] == 500
        assert captured_body["temperature"] == 0.7
