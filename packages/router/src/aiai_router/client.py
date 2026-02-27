"""Async OpenRouter API client."""

from __future__ import annotations

import os
import time
import uuid

import httpx
from aiai_core.types import ModelTier, RouteRequest, RouteResponse

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 120.0


class OpenRouterError(Exception):
    """Base error for OpenRouter API issues."""


class OpenRouterClient:
    """Async client for the OpenRouter chat completions API.

    Sends requests to a specified model and parses the response into
    a RouteResponse with token counts, cost, and latency.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._timeout = timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=self._timeout)

    async def complete(
        self,
        request: RouteRequest,
        model: str,
        tier: ModelTier,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> RouteResponse:
        """Send a completion request and return a parsed RouteResponse.

        Args:
            request: The route request containing prompt and metadata.
            model: OpenRouter model ID (e.g. "anthropic/claude-sonnet-4").
            tier: The model tier being used.
            max_tokens: Override max tokens (uses request value if not set).
            temperature: Override temperature (uses request value if not set).

        Returns:
            RouteResponse with content, token counts, cost, and latency.

        Raises:
            OpenRouterError: On API errors or unexpected responses.
        """
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
        }

        effective_max_tokens = max_tokens or request.max_tokens
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens

        effective_temp = temperature if temperature is not None else request.temperature
        if effective_temp is not None:
            payload["temperature"] = effective_temp

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            resp = await self._client.post(
                self._base_url,
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"HTTP error calling {model}: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter returned {resp.status_code} for {model}: {resp.text}"
            )

        data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise OpenRouterError(f"Unexpected response structure from {model}: {data}") from exc

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost_usd = float(data.get("usage", {}).get("cost", 0.0) or 0.0)

        request_id = data.get("id", str(uuid.uuid4()))

        return RouteResponse(
            content=content,
            model=model,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()
