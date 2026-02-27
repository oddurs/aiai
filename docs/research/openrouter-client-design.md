# OpenRouter Client Design Document

**For: aiai -- AI that builds itself**
**Purpose: Build a production-quality Python OpenRouter client for cost-optimized model routing**
**Date: 2026-02-26**

---

## Table of Contents

1. [OpenRouter API Deep Dive](#1-openrouter-api-deep-dive)
2. [Model Routing Strategies](#2-model-routing-strategies)
3. [Cost Tracking Implementation](#3-cost-tracking-implementation)
4. [Fallback Chain Design](#4-fallback-chain-design)
5. [Prompt Caching with OpenRouter](#5-prompt-caching-with-openrouter)
6. [Async Python HTTP Patterns](#6-async-python-http-patterns)
7. [Testing OpenRouter Clients](#7-testing-openrouter-clients)
8. [Real-World OpenRouter Usage Patterns](#8-real-world-openrouter-usage-patterns)
9. [Pricing Data and Cost Models](#9-pricing-data-and-cost-models)

---

## 1. OpenRouter API Deep Dive

### 1.1 Core Endpoints

OpenRouter exposes an OpenAI-compatible API surface with additional routing and observability extensions. The base URL is `https://openrouter.ai/api/v1`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat/completions` | POST | Send chat messages, get model responses |
| `/models` | GET | List all available models with pricing and capabilities |
| `/generation?id=<gen_id>` | GET | Query generation stats (tokens, cost, latency) |
| `/credits` | GET | Check remaining credits (requires management key) |
| `/key` | GET | Check rate limits, credit balance, usage stats |

### 1.2 Authentication and Headers

Every request requires an API key in the `Authorization` header. Two additional headers identify your application to OpenRouter and appear in their dashboard.

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/your-org/aiai",  # Your app URL
    "X-Title": "aiai",  # Your app name (also accepts X-OpenRouter-Title)
}
```

The `HTTP-Referer` and `X-Title` headers are optional but recommended. They help OpenRouter attribute traffic and show up in their analytics. `X-OpenRouter-Categories` can also be set for marketplace categorization.

### 1.3 Chat Completions Request Format

The `/chat/completions` endpoint accepts an OpenAI-compatible payload with OpenRouter-specific extensions.

**Minimal request:**

```python
{
    "model": "anthropic/claude-sonnet-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quicksort."}
    ]
}
```

**Full request with all OpenRouter extensions:**

```python
{
    # === Standard OpenAI parameters ===
    "model": "anthropic/claude-sonnet-4",
    "messages": [...],
    "temperature": 0.0,           # 0.0-2.0, default 1.0
    "top_p": 1.0,                 # 0.0-1.0, nucleus sampling
    "top_k": 0,                   # 0+, top-k sampling
    "frequency_penalty": 0.0,     # -2.0 to 2.0
    "presence_penalty": 0.0,      # -2.0 to 2.0
    "repetition_penalty": 1.0,    # 0.0-2.0
    "min_p": 0.0,                 # 0.0-1.0
    "top_a": 0.0,                 # 0.0-1.0
    "max_completion_tokens": 4096,# Replaces deprecated max_tokens
    "stop": ["\n\n"],             # Up to 4 stop sequences
    "seed": 42,                   # Deterministic sampling
    "stream": False,              # SSE streaming
    "logprobs": True,             # Return log probabilities
    "top_logprobs": 5,            # 0-20, per-position top probs

    # === Tool calling ===
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    "tool_choice": "auto",          # "auto", "none", "required", or specific function
    "parallel_tool_calls": True,    # Allow simultaneous tool calls

    # === Output format ===
    "response_format": {"type": "json_object"},  # Or json_schema, text

    # === Reasoning control (for models that support it) ===
    "reasoning": {
        "effort": "high",  # xhigh, high, medium, low, minimal, none
    },

    # === OpenRouter-specific extensions ===
    "models": [                     # Fallback chain (alternative to single model)
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "google/gemini-2.5-pro"
    ],
    "provider": {                   # Provider routing preferences
        "order": ["Anthropic"],     # Try providers in this order
        "allow_fallbacks": True,    # Fall back to other providers on failure
        "require_parameters": True, # Only use providers supporting all params
        "data_collection": "deny",  # "allow" or "deny"
        "zdr": False,               # Zero Data Retention endpoints only
        "sort": "price",            # "price", "throughput", "latency"
        "quantizations": ["fp16", "bf16"],  # Filter by quantization
        "only": ["Anthropic", "Google"],    # Whitelist providers
        "ignore": ["Together"],             # Blacklist providers
        "max_price": {              # Max USD per million tokens
            "prompt": 5.0,
            "completion": 25.0
        },
        "preferred_min_throughput": {"p90": 50},  # Min tokens/sec
        "preferred_max_latency": {"p90": 2.0},    # Max seconds
    },
    "transforms": ["middle-out"],   # Truncate middle on overflow

    # === Observability ===
    "user": "agent-builder-01",     # User ID for abuse detection
    "session_id": "session-abc123", # Group requests (max 128 chars)
    "trace": {                      # Distributed tracing
        "trace_id": "trace-xyz",
        "span_name": "code-generation",
    },
    "metadata": {                   # Custom key-value pairs (max 16)
        "task_complexity": "complex",
        "agent_type": "builder",
    },
}
```

### 1.4 Chat Completions Response Format

**Non-streaming response:**

```python
{
    "id": "gen-abc123xyz",
    "object": "chat.completion",
    "created": 1740576000,
    "model": "anthropic/claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Quicksort is a divide-and-conquer sorting algorithm...",
                "tool_calls": [           # Present when model calls tools
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\": \"Oslo\"}"
                        }
                    }
                ],
                "reasoning": "...",       # If reasoning was requested
            },
            "finish_reason": "stop",      # stop, tool_calls, length, content_filter, error
            "logprobs": {
                "content": [...],
                "refusal": [...]
            }
        }
    ],
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 320,
        "total_tokens": 470,
        "cost": 0.00234,                  # USD cost for this request
        "prompt_tokens_details": {
            "cached_tokens": 100,         # Tokens read from cache
            "cache_write_tokens": 0,      # Tokens written to cache
            "audio_tokens": 0,
            "video_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    },
    "system_fingerprint": "fp_abc123"
}
```

**Streaming response (SSE format):**

When `stream: true`, the response is delivered as Server-Sent Events. Each chunk has the same structure but with `delta` instead of `message`:

```
data: {"id":"gen-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"gen-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Quick"},"finish_reason":null}]}

data: {"id":"gen-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"sort"},"finish_reason":null}]}

data: {"id":"gen-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

The `usage` object is only present in the final chunk before `[DONE]`.

### 1.5 Tool Use / Function Calling

Tool calling through OpenRouter follows the OpenAI format exactly. The flow:

1. Send request with `tools` array defining available functions
2. Model responds with `finish_reason: "tool_calls"` and `tool_calls` in the message
3. Execute the function(s) locally
4. Send results back as `tool` role messages
5. Model generates final response

```python
# Step 1: Initial request with tools
request = {
    "model": "anthropic/claude-sonnet-4",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        }
    ],
    "messages": [
        {"role": "user", "content": "What's in src/main.py?"}
    ]
}

# Step 2: Model returns tool_calls
# response.choices[0].message.tool_calls = [
#     {"id": "call_1", "type": "function",
#      "function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'}}
# ]

# Step 3: Execute function, then send result back
follow_up = {
    "model": "anthropic/claude-sonnet-4",
    "tools": [...],  # Same tools
    "messages": [
        {"role": "user", "content": "What's in src/main.py?"},
        {"role": "assistant", "tool_calls": [...]},  # Model's tool call
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "def main():\n    print('hello')"
        }
    ]
}
```

The `tool_choice` parameter controls tool calling behavior:
- `"auto"` -- model decides whether to call tools (default)
- `"none"` -- model never calls tools
- `"required"` -- model must call at least one tool
- `{"type": "function", "function": {"name": "read_file"}}` -- force a specific tool

### 1.6 Provider Routing

The `provider` object in the request body controls how OpenRouter routes across infrastructure providers. This is distinct from model selection -- it controls which provider hosts the model.

**Default behavior:** Load-balances across providers, prioritizing price. Uses inverse-square-of-price weighting.

**Sort modes:**
- `"price"` -- cheapest provider first (equivalent to `:floor` suffix)
- `"throughput"` -- fastest tokens/sec first (equivalent to `:nitro` suffix)
- `"latency"` -- lowest latency first

**Model slug shortcuts:**
```python
# These are equivalent:
model = "anthropic/claude-sonnet-4:floor"
# vs.
model = "anthropic/claude-sonnet-4"
provider = {"sort": "price"}

model = "anthropic/claude-sonnet-4:nitro"
# vs.
model = "anthropic/claude-sonnet-4"
provider = {"sort": "throughput"}
```

**Performance thresholds** use percentile-based evaluation over rolling 5-minute windows (p50, p75, p90, p99). Endpoints that do not meet thresholds are deprioritized (moved to end of list) rather than excluded entirely.

**Provider enforcement examples:**

```python
# Only Anthropic, no fallbacks
provider = {
    "order": ["Anthropic"],
    "allow_fallbacks": False
}

# Cheapest across Azure or Google, with performance floor
provider = {
    "only": ["Azure", "Google"],
    "sort": "price",
    "preferred_min_throughput": {"p90": 50}
}

# Exclude specific providers, require ZDR
provider = {
    "ignore": ["DeepInfra", "Together"],
    "zdr": True,
    "data_collection": "deny"
}
```

### 1.7 The Models Endpoint

`GET /api/v1/models` returns metadata and pricing for every available model.

```python
{
    "data": [
        {
            "id": "anthropic/claude-sonnet-4",
            "name": "Claude Sonnet 4",
            "created": 1720000000,
            "description": "...",
            "context_length": 200000,
            "pricing": {
                "prompt": "0.000003",       # USD per token (string to avoid float issues)
                "completion": "0.000015",
                "request": "0",
                "image": "0",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
                "internal_reasoning": "0",
                "discount": 0
            },
            "architecture": {
                "tokenizer": "claude",
                "instruct_type": "claude",
                "modality": "text+image->text",
                "input_modalities": ["text", "image", "file"],
                "output_modalities": ["text"]
            },
            "top_provider": {
                "context_length": 200000,
                "max_completion_tokens": 16384,
                "is_moderated": false
            },
            "per_request_limits": {
                "prompt_tokens": 200000,
                "completion_tokens": 16384
            },
            "supported_parameters": [
                "temperature", "top_p", "top_k", "frequency_penalty",
                "presence_penalty", "stop", "max_tokens", "tools",
                "tool_choice", "response_format", "seed"
            ],
            "default_parameters": {
                "temperature": 1.0,
                "top_p": null,
                "frequency_penalty": null
            }
        }
    ]
}
```

Pricing values are strings (not floats) to avoid floating-point precision issues. They represent USD per token. To get per-million-token pricing: multiply by 1,000,000.

### 1.8 The Generation Stats Endpoint

After a request completes, query `GET /api/v1/generation?id=<gen_id>` for detailed stats.

```python
{
    "data": {
        "id": "gen-abc123",
        "model": "anthropic/claude-sonnet-4",
        "streamed": true,
        "generation_time": 2450,        # milliseconds
        "created_at": "2026-02-26T10:00:00Z",
        "tokens_prompt": 150,
        "tokens_completion": 320,
        "native_tokens_prompt": 155,    # Provider's own token count
        "native_tokens_completion": 318,
        "native_tokens_cached": 100,
        "native_tokens_reasoning": null,
        "total_cost": 0.00234,
        "cache_discount": -0.0003,      # Negative = savings from caching
        "finish_reason": "stop",
        "native_finish_reason": "end_turn",
        "provider_name": "Anthropic",
        "latency": 2600,               # Total ms including network
        "is_byok": false,
        "origin": "https://github.com/your-org/aiai",
        "cancelled": false,
        "provider_responses": [...]     # Fallback attempt details
    }
}
```

This endpoint is especially useful for:
- Auditing costs after streaming requests (where usage may not arrive in the final chunk)
- Getting native token counts from the provider's tokenizer
- Inspecting fallback attempt history via `provider_responses`
- Calculating cache savings via `cache_discount`

### 1.9 The Key/Credits Endpoints

**Check API key info and rate limits:**

```python
# GET /api/v1/key
# Headers: Authorization: Bearer <api_key>
{
    "data": {
        "label": "aiai-production",
        "limit": 100.0,
        "usage": 42.5,
        "is_free_tier": false,
        # ... additional usage breakdowns
    }
}
```

**Check credit balance (management key required):**

```python
# GET /api/v1/credits
# Headers: Authorization: Bearer <management_key>
{
    "data": {
        "total_credits": 200.0,
        "total_usage": 42.5
    }
}
```

Remaining balance: `total_credits - total_usage`. A negative balance triggers 402 errors on all requests, even free models.

### 1.10 HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Parse response |
| 400 | Invalid parameters | Fix request, do not retry |
| 401 | Authentication failure | Check API key |
| 402 | Insufficient credits | Top up balance |
| 404 | Resource not found | Check model ID / generation ID |
| 408 | Request timeout | Retry with backoff |
| 413 | Payload too large | Reduce input size |
| 422 | Semantic validation error | Fix request structure |
| 429 | Rate limit exceeded | Retry with exponential backoff |
| 500 | Internal server error | Retry with backoff |
| 502 | Provider failure | Retry or fallback to different provider |
| 503 | Service unavailable | Retry with backoff |

---

## 2. Model Routing Strategies

### 2.1 Complexity-Based Routing

The core routing strategy for aiai: agents declare task complexity, the router maps it to a model tier, the tier defines a fallback chain.

This is already defined in `config/models.yaml`:

```yaml
routing:
  trivial: nano      # Haiku, Flash-Lite
  simple: fast       # Sonnet, Flash, GPT-4o-mini
  medium: balanced   # Sonnet, DeepSeek-R1, GPT-4o
  complex: powerful  # Opus, GPT-4o, DeepSeek-R1
  critical: max      # Opus, o1-pro
```

**Implementation pattern:**

```python
from enum import Enum
from dataclasses import dataclass

class Complexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"

@dataclass
class ModelTier:
    name: str
    models: list[str]       # Ordered fallback chain
    max_tokens: int
    temperature: float

class Router:
    def __init__(self, config: dict):
        self.tiers = self._load_tiers(config["tiers"])
        self.routing = config["routing"]

    def select(self, complexity: Complexity) -> ModelTier:
        tier_name = self.routing[complexity.value]
        return self.tiers[tier_name]

    def get_primary_model(self, complexity: Complexity) -> str:
        return self.select(complexity).models[0]

    def get_fallback_chain(self, complexity: Complexity) -> list[str]:
        return self.select(complexity).models
```

### 2.2 Cascading: Try Cheap First, Escalate on Failure

Cascading is different from fallback chains. Fallbacks handle infrastructure failures (429, 5xx, timeouts). Cascading handles quality failures -- the cheap model returned something, but it was not good enough.

**When to escalate:**
- Model returns empty or truncated content
- Model says "I can't help with that" or refuses the task
- JSON output fails schema validation
- Code output has obvious syntax errors (detectable without execution)
- Confidence score (when available via logprobs) is below threshold
- Output length is suspiciously short for the task

**Cascade implementation:**

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class CascadeResult:
    content: str
    model_used: str
    attempts: list[dict]
    escalated: bool

REFUSAL_PATTERNS = [
    r"(?i)i('m| am) (unable|not able|cannot|can't)",
    r"(?i)i (don't|do not) (have|know)",
    r"(?i)as an ai",
    r"(?i)i('m| am) sorry.{0,20}(can't|cannot|unable)",
]

def is_refusal(content: str) -> bool:
    return any(re.search(p, content) for p in REFUSAL_PATTERNS)

def is_quality_sufficient(content: str, task_type: str) -> bool:
    """Quick quality check without calling another model."""
    if not content or len(content.strip()) < 10:
        return False
    if is_refusal(content):
        return False
    if task_type == "json":
        try:
            import json
            json.loads(content)
        except json.JSONDecodeError:
            return False
    if task_type == "code":
        try:
            compile(content, "<string>", "exec")
        except SyntaxError:
            return False
    return True

async def cascade(
    client: "OpenRouterClient",
    messages: list[dict],
    tiers: list[str],          # e.g. ["nano", "fast", "balanced"]
    task_type: str = "text",
) -> CascadeResult:
    attempts = []
    for tier_name in tiers:
        tier = client.router.tiers[tier_name]
        model = tier.models[0]
        response = await client.chat(
            model=model,
            messages=messages,
            max_tokens=tier.max_tokens,
        )
        content = response.choices[0].message.content
        attempts.append({"model": model, "tier": tier_name, "content": content})

        if is_quality_sufficient(content, task_type):
            return CascadeResult(
                content=content,
                model_used=model,
                attempts=attempts,
                escalated=len(attempts) > 1,
            )

    # Final attempt already used the best tier
    return CascadeResult(
        content=attempts[-1]["content"],
        model_used=attempts[-1]["model"],
        attempts=attempts,
        escalated=True,
    )
```

### 2.3 RouteLLM-Style Quality Routing

RouteLLM (LMSYS, ICLR 2025) trains a classifier on preference data to predict whether a query needs a strong or weak model. Their matrix factorization router achieved 95% of GPT-4 quality using only 26% GPT-4 calls (48% cost reduction). With data augmentation, 95% quality with 14% strong model calls (75% cost reduction).

**For aiai, a simpler approach works:** Use heuristics first, graduate to a learned router later.

```python
def estimate_complexity(messages: list[dict]) -> Complexity:
    """Heuristic complexity estimation from message content."""
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            break

    total_tokens_approx = sum(
        len(str(m.get("content", ""))) // 4 for m in messages
    )

    # Long context suggests complexity
    if total_tokens_approx > 50000:
        return Complexity.COMPLEX

    text = last_user_msg.lower()

    # Trivial patterns
    if any(p in text for p in ["format this", "rename", "fix typo", "fix indent"]):
        return Complexity.TRIVIAL

    # Simple patterns
    if any(p in text for p in ["summarize", "explain", "translate", "list"]):
        return Complexity.SIMPLE

    # Complex patterns
    if any(p in text for p in ["architect", "design", "refactor", "security",
                                "debug", "performance", "optimize"]):
        return Complexity.COMPLEX

    # Critical patterns
    if any(p in text for p in ["system-wide", "breaking change", "migration",
                                "core architecture"]):
        return Complexity.CRITICAL

    # Default to medium
    return Complexity.MEDIUM
```

### 2.4 Latency-Aware Routing

OpenRouter's provider routing supports latency preferences. Use this for interactive vs. batch workloads:

```python
def get_provider_config(latency_sensitive: bool = False) -> dict:
    if latency_sensitive:
        return {
            "sort": "latency",
            "preferred_max_latency": {"p90": 3.0},  # 3 second max
            "preferred_min_throughput": {"p90": 80},  # 80 tokens/sec min
        }
    else:
        return {
            "sort": "price",
            "allow_fallbacks": True,
        }
```

### 2.5 OpenRouter Auto Router

OpenRouter has a built-in auto router at `model: "openrouter/auto"` powered by Not Diamond. It analyzes prompt complexity and selects the optimal model. No extra fee beyond the selected model's pricing.

For aiai, we should **not** use auto router as the primary strategy because:
- We need deterministic, auditable routing decisions
- We need to control costs at the tier level
- We need to log which model was selected and why

However, auto router is useful as a baseline comparison during evaluation. Run the same prompts through our router and auto router, compare quality and cost.

---

## 3. Cost Tracking Implementation

### 3.1 Extracting Cost from Responses

Every chat completion response includes a `usage` object with token counts and cost.

```python
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import json

@dataclass
class UsageRecord:
    request_id: str
    model: str
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    complexity: str = ""
    agent_id: str = ""
    session_id: str = ""
    latency_ms: Optional[float] = None

    def to_jsonl(self) -> str:
        return json.dumps({
            "request_id": self.request_id,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "complexity": self.complexity,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
        })

def extract_usage(response: dict) -> UsageRecord:
    usage = response.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details", {})
    completion_details = usage.get("completion_tokens_details", {})

    return UsageRecord(
        request_id=response["id"],
        model=response["model"],
        timestamp=datetime.utcnow(),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        cost_usd=usage.get("cost", 0.0),
        cached_tokens=prompt_details.get("cached_tokens", 0),
        cache_write_tokens=prompt_details.get("cache_write_tokens", 0),
        reasoning_tokens=completion_details.get("reasoning_tokens", 0),
    )
```

### 3.2 JSONL Cost Logger

Log every request to a JSONL file for analysis.

```python
import asyncio
import aiofiles
from pathlib import Path
from typing import Optional

class CostLogger:
    def __init__(self, log_path: str = "logs/model-costs.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, record: UsageRecord) -> None:
        async with self._lock:
            async with aiofiles.open(self.log_path, "a") as f:
                await f.write(record.to_jsonl() + "\n")

    def log_sync(self, record: UsageRecord) -> None:
        with open(self.log_path, "a") as f:
            f.write(record.to_jsonl() + "\n")

    async def get_daily_spend(self, target_date: Optional[date] = None) -> float:
        """Calculate total spend for a given day."""
        target = target_date or date.today()
        total = 0.0
        try:
            async with aiofiles.open(self.log_path, "r") as f:
                async for line in f:
                    record = json.loads(line)
                    record_date = datetime.fromisoformat(
                        record["timestamp"]
                    ).date()
                    if record_date == target:
                        total += record.get("cost_usd", 0.0)
        except FileNotFoundError:
            pass
        return total

    async def get_model_spend(self, model: str) -> float:
        """Calculate total spend for a specific model."""
        total = 0.0
        try:
            async with aiofiles.open(self.log_path, "r") as f:
                async for line in f:
                    record = json.loads(line)
                    if record.get("model") == model:
                        total += record.get("cost_usd", 0.0)
        except FileNotFoundError:
            pass
        return total
```

### 3.3 Budget Enforcement

Check budget before every request. Reject if daily budget would be exceeded.

```python
class BudgetError(Exception):
    """Raised when a request would exceed the daily budget."""
    pass

class CostWarning(Exception):
    """Raised when a single request exceeds the warning threshold."""
    pass

class BudgetEnforcer:
    def __init__(
        self,
        daily_budget_usd: float = 50.0,
        warn_threshold_usd: float = 1.0,
        logger: Optional[CostLogger] = None,
    ):
        self.daily_budget_usd = daily_budget_usd
        self.warn_threshold_usd = warn_threshold_usd
        self.logger = logger or CostLogger()
        self._daily_cache: dict[date, float] = {}

    async def check_budget(self) -> float:
        """Return remaining budget for today. Raises BudgetError if exhausted."""
        today = date.today()
        if today not in self._daily_cache:
            self._daily_cache[today] = await self.logger.get_daily_spend(today)
        spent = self._daily_cache[today]
        remaining = self.daily_budget_usd - spent
        if remaining <= 0:
            raise BudgetError(
                f"Daily budget exhausted. Spent ${spent:.4f} of ${self.daily_budget_usd}"
            )
        return remaining

    def record_spend(self, cost_usd: float) -> None:
        """Update in-memory daily spend cache."""
        today = date.today()
        self._daily_cache[today] = self._daily_cache.get(today, 0.0) + cost_usd
        if cost_usd > self.warn_threshold_usd:
            import logging
            logging.warning(
                f"Single request cost ${cost_usd:.4f} exceeds "
                f"warning threshold ${self.warn_threshold_usd}"
            )
```

### 3.4 Pre-Request Cost Estimation

Estimate cost before making a call so we can check against budget limits and choose the cheapest adequate model.

```python
class CostEstimator:
    def __init__(self):
        self._pricing_cache: dict[str, dict] = {}

    async def load_pricing(self, client: "OpenRouterClient") -> None:
        """Fetch pricing for all models from /api/v1/models."""
        response = await client.get_models()
        for model in response["data"]:
            self._pricing_cache[model["id"]] = {
                "prompt": float(model["pricing"]["prompt"]),
                "completion": float(model["pricing"]["completion"]),
                "cache_read": float(model["pricing"].get("input_cache_read", "0")),
                "cache_write": float(model["pricing"].get("input_cache_write", "0")),
            }

    def estimate(
        self,
        model: str,
        prompt_tokens: int,
        max_completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Estimate worst-case cost in USD for a request."""
        pricing = self._pricing_cache.get(model)
        if not pricing:
            return float("inf")  # Unknown model, assume expensive

        uncached_prompt = prompt_tokens - cached_tokens
        prompt_cost = (uncached_prompt * pricing["prompt"]) + \
                      (cached_tokens * pricing["cache_read"])
        completion_cost = max_completion_tokens * pricing["completion"]

        return prompt_cost + completion_cost

    def cheapest_model(
        self,
        models: list[str],
        prompt_tokens: int,
        max_completion_tokens: int,
    ) -> str:
        """Select the cheapest model from a list for the given token counts."""
        costs = [
            (self.estimate(m, prompt_tokens, max_completion_tokens), m)
            for m in models
        ]
        costs.sort(key=lambda x: x[0])
        return costs[0][1]

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4
```

### 3.5 Cost from the Generation Endpoint

For precise cost tracking (especially after streaming), query the generation endpoint:

```python
async def get_generation_cost(
    client: "OpenRouterClient",
    generation_id: str,
) -> dict:
    """Fetch exact cost and token counts from OpenRouter."""
    response = await client._http.get(
        "https://openrouter.ai/api/v1/generation",
        params={"id": generation_id},
    )
    data = response.json()["data"]
    return {
        "total_cost": data["total_cost"],
        "tokens_prompt": data["tokens_prompt"],
        "tokens_completion": data["tokens_completion"],
        "native_tokens_cached": data.get("native_tokens_cached", 0),
        "cache_discount": data.get("cache_discount", 0),
        "generation_time_ms": data.get("generation_time"),
        "provider": data.get("provider_name"),
    }
```

---

## 4. Fallback Chain Design

### 4.1 Failure Classification

Not all failures should be handled the same way. Classify failures to determine the correct response.

```python
from enum import Enum

class FailureType(Enum):
    RATE_LIMIT = "rate_limit"         # 429 -- retry with backoff or try another provider
    SERVER_ERROR = "server_error"     # 500/502/503 -- retry with backoff
    TIMEOUT = "timeout"              # Request timed out -- retry or try faster model
    CONTEXT_LENGTH = "context_length" # Input too long -- try model with larger context
    CONTENT_FILTER = "content_filter" # Moderation rejection -- try different provider
    AUTH_ERROR = "auth_error"         # 401/402 -- do not retry, fix credentials/credits
    BAD_REQUEST = "bad_request"       # 400/422 -- do not retry, fix request
    UNKNOWN = "unknown"

def classify_failure(status_code: int, response_body: dict | None = None) -> FailureType:
    if status_code == 429:
        return FailureType.RATE_LIMIT
    elif status_code in (500, 502, 503):
        return FailureType.SERVER_ERROR
    elif status_code == 408:
        return FailureType.TIMEOUT
    elif status_code in (401, 402):
        return FailureType.AUTH_ERROR
    elif status_code in (400, 422):
        # Check if it's a context length error
        if response_body:
            error_msg = str(response_body.get("error", {}).get("message", "")).lower()
            if "context" in error_msg or "token" in error_msg or "length" in error_msg:
                return FailureType.CONTEXT_LENGTH
            if "content" in error_msg and "filter" in error_msg:
                return FailureType.CONTENT_FILTER
        return FailureType.BAD_REQUEST
    elif status_code == 413:
        return FailureType.CONTEXT_LENGTH
    else:
        return FailureType.UNKNOWN

# Which failures are retryable on the same model?
RETRYABLE_SAME_MODEL = {
    FailureType.RATE_LIMIT,
    FailureType.SERVER_ERROR,
    FailureType.TIMEOUT,
}

# Which failures should trigger fallback to a different model?
FALLBACK_TO_NEXT = {
    FailureType.RATE_LIMIT,
    FailureType.SERVER_ERROR,
    FailureType.TIMEOUT,
    FailureType.CONTEXT_LENGTH,
    FailureType.CONTENT_FILTER,
}

# Which failures should stop all retries?
NON_RETRYABLE = {
    FailureType.AUTH_ERROR,
    FailureType.BAD_REQUEST,
}
```

### 4.2 Retry Strategy

Exponential backoff with jitter prevents thundering herd problems and respects rate limits.

```python
import random
import asyncio
import httpx
from typing import Optional

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.5,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and full jitter."""
        exp_delay = self.base_delay * (2 ** attempt)
        capped = min(exp_delay, self.max_delay)
        jitter = random.uniform(0, capped * self.jitter_factor)
        return capped + jitter

async def retry_request(
    func,
    retry_config: RetryConfig,
    *args,
    **kwargs,
) -> httpx.Response:
    """Retry a request with exponential backoff."""
    last_error = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            response = await func(*args, **kwargs)
            if response.status_code == 200:
                return response

            failure = classify_failure(response.status_code)
            if failure in NON_RETRYABLE:
                raise RequestError(
                    f"Non-retryable error: {response.status_code}",
                    status_code=response.status_code,
                    body=response.json() if response.content else None,
                )
            if attempt < retry_config.max_retries and failure in RETRYABLE_SAME_MODEL:
                delay = retry_config.get_delay(attempt)
                # Respect Retry-After header if present
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    delay = max(delay, float(retry_after))
                await asyncio.sleep(delay)
                continue

            last_error = response
            break

        except httpx.TimeoutException:
            if attempt < retry_config.max_retries:
                delay = retry_config.get_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise

    if last_error:
        raise RequestError(
            f"Request failed after {retry_config.max_retries + 1} attempts",
            status_code=last_error.status_code,
            body=last_error.json() if last_error.content else None,
        )

class RequestError(Exception):
    def __init__(self, message: str, status_code: int = 0, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
```

### 4.3 Fallback Chain Execution

When retries on the primary model are exhausted, fall through to the next model in the chain.

```python
from dataclasses import dataclass

@dataclass
class FallbackResult:
    response: dict
    model_used: str
    models_tried: list[str]
    total_attempts: int
    fallback_triggered: bool

async def execute_with_fallback(
    client: "OpenRouterClient",
    models: list[str],
    messages: list[dict],
    retry_config: RetryConfig = RetryConfig(),
    **kwargs,
) -> FallbackResult:
    """Try each model in sequence until one succeeds."""
    models_tried = []
    total_attempts = 0

    for model in models:
        models_tried.append(model)
        try:
            for attempt in range(retry_config.max_retries + 1):
                total_attempts += 1
                try:
                    response = await client._raw_chat(
                        model=model,
                        messages=messages,
                        **kwargs,
                    )
                    return FallbackResult(
                        response=response,
                        model_used=model,
                        models_tried=models_tried,
                        total_attempts=total_attempts,
                        fallback_triggered=len(models_tried) > 1,
                    )
                except RequestError as e:
                    failure = classify_failure(e.status_code)
                    if failure in NON_RETRYABLE:
                        raise  # Don't try other models for auth/bad request errors
                    if attempt < retry_config.max_retries and failure in RETRYABLE_SAME_MODEL:
                        delay = retry_config.get_delay(attempt)
                        await asyncio.sleep(delay)
                        continue
                    break  # Move to next model
                except httpx.TimeoutException:
                    if attempt < retry_config.max_retries:
                        delay = retry_config.get_delay(attempt)
                        await asyncio.sleep(delay)
                        continue
                    break  # Move to next model

        except RequestError:
            raise  # Non-retryable, propagate

    raise RequestError(
        f"All models exhausted: {models_tried}",
        status_code=503,
    )
```

### 4.4 Circuit Breaker

Stop sending traffic to persistently failing models. The circuit breaker has three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing recovery).

```python
import time
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,  # seconds
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

class CircuitBreakerRegistry:
    """One circuit breaker per model."""

    def __init__(self, **breaker_kwargs):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._kwargs = breaker_kwargs

    def get(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(**self._kwargs)
        return self._breakers[model]

    def available_models(self, models: list[str]) -> list[str]:
        """Filter models to only those with non-open circuit breakers."""
        return [m for m in models if self.get(m).allow_request()]
```

### 4.5 Using OpenRouter's Built-in Fallbacks

OpenRouter supports model-level fallbacks natively via the `models` array. This is simpler than client-side fallback but gives less control.

```python
# Server-side fallback via OpenRouter
request = {
    "model": "anthropic/claude-sonnet-4",  # Primary
    "models": [                            # Fallback chain
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
    ],
    "provider": {
        "allow_fallbacks": True,
    },
    "messages": [...]
}
# OpenRouter tries each model in order.
# The response.model field tells you which model was actually used.
# You're billed for the model that served the response, not the primary.
```

**When to use server-side vs. client-side fallback:**

| Feature | Server-side (OpenRouter `models`) | Client-side |
|---------|----------------------------------|-------------|
| Latency | Lower (single network call) | Higher (multiple round trips) |
| Control | Limited (OpenRouter decides when to failover) | Full (you decide) |
| Retry logic | OpenRouter defaults | Custom backoff, jitter, circuit breakers |
| Observability | Check `response.model` | Full attempt logging |
| Cost tracking | Must check response after | Can estimate before each attempt |

**Recommendation for aiai:** Use client-side fallback for full control and observability. Use server-side fallback as a secondary safety net by always including 2-3 models in the `models` array even when doing client-side fallback.

---

## 5. Prompt Caching with OpenRouter

### 5.1 How Prompt Caching Works

Prompt caching stores the KV cache of processed prompt prefixes. When subsequent requests share the same prefix, computation resumes from the cached point instead of reprocessing. This is different from response caching -- the prompt is still processed, just faster and cheaper.

### 5.2 Provider-Specific Behavior

**Automatic caching (no configuration needed):**
- OpenAI -- all requests cached automatically; 50% discount on cache reads
- DeepSeek -- automatic; ~90% discount on cache hits
- Gemini 2.5 -- implicit caching for Pro and Flash
- Groq -- automatic
- Grok -- automatic

**Manual caching (requires `cache_control` breakpoints):**
- Anthropic Claude -- must add `cache_control` per message content block
- Minimum token requirements vary by model (1024 to 4096 tokens)
- Maximum 4 cache breakpoints per request

### 5.3 Anthropic Cache Control Syntax

To enable caching for Anthropic models through OpenRouter, add `cache_control` to message content blocks:

```python
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": LARGE_SYSTEM_PROMPT,  # Must be >= 1024 tokens
                "cache_control": {"type": "ephemeral"}  # 5-minute TTL
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": LARGE_CONTEXT_DOCUMENT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}  # 1-hour TTL
            }
        ]
    },
    {
        "role": "user",
        "content": "Now analyze the above document."
    }
]
```

**TTL options and pricing (Anthropic):**

| TTL | Cache Write Cost | Cache Read Cost | Best For |
|-----|-----------------|-----------------|----------|
| 5 minutes (default) | 1.25x base input | 0.1x base input (90% off) | Short conversations, quick iterations |
| 1 hour | 2x base input | 0.1x base input (90% off) | Long sessions, persistent context |

The 1-hour TTL is supported on Opus 4.5, Opus 4.1, Opus 4, Sonnet 4.5, Sonnet 4, Haiku 4.5, and Haiku 3.5 across all providers (Anthropic, AWS Bedrock, Google Vertex AI).

### 5.4 Structuring Prompts for Maximum Cache Hits

Cache matching works on **prefixes** -- the beginning of your prompt must be identical across requests for the cache to hit. Structure prompts with the most stable content first.

**Optimal prompt structure for caching:**

```
[1] Tools / function definitions     <- Most stable, cache this
[2] System prompt                    <- Very stable, cache this
[3] Long reference documents / RAG   <- Stable per session, cache this
[4] Conversation history             <- Changes every turn
[5] Current user message             <- Changes every turn
```

```python
def build_cached_messages(
    system_prompt: str,
    tools_description: str,
    reference_docs: str,
    conversation: list[dict],
    user_message: str,
) -> list[dict]:
    """Structure messages for maximum cache hits with Anthropic."""
    messages = []

    # System message with tools + system prompt (most stable, always cached)
    system_content = []
    if tools_description:
        system_content.append({
            "type": "text",
            "text": tools_description,
            "cache_control": {"type": "ephemeral"},  # Breakpoint 1
        })
    system_content.append({
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},  # Breakpoint 2
    })
    messages.append({"role": "system", "content": system_content})

    # Reference docs as a user message (stable per session)
    if reference_docs:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Reference documentation:\n\n{reference_docs}",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},  # Breakpoint 3
                }
            ]
        })
        messages.append({
            "role": "assistant",
            "content": "I've reviewed the reference documentation. What would you like me to do?"
        })

    # Conversation history (changes every turn, not cached)
    messages.extend(conversation)

    # Current user message
    messages.append({"role": "user", "content": user_message})

    return messages
```

### 5.5 Provider Sticky Routing

OpenRouter uses sticky routing to maximize cache hits. After a cached request, subsequent requests are routed to the same provider endpoint. This happens automatically when:
- Cache read pricing beats regular input pricing
- The account + model + conversation combination matches
- Conversation is identified by hashing the first system and user messages

If you set `provider.order` manually, your preference is respected over sticky routing.

### 5.6 Cache Cost Savings Calculation

```python
def calculate_cache_savings(
    prompt_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
    model_input_price: float,  # USD per token
) -> dict:
    """Calculate actual savings from prompt caching."""
    # Without caching: all tokens at full price
    cost_without_cache = prompt_tokens * model_input_price

    # With caching:
    # - Uncached tokens at full price
    # - Cache write tokens at 1.25x (5min TTL) or 2x (1h TTL)
    # - Cache read tokens at 0.1x
    uncached = prompt_tokens - cached_tokens - cache_write_tokens
    cost_uncached = uncached * model_input_price
    cost_cache_write = cache_write_tokens * model_input_price * 1.25
    cost_cache_read = cached_tokens * model_input_price * 0.1
    cost_with_cache = cost_uncached + cost_cache_write + cost_cache_read

    savings = cost_without_cache - cost_with_cache
    pct_savings = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0

    return {
        "cost_without_cache": cost_without_cache,
        "cost_with_cache": cost_with_cache,
        "savings_usd": savings,
        "savings_pct": pct_savings,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }
```

**Typical savings in an agentic workflow:**

For aiai, system prompts + tool definitions are ~10K tokens. With 90% cache read discount:

```
Turn 1: 10K tokens at 1.25x = $0.0000375/tok (cache write)
Turn 2+: 10K tokens at 0.1x  = $0.0000003/tok (cache read)

Without caching: 10K * $0.000003 = $0.03/turn  (Claude Sonnet 4)
With caching:    10K * $0.0000003 = $0.003/turn (after initial write)

Savings per turn: $0.027 (90%)
Over 100 turns: $2.70 saved on system prompt alone
```

---

## 6. Async Python HTTP Patterns

### 6.1 httpx AsyncClient Configuration

httpx is the recommended HTTP library for async Python. It supports HTTP/2, connection pooling, and has a clean async API.

```python
import httpx

def create_http_client(
    api_key: str,
    app_url: str = "https://github.com/your-org/aiai",
    app_name: str = "aiai",
) -> httpx.AsyncClient:
    """Create a properly configured httpx async client."""
    return httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,
            "X-Title": app_name,
        },
        timeout=httpx.Timeout(
            connect=10.0,    # Max seconds to establish connection
            read=120.0,      # Max seconds to receive response
            write=30.0,      # Max seconds to send request
            pool=10.0,       # Max seconds waiting for connection from pool
        ),
        limits=httpx.Limits(
            max_connections=100,         # Total connections in pool
            max_keepalive_connections=20, # Keep-alive connections
            keepalive_expiry=30.0,       # Seconds before idle conn expires
        ),
        transport=httpx.AsyncHTTPTransport(
            retries=1,  # Retry on connection errors only (not HTTP errors)
        ),
        follow_redirects=True,
    )
```

### 6.2 Timeout Configuration Details

LLM inference is slow. Standard web API timeouts (5-10 seconds) will kill valid requests.

```python
# For different workload types:

INTERACTIVE_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=60.0,     # 60s for interactive responses
    write=10.0,
    pool=5.0,
)

BATCH_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=300.0,    # 5 minutes for long generations
    write=30.0,
    pool=10.0,
)

STREAMING_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=5.0,      # 5s between chunks (not total)
    write=10.0,
    pool=5.0,
)
```

For streaming, the `read` timeout applies between chunks, not for the total response. So 5 seconds is appropriate -- if you don't receive a chunk within 5 seconds, something is wrong.

### 6.3 The Client Class Pattern

Structure the client as a context manager for proper resource cleanup.

```python
import httpx
import json
import logging
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Async OpenRouter client with connection pooling, retry, and cost tracking."""

    def __init__(
        self,
        api_key: str,
        app_url: str = "https://github.com/your-org/aiai",
        app_name: str = "aiai",
        timeout: Optional[httpx.Timeout] = None,
    ):
        self._api_key = api_key
        self._app_url = app_url
        self._app_name = app_name
        self._timeout = timeout or httpx.Timeout(
            connect=10.0, read=120.0, write=30.0, pool=10.0
        )
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OpenRouterClient":
        self._http = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._app_url,
                "X-Title": self._app_name,
            },
            timeout=self._timeout,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            transport=httpx.AsyncHTTPTransport(retries=1),
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Send a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Stream a chat completion response."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        async with self._http.stream(
            "POST",
            "/chat/completions",
            json=payload,
            timeout=httpx.Timeout(connect=10.0, read=5.0, write=10.0, pool=5.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield json.loads(data)

    async def get_models(self) -> dict:
        """Fetch available models and pricing."""
        response = await self._http.get("/models")
        response.raise_for_status()
        return response.json()

    async def get_generation(self, generation_id: str) -> dict:
        """Fetch generation stats."""
        response = await self._http.get(
            "/generation",
            params={"id": generation_id},
        )
        response.raise_for_status()
        return response.json()

    async def get_key_info(self) -> dict:
        """Check API key rate limits and usage."""
        response = await self._http.get("/key")
        response.raise_for_status()
        return response.json()
```

### 6.4 Streaming with Content Accumulation

For streaming, accumulate the full response while yielding chunks for real-time display:

```python
@dataclass
class StreamResult:
    content: str
    model: str
    finish_reason: str
    usage: dict
    generation_id: str
    chunks: list[dict]

async def stream_with_accumulation(
    client: OpenRouterClient,
    model: str,
    messages: list[dict],
    on_chunk: Optional[callable] = None,
    **kwargs,
) -> StreamResult:
    """Stream response while accumulating full content."""
    content_parts = []
    generation_id = ""
    model_used = model
    finish_reason = ""
    usage = {}
    chunks = []

    async for chunk in client.chat_stream(model, messages, **kwargs):
        chunks.append(chunk)
        generation_id = chunk.get("id", generation_id)
        model_used = chunk.get("model", model_used)

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
                if on_chunk:
                    on_chunk(delta["content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        if "usage" in chunk:
            usage = chunk["usage"]

    return StreamResult(
        content="".join(content_parts),
        model=model_used,
        finish_reason=finish_reason,
        usage=usage,
        generation_id=generation_id,
        chunks=chunks,
    )
```

### 6.5 Logging and Observability

Structured logging for every request:

```python
import time
import logging

logger = logging.getLogger("openrouter")

class ObservableClient(OpenRouterClient):
    """Wraps OpenRouterClient with logging and metrics."""

    async def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        start = time.monotonic()
        request_id = None
        try:
            response = await super().chat(model, messages, **kwargs)
            request_id = response.get("id")
            elapsed = time.monotonic() - start
            usage = response.get("usage", {})

            logger.info(
                "openrouter_request",
                extra={
                    "request_id": request_id,
                    "model": response.get("model", model),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    "cost_usd": usage.get("cost", 0),
                    "latency_ms": round(elapsed * 1000, 1),
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                    "status": "success",
                },
            )
            return response

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(
                "openrouter_request",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": round(elapsed * 1000, 1),
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise
```

---

## 7. Testing OpenRouter Clients

### 7.1 respx for httpx Mocking

respx is the standard mocking library for httpx. It intercepts HTTP requests and returns canned responses.

```python
# Install: pip install respx pytest-asyncio

import pytest
import respx
import httpx

@pytest.fixture
def mock_openrouter():
    """Set up respx to mock OpenRouter API."""
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        yield router

@pytest.fixture
def client():
    """Create a test client."""
    return OpenRouterClient(api_key="test-key-123")
```

### 7.2 Response Fixtures

Create reusable response fixtures for common scenarios:

```python
# tests/fixtures/responses.py

CHAT_RESPONSE_SUCCESS = {
    "id": "gen-test-001",
    "object": "chat.completion",
    "created": 1740576000,
    "model": "anthropic/claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
        "cost": 0.00015,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
        }
    }
}

CHAT_RESPONSE_TOOL_CALL = {
    "id": "gen-test-002",
    "object": "chat.completion",
    "model": "anthropic/claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "src/main.py"}'
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
        "cost": 0.00045,
    }
}

CHAT_RESPONSE_CACHED = {
    "id": "gen-test-003",
    "object": "chat.completion",
    "model": "anthropic/claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Based on the cached context..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10500,
        "completion_tokens": 200,
        "total_tokens": 10700,
        "cost": 0.0035,
        "prompt_tokens_details": {
            "cached_tokens": 10000,
            "cache_write_tokens": 0,
        }
    }
}

ERROR_RATE_LIMIT = {
    "error": {
        "message": "Rate limit exceeded. Please retry after 2 seconds.",
        "type": "rate_limit_error",
        "code": 429
    }
}

ERROR_CONTEXT_LENGTH = {
    "error": {
        "message": "This model's maximum context length is 200000 tokens. Your messages resulted in 250000 tokens.",
        "type": "invalid_request_error",
        "code": 400
    }
}

MODELS_RESPONSE = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4",
            "name": "Claude Sonnet 4",
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
            },
            "context_length": 200000,
            "top_provider": {
                "max_completion_tokens": 16384,
            },
            "supported_parameters": ["temperature", "tools"],
        },
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o Mini",
            "pricing": {
                "prompt": "0.00000015",
                "completion": "0.0000006",
                "input_cache_read": "0.000000075",
                "input_cache_write": "0",
            },
            "context_length": 128000,
            "top_provider": {
                "max_completion_tokens": 16384,
            },
            "supported_parameters": ["temperature", "tools"],
        }
    ]
}

GENERATION_RESPONSE = {
    "data": {
        "id": "gen-test-001",
        "model": "anthropic/claude-sonnet-4",
        "total_cost": 0.00015,
        "tokens_prompt": 10,
        "tokens_completion": 8,
        "native_tokens_prompt": 12,
        "native_tokens_completion": 8,
        "native_tokens_cached": 0,
        "generation_time": 1200,
        "latency": 1450,
        "provider_name": "Anthropic",
        "finish_reason": "stop",
        "streamed": False,
        "cancelled": False,
    }
}
```

### 7.3 Testing Happy Path

```python
import pytest

@pytest.mark.asyncio
async def test_chat_completion(mock_openrouter, client):
    mock_openrouter.post("/chat/completions").respond(
        200, json=CHAT_RESPONSE_SUCCESS
    )

    async with client:
        response = await client.chat(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
    assert response["usage"]["cost"] == 0.00015
    assert response["model"] == "anthropic/claude-sonnet-4"

@pytest.mark.asyncio
async def test_tool_calling(mock_openrouter, client):
    mock_openrouter.post("/chat/completions").respond(
        200, json=CHAT_RESPONSE_TOOL_CALL
    )

    async with client:
        response = await client.chat(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": "Read src/main.py"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }],
        )

    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "read_file"
    assert response["choices"][0]["finish_reason"] == "tool_calls"
```

### 7.4 Testing Error Paths

```python
@pytest.mark.asyncio
async def test_rate_limit_429(mock_openrouter, client):
    mock_openrouter.post("/chat/completions").respond(
        429,
        json=ERROR_RATE_LIMIT,
        headers={"Retry-After": "2"},
    )

    async with client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.chat(
                model="anthropic/claude-sonnet-4",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.response.status_code == 429

@pytest.mark.asyncio
async def test_server_error_500(mock_openrouter, client):
    mock_openrouter.post("/chat/completions").respond(500, json={
        "error": {"message": "Internal server error"}
    })

    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(
                model="anthropic/claude-sonnet-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

@pytest.mark.asyncio
async def test_timeout():
    """Test request timeout handling."""
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )

        client = OpenRouterClient(api_key="test-key")
        async with client:
            with pytest.raises(httpx.ReadTimeout):
                await client.chat(
                    model="anthropic/claude-sonnet-4",
                    messages=[{"role": "user", "content": "Hello"}],
                )

@pytest.mark.asyncio
async def test_context_length_exceeded(mock_openrouter, client):
    mock_openrouter.post("/chat/completions").respond(
        400, json=ERROR_CONTEXT_LENGTH
    )

    async with client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.chat(
                model="anthropic/claude-sonnet-4",
                messages=[{"role": "user", "content": "x" * 1000000}],
            )
        assert exc_info.value.response.status_code == 400
```

### 7.5 Testing Fallback Chains

```python
@pytest.mark.asyncio
async def test_fallback_on_rate_limit():
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        call_count = 0

        def route_handler(request):
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            model = body["model"]

            if model == "anthropic/claude-sonnet-4" and call_count <= 2:
                return httpx.Response(429, json=ERROR_RATE_LIMIT)
            else:
                response = CHAT_RESPONSE_SUCCESS.copy()
                response["model"] = model
                return httpx.Response(200, json=response)

        router.post("/chat/completions").mock(side_effect=route_handler)

        client = OpenRouterClient(api_key="test-key")
        async with client:
            result = await execute_with_fallback(
                client,
                models=["anthropic/claude-sonnet-4", "openai/gpt-4o"],
                messages=[{"role": "user", "content": "Hello"}],
                retry_config=RetryConfig(max_retries=1),
            )

        assert result.fallback_triggered
        assert result.model_used == "openai/gpt-4o"
        assert len(result.models_tried) == 2
```

### 7.6 Testing the Circuit Breaker

```python
def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request()

    for _ in range(3):
        cb.record_failure()

    assert cb.state == CircuitState.OPEN
    assert not cb.allow_request()

def test_circuit_breaker_half_open_after_recovery():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    import time
    time.sleep(0.15)

    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request()  # One test request allowed

def test_circuit_breaker_closes_on_success():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    import time
    time.sleep(0.15)

    cb.record_success()
    assert cb.state == CircuitState.CLOSED
```

### 7.7 Integration Testing (Gated by Environment)

```python
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

@pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set"
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_chat_completion():
    """Integration test against real OpenRouter API."""
    client = OpenRouterClient(api_key=OPENROUTER_API_KEY)
    async with client:
        response = await client.chat(
            model="openai/gpt-4o-mini",  # Use cheapest model
            messages=[{"role": "user", "content": "Say 'test passed' and nothing else."}],
            max_completion_tokens=10,
        )

    content = response["choices"][0]["message"]["content"]
    assert "test passed" in content.lower()
    assert response["usage"]["total_tokens"] > 0
    assert response["usage"]["cost"] > 0

@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_models_endpoint():
    """Verify models endpoint returns expected structure."""
    client = OpenRouterClient(api_key=OPENROUTER_API_KEY)
    async with client:
        models = await client.get_models()

    assert "data" in models
    assert len(models["data"]) > 100  # Should have 300+ models

    # Check structure of first model
    model = models["data"][0]
    assert "id" in model
    assert "pricing" in model
    assert "prompt" in model["pricing"]
    assert "completion" in model["pricing"]

@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_streaming():
    """Integration test for streaming responses."""
    client = OpenRouterClient(api_key=OPENROUTER_API_KEY)
    chunks_received = 0
    content_parts = []

    async with client:
        async for chunk in client.chat_stream(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_completion_tokens=50,
        ):
            chunks_received += 1
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if "content" in delta:
                    content_parts.append(delta["content"])

    assert chunks_received > 1
    full_content = "".join(content_parts)
    assert "1" in full_content
    assert "5" in full_content
```

### 7.8 Snapshot Testing for Response Parsing

Use pytest-snapshot or inline snapshots to catch response parsing regressions:

```python
def test_usage_extraction():
    """Ensure usage extraction handles all field combinations."""
    record = extract_usage(CHAT_RESPONSE_SUCCESS)
    assert record.request_id == "gen-test-001"
    assert record.model == "anthropic/claude-sonnet-4"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 8
    assert record.cost_usd == 0.00015
    assert record.cached_tokens == 0

def test_usage_extraction_with_cache():
    """Ensure cached tokens are extracted correctly."""
    record = extract_usage(CHAT_RESPONSE_CACHED)
    assert record.cached_tokens == 10000
    assert record.prompt_tokens == 10500

def test_usage_extraction_missing_fields():
    """Handle responses with missing optional fields gracefully."""
    minimal_response = {
        "id": "gen-minimal",
        "model": "test/model",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
    }
    record = extract_usage(minimal_response)
    assert record.cost_usd == 0.0
    assert record.cached_tokens == 0
    assert record.reasoning_tokens == 0
```

---

## 8. Real-World OpenRouter Usage Patterns

### 8.1 Open-Source Projects Using OpenRouter

**Official OpenRouter Python SDK** (`openrouter` on PyPI):
- Uses httpx under the hood for both sync and async
- Context manager protocol for resource cleanup
- Type-safe with full typing
- GitHub: https://github.com/OpenRouterTeam/python-sdk

**aider (AI pair programming):**
- Uses OpenRouter for multi-model access
- Configures via `--openrouter-api-key` flag
- Supports `:nitro` and `:floor` suffixes
- Docs: https://aider.chat/docs/llms/openrouter.html

**Pydantic AI:**
- Has a dedicated `pydantic_ai.models.openrouter` module
- Wraps OpenRouter with structured output support
- Docs: https://ai.pydantic.dev/api/models/openrouter/

**LiteLLM:**
- Open-source proxy that supports OpenRouter as a provider
- Handles routing, fallbacks, cost tracking at the proxy level
- Can sit between your app and OpenRouter for additional control

### 8.2 Common Pitfalls

**1. Rate limiting surprise on free models:**
Free models (model IDs ending in `:free`) have strict limits: ~20 requests/minute, ~200/day. Multiple API keys or accounts do not circumvent global limits.

**2. Cost surprises with reasoning models:**
O-series models (o3, o4-mini) bill reasoning tokens as output tokens. A 500-token visible response can consume 2000+ total tokens. The `completion_tokens_details.reasoning_tokens` field reveals the true count.

**3. Model availability changes:**
Models are added and removed frequently. Always handle 404 (model not found) gracefully. Cache the models list and refresh periodically (every 5-10 minutes).

**4. Negative credit balance blocks everything:**
If your balance goes negative, all requests fail with 402 -- including free models. Monitor credits proactively.

**5. Streaming usage data may be absent:**
The `usage` field in streaming responses is only guaranteed in the final chunk. For precise accounting, query the `/generation` endpoint after streaming completes.

**6. Provider routing affects caching:**
If you let OpenRouter load-balance across providers, cache hits are less likely. Use sticky routing or set `provider.order` to pin to a specific provider when caching matters.

### 8.3 OpenRouter vs. Direct API Access

| Factor | OpenRouter | Direct API |
|--------|-----------|------------|
| **API keys** | One key for all providers | One key per provider |
| **Pricing** | Passthrough (no markup on tokens) | Direct pricing |
| **Platform fee** | 5.5% on credit purchases | None |
| **Fallbacks** | Built-in cross-provider | Must implement yourself |
| **Model discovery** | Single `/models` endpoint | Provider-specific |
| **Prompt caching** | Supported, with sticky routing | Provider-native |
| **Latency overhead** | ~25-40ms per request | None |
| **Rate limits** | OpenRouter limits + provider limits | Provider limits only |
| **Observability** | Activity page, `/generation` endpoint | Provider dashboards |
| **ZDR/compliance** | Configurable per request | Provider-specific |

**For aiai, OpenRouter is the right choice because:**
- Single integration point for all models
- Built-in provider fallback reduces our engineering burden
- Model discovery API enables dynamic pricing comparison
- The 5.5% credit purchase fee is worth the operational simplicity
- The ~30ms latency overhead is negligible for our workloads

### 8.4 Provider Routing Strategies in Practice

**Strategy 1: Price-first with quality floor**
```python
provider = {
    "sort": "price",
    "preferred_min_throughput": {"p90": 40},
    "allow_fallbacks": True,
}
```

**Strategy 2: Throughput-first for interactive use**
```python
provider = {
    "sort": "throughput",
    "preferred_max_latency": {"p90": 3.0},
}
```

**Strategy 3: Compliance-first**
```python
provider = {
    "data_collection": "deny",
    "zdr": True,
    "only": ["Anthropic", "Azure"],
}
```

**Strategy 4: Quantization filtering**
```python
# Only use high-precision quantizations
provider = {
    "quantizations": ["fp16", "bf16", "fp32"],
    "sort": "price",
}
```

---

## 9. Pricing Data and Cost Models

### 9.1 Current Model Pricing (February 2026)

All prices per million tokens (input / output). Sourced from OpenRouter `/models` endpoint.

#### Frontier Models

| Model | Input $/M | Output $/M | Context | Cache Read $/M | Cache Write $/M |
|-------|-----------|------------|---------|---------------|-----------------|
| Claude Opus 4.5 | $5.00 | $25.00 | 200K | $0.50 | $6.25 |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 200K | $0.30 | $3.75 |
| Claude Sonnet 4 | $3.00 | $15.00 | 200K | $0.30 | $3.75 |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | $0.10 | $1.25 |
| GPT-4o | $2.50 | $10.00 | 128K | $1.25 | -- |
| GPT-4.1 | $2.00 | $8.00 | 1M | $1.00 | -- |
| GPT-4o-mini | $0.15 | $0.60 | 128K | $0.075 | -- |
| o3-mini | $1.10 | $4.40 | 200K | $0.55 | -- |
| o4-mini | $1.10 | $4.40 | 200K | $0.55 | -- |
| Gemini 2.5 Pro | $1.25 | $10.00 | 1M | varies | -- |
| Gemini 2.5 Flash | $0.15 | $0.60 | 1M | varies | -- |
| Gemini 2.0 Flash | $0.10 | $0.40 | 1M | varies | -- |

#### Cost-Optimized Models

| Model | Input $/M | Output $/M | Context | Notes |
|-------|-----------|------------|---------|-------|
| DeepSeek V3 | $0.14 | $0.28 | 64K | Cache hit: $0.028/M |
| DeepSeek R1 | $0.55 | $1.68 | 64K | Reasoning model |
| Llama 3.3 70B | $0.10-0.40 | $0.10-0.40 | 128K | Varies by provider |
| Qwen 2.5 72B | $0.10-0.35 | $0.10-0.35 | 128K | Varies by provider |
| Mistral Large | $2.00 | $6.00 | 128K | |

#### Free Models (Rate Limited)

| Model | Limit |
|-------|-------|
| Various `:free` variants | ~20 req/min, ~200 req/day |

### 9.2 Pricing Modifiers

| Modifier | Anthropic | OpenAI | Google |
|----------|-----------|--------|--------|
| Cache write (5m) | 1.25x input | Free (automatic) | Varies |
| Cache write (1h) | 2.0x input | N/A | Varies |
| Cache read | 0.1x input (90% off) | 0.25-0.5x input (50-75% off) | Varies |
| Batch processing | 0.5x (50% off) | 0.5x (50% off) | Available |
| Long context (>200K) | 2x input, 1.5x output | N/A | 2x input, 1.5x output |
| Reasoning tokens | Billed as output | Billed as output | Billed as output |

### 9.3 Building a Pricing Table

Fetch pricing dynamically from the `/models` endpoint and cache it:

```python
import time
from dataclasses import dataclass

@dataclass
class ModelPricing:
    model_id: str
    prompt_per_token: float      # USD per token
    completion_per_token: float
    cache_read_per_token: float
    cache_write_per_token: float
    context_length: int
    max_completion_tokens: int
    prompt_per_million: float    # USD per million tokens (convenience)
    completion_per_million: float

class PricingTable:
    def __init__(self):
        self._models: dict[str, ModelPricing] = {}
        self._last_refresh: float = 0
        self._refresh_interval: float = 300  # 5 minutes

    async def refresh(self, client: OpenRouterClient) -> None:
        """Fetch latest pricing from OpenRouter."""
        response = await client.get_models()
        self._models.clear()

        for model_data in response["data"]:
            model_id = model_data["id"]
            pricing = model_data.get("pricing", {})

            prompt_per_token = float(pricing.get("prompt", "0"))
            completion_per_token = float(pricing.get("completion", "0"))
            cache_read = float(pricing.get("input_cache_read", "0"))
            cache_write = float(pricing.get("input_cache_write", "0"))

            top_provider = model_data.get("top_provider", {})
            context_length = model_data.get("context_length") or 0
            max_completion = top_provider.get("max_completion_tokens") or 4096

            self._models[model_id] = ModelPricing(
                model_id=model_id,
                prompt_per_token=prompt_per_token,
                completion_per_token=completion_per_token,
                cache_read_per_token=cache_read,
                cache_write_per_token=cache_write,
                context_length=context_length,
                max_completion_tokens=max_completion,
                prompt_per_million=prompt_per_token * 1_000_000,
                completion_per_million=completion_per_token * 1_000_000,
            )

        self._last_refresh = time.monotonic()

    async def ensure_fresh(self, client: OpenRouterClient) -> None:
        """Refresh if stale."""
        if time.monotonic() - self._last_refresh > self._refresh_interval:
            await self.refresh(client)

    def get(self, model_id: str) -> ModelPricing | None:
        return self._models.get(model_id)

    def estimate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost for a request."""
        pricing = self.get(model_id)
        if not pricing:
            return None

        uncached = prompt_tokens - cached_tokens
        cost = (
            uncached * pricing.prompt_per_token
            + cached_tokens * pricing.cache_read_per_token
            + completion_tokens * pricing.completion_per_token
        )
        return cost

    def rank_by_cost(
        self,
        model_ids: list[str],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> list[tuple[str, float]]:
        """Rank models by estimated cost, cheapest first."""
        costs = []
        for mid in model_ids:
            cost = self.estimate_cost(mid, prompt_tokens, completion_tokens)
            if cost is not None:
                costs.append((mid, cost))
        costs.sort(key=lambda x: x[1])
        return costs
```

### 9.4 Handling Pricing Changes

Model pricing on OpenRouter changes when providers update their rates. The `/models` endpoint always returns current pricing.

**Strategy:**
1. Cache pricing in memory with a 5-minute TTL
2. Refresh on cache miss (new model not in table)
3. Log pricing changes when detected during refresh
4. Never hardcode prices -- always fetch from the API

```python
async def detect_pricing_changes(
    old_table: dict[str, ModelPricing],
    new_table: dict[str, ModelPricing],
) -> list[dict]:
    """Detect and log pricing changes between refreshes."""
    changes = []
    for model_id, new_pricing in new_table.items():
        old_pricing = old_table.get(model_id)
        if old_pricing:
            if old_pricing.prompt_per_token != new_pricing.prompt_per_token:
                changes.append({
                    "model": model_id,
                    "field": "prompt",
                    "old": old_pricing.prompt_per_million,
                    "new": new_pricing.prompt_per_million,
                    "change_pct": (
                        (new_pricing.prompt_per_million - old_pricing.prompt_per_million)
                        / old_pricing.prompt_per_million * 100
                        if old_pricing.prompt_per_million > 0 else 0
                    ),
                })
    return changes
```

### 9.5 Cost Comparison: OpenRouter vs Direct API

OpenRouter passes through model pricing at no per-token markup. The cost difference comes from:

1. **Credit purchase fee:** 5.5% on credit card (5% crypto). This is the effective "markup."
2. **No volume discounts:** Direct API agreements with providers may include volume discounts for large customers.
3. **No commitment pricing:** Some providers offer reserved capacity at lower rates.

**Effective cost comparison for $1000/month spend:**

| | Direct API | OpenRouter |
|---|-----------|-----------|
| Token costs | $1000.00 | $1000.00 |
| Platform fee | $0.00 | $55.00 (5.5%) |
| Integration cost (eng time) | Higher (multi-provider) | Lower (single integration) |
| Fallback engineering | You build it | Built-in |
| Total | $1000 + eng time | $1055 |

For aiai at current scale, the 5.5% premium is well worth the simplified integration.

### 9.6 Per-Request Cost Estimation Example

```python
# Example: Estimating cost for aiai's typical workloads

pricing = PricingTable()

# Trivial task (formatting, renaming)
# ~500 input tokens, ~100 output tokens, using Haiku
trivial_cost = pricing.estimate_cost(
    "anthropic/claude-haiku-4-5",
    prompt_tokens=500,
    completion_tokens=100,
)
# ~= 500 * $0.000001 + 100 * $0.000005 = $0.001

# Simple task (single-file edit)
# ~2000 input tokens, ~500 output tokens, using Sonnet
simple_cost = pricing.estimate_cost(
    "anthropic/claude-sonnet-4",
    prompt_tokens=2000,
    completion_tokens=500,
)
# ~= 2000 * $0.000003 + 500 * $0.000015 = $0.0135

# Complex task (architecture decision)
# ~20000 input tokens (with context), ~5000 output tokens, using Opus
complex_cost = pricing.estimate_cost(
    "anthropic/claude-opus-4",
    prompt_tokens=20000,
    completion_tokens=5000,
)
# Much higher -- this is where routing saves money

# Same complex task with 90% cache hit (system prompt cached)
complex_cached = pricing.estimate_cost(
    "anthropic/claude-opus-4",
    prompt_tokens=20000,
    completion_tokens=5000,
    cached_tokens=15000,  # 75% of input cached
)
# Significant savings from caching
```

---

## Appendix A: Complete Client Architecture

Putting it all together -- the full client architecture for aiai:

```
src/router/
  __init__.py
  client.py           # OpenRouterClient (httpx async, streaming, context manager)
  router.py           # Complexity-based model selection, cascading
  fallback.py          # Retry logic, fallback chains, circuit breakers
  cost.py              # Cost tracking, budget enforcement, JSONL logging
  pricing.py           # PricingTable, cost estimation
  cache.py             # Prompt caching utilities, message structuring
  config.py            # Load config/models.yaml, validate
  types.py             # Dataclasses, enums (Complexity, FailureType, etc.)

tests/
  test_client.py       # Client tests with respx mocks
  test_router.py       # Routing logic tests
  test_fallback.py     # Retry, fallback, circuit breaker tests
  test_cost.py         # Cost tracking, budget enforcement tests
  test_pricing.py      # Pricing table, estimation tests
  test_cache.py        # Prompt caching tests
  test_integration.py  # Real API tests (gated by OPENROUTER_API_KEY)
  fixtures/
    responses.py       # Canned API responses
```

## Appendix B: Configuration Reference

The full `config/models.yaml` schema:

```yaml
provider: openrouter

tiers:
  <tier_name>:
    models:
      - <model_id>      # Ordered fallback list
    max_tokens: <int>
    temperature: <float>

routing:
  trivial: <tier_name>
  simple: <tier_name>
  medium: <tier_name>
  complex: <tier_name>
  critical: <tier_name>

cost:
  log_file: <path>              # JSONL cost log path
  warn_threshold_usd: <float>   # Per-request warning threshold
  daily_budget_usd: <float>     # Daily hard limit

provider_defaults:
  sort: "price"                 # Default provider sort
  allow_fallbacks: true
  require_parameters: false
  data_collection: "allow"      # or "deny"

retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 60.0
  jitter_factor: 0.5

circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60.0
  half_open_max_calls: 1
```

## Appendix C: Key Dependencies

```
# pyproject.toml or requirements.txt
httpx>=0.27.0          # Async HTTP client
aiofiles>=24.0         # Async file I/O for cost logging
pyyaml>=6.0            # Config file parsing
tenacity>=9.0          # Advanced retry logic (optional, for complex retry needs)

# Testing
pytest>=8.0
pytest-asyncio>=0.24
respx>=0.22            # httpx mocking
```

## References

- [OpenRouter API Reference](https://openrouter.ai/docs/api/reference/overview)
- [OpenRouter Chat Completions](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [OpenRouter Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [OpenRouter Model Fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks)
- [OpenRouter Prompt Caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
- [OpenRouter Usage Accounting](https://openrouter.ai/docs/guides/guides/usage-accounting)
- [OpenRouter Python SDK](https://github.com/OpenRouterTeam/python-sdk)
- [OpenRouter Rate Limits](https://openrouter.ai/docs/api/reference/limits)
- [OpenRouter Credits API](https://openrouter.ai/docs/api/api-reference/credits/get-credits)
- [OpenRouter Generation Stats](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [OpenRouter OpenAPI Spec](https://openrouter.ai/openapi.yaml)
- [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM)
- [RouteLLM Paper (ICLR 2025)](https://arxiv.org/abs/2406.18665)
- [Unified Routing and Cascading Paper](https://arxiv.org/abs/2410.10347)
- [httpx Documentation](https://www.python-httpx.org/)
- [httpx Async Support](https://www.python-httpx.org/async/)
- [httpx Timeouts](https://www.python-httpx.org/advanced/timeouts/)
- [httpx Resource Limits](https://www.python-httpx.org/advanced/resource-limits/)
- [respx (httpx mocking)](https://lundberg.github.io/respx/)
- [PyBreaker (Circuit Breaker)](https://pypi.org/project/pybreaker/)
- [Tenacity (Retry Library)](https://tenacity.readthedocs.io/)
