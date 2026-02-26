# LLM Cost Optimization, Model Routing, and Multi-Model Architectures

**Research compiled: 2026-02-26**

---

## Table of Contents

1. [Model Routing Systems](#1-model-routing-systems)
2. [Cost Optimization Strategies](#2-cost-optimization-strategies)
3. [OpenRouter Specifics](#3-openrouter-specifics)
4. [Token Economics](#4-token-economics)
5. [Multi-Model Architectures](#5-multi-model-architectures)
6. [Benchmarking Model Quality vs Cost](#6-benchmarking-model-quality-vs-cost)
7. [Rate Limiting and Reliability](#7-rate-limiting-and-reliability)
8. [Key Takeaways and Recommendations](#8-key-takeaways-and-recommendations)
9. [References](#9-references)

---

## 1. Model Routing Systems

### 1.1 What Is Model Routing?

Model routing is the practice of dynamically selecting which LLM should handle a given request based on factors like query complexity, cost constraints, latency requirements, and model capabilities. Rather than sending all requests to a single expensive model, a router acts as a "meta-model" that directs traffic to the most appropriate model for each individual query.

The landscape evolved from scattered academic research in mid-2023, to over a dozen papers by 2024, and by 2025 both academic routers and commercial products (most notably GPT-5 with a built-in router) had emerged.

### 1.2 Commercial Routing Systems

#### OpenRouter Auto Router
- Powered by Not Diamond under the hood
- Analyzes prompt complexity, task type, and model capabilities
- No additional fee for using the auto router -- you pay the standard rate of whichever model is selected
- Supports `:nitro` (sort by throughput) and `:floor` (sort by lowest price) suffixes
- Docs: https://openrouter.ai/docs/guides/routing/routers/auto-router

#### Martian Model Router
- Dynamically routes requests to the "best" LLM in real-time
- Core technology focuses on predicting model behavior for a given query
- Used in enterprise contexts; partnered with Accenture for enterprise AI deployments
- Hosted an interpretability hackathon with Apart Research in 2025
- GitHub: https://github.com/withmartian
- Coverage: https://venturebeat.com/ai/why-accenture-and-martian-see-model-routing-as-key-to-enterprise-ai-success

#### Not Diamond
- AI model router that determines which LLM is best-suited per query
- Three optimization modes: Quality (default), Cost, and Latency
- Claims up to 25% accuracy improvement and 10-100x inference cost savings
- Stack-agnostic -- works as an optimization layer, not a gateway
- Strategic investors include IBM and SAP; powers SAP's prompt optimization service
- SDK: https://github.com/Not-Diamond/notdiamond-python
- Docs: https://docs.notdiamond.ai/docs/what-is-model-routing

#### Unify.ai
- Routes to the "perfect model and provider" for each individual prompt
- Optimizes across quality, cost, and speed dimensions
- RouteLLM benchmarks showed Unify AI was outperformed by the open-source RouteLLM framework while being more expensive

### 1.3 Open-Source Routing Frameworks

#### RouteLLM (LMSYS / UC Berkeley)
- **Paper**: "RouteLLM: Learning to Route LLMs with Preference Data" (arXiv:2406.18665)
- Routes between a strong/expensive model and a weak/cheap model
- Trained on public Chatbot Arena preference data
- Four router types: SW Ranking, Matrix Factorization, BERT classifier, Causal LLM classifier
- **Results**: 85% cost reduction on MT Bench, 45% on MMLU, 35% on GSM8K vs. using GPT-4 only
- Key property: **transfer learning** -- routers trained on GPT-4/Mixtral pair generalize to other model pairs without retraining
- GitHub: https://github.com/lm-sys/RouteLLM
- Blog: https://lmsys.org/blog/2024-07-01-routellm/

#### RouterArena (2025)
- **Paper**: arXiv:2510.00202 -- first open platform for comprehensive LLM router evaluation
- Dataset: 8,400 queries across 9 domains and 44 categories at 3 difficulty levels
- Metrics: accuracy, cost, routing optimality, robustness to perturbations, overhead latency
- **Key finding**: Commercial routers do not necessarily outperform open-source routers. GPT-5's built-in router ranks #7 (restricted model pool), Not Diamond ranks #12 (tends to select expensive models)
- No single router ranks top across all metrics -- inherent trade-offs exist
- Paper: https://arxiv.org/abs/2510.00202

### 1.4 Academic Papers on Model Routing

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| RouteLLM (LMSYS) | 2024 | Preference-data-based routing, 85% cost reduction on MT-Bench |
| Router-R1 (Zhang et al.) | 2025 | Multi-round routing + aggregation via reinforcement learning (arXiv:2506.09033) |
| GraphRouter (Feng et al.) | 2025 | Graph-based LLM selection, accepted at ICLR 2025 |
| IRT-Router | 2025 | Item Response Theory for interpretable multi-LLM routing (ACL 2025) |
| Unified Routing & Cascading | 2024 | Proves optimality of cascade routing strategy, unifies routing + cascading (arXiv:2410.10347) |
| RouterArena | 2025 | Open evaluation platform for LLM routers (arXiv:2510.00202) |

### 1.5 How Routing Decisions Are Made

Routing systems generally use one or more of the following signals:

1. **Query complexity estimation** -- Classify whether the query is simple (use cheap model) or complex (use expensive model). RouteLLM uses preference-data-trained classifiers for this.
2. **Task type detection** -- Coding, math, creative writing, summarization, etc. each have different model strengths.
3. **Cost/quality tradeoff threshold** -- User-configurable: "I want 90% of GPT-4 quality at minimum cost" translates to a routing threshold.
4. **Historical model performance** -- GraphRouter and IRT-Router use performance prediction models.
5. **Embedding similarity** -- Match query embeddings to clusters of known good model-query pairs.

---

## 2. Cost Optimization Strategies

### 2.1 Overview of Techniques

Research shows that strategic LLM cost optimization can cut inference expenses by up to 98% while maintaining or even improving accuracy. The techniques below are ordered roughly from easiest to hardest to implement.

### 2.2 Prompt Caching

#### Anthropic Prompt Caching
- **How it works**: Stores KV cache representations of prompt prefixes. On subsequent requests, if the prefix matches, computation resumes from the cached point rather than reprocessing from scratch.
- **Pricing**: Cache writes cost 1.25x base input price; cache reads cost 0.1x base input price (90% discount)
- **Cache TTL**: Minimum 5 minutes of inactivity; 1-hour TTL also available
- **Implementation**: Two modes -- automatic (single `cache_control` field) or explicit (up to 4 breakpoints)
- **Real-world impact**: One developer reported going from $720/month to $72/month (90% reduction)
- **Cache order**: tools -> system -> messages (forms a hierarchy)
- Docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

#### OpenAI Automatic Caching
- Enabled by default for all API requests
- 50% cost savings on cached input tokens
- No explicit configuration needed

#### DeepSeek Cache Mechanism
- Cache hit: $0.028/M tokens (vs. $0.28/M standard) -- 90% discount
- Automatic for repeated prompt prefixes

#### Key Insight
For agentic workflows that repeatedly send large system prompts + tool definitions, prompt caching is the single highest-impact optimization. A coding agent sending 10K tokens of system prompt on every turn saves 90% on those tokens with Anthropic caching.

### 2.3 Semantic Caching

Unlike prompt caching (exact prefix match), semantic caching retrieves stored LLM responses based on **semantic similarity** between prompts using embeddings.

- **How it works**: Incoming query is embedded, compared against cache via cosine similarity. If above threshold, return cached response without calling the LLM.
- **Tools**: GPTCache (Zilliz), ScyllaDB semantic caching, AWS solutions, custom vector DB implementations
- **Best for**: Customer support, FAQ-type queries, repeated intents
- **Not suitable for**: Highly contextual or personalized queries, creative tasks
- **Trade-off**: Cache hit rate vs. staleness risk. Typical similarity thresholds: 0.92-0.97
- GitHub (GPTCache): https://github.com/zilliztech/GPTCache

### 2.4 Batch Processing

Both OpenAI and Anthropic offer batch APIs with significant discounts:

| Provider | Batch Discount | Turnaround Time | API |
|----------|---------------|-----------------|-----|
| OpenAI | 50% off standard pricing | Within 24 hours | Batch API |
| Anthropic | 50% off standard pricing | Async processing | Message Batches API |

- **Best for**: Non-real-time workloads -- evaluation, data labeling, bulk classification, report generation
- **Real-world example**: 10M tokens/month at Anthropic batch pricing saves ~$25,000/year
- Blog: https://engineering.miko.ai/save-50-on-openai-api-costs-using-batch-requests-6ad41214b4ac

### 2.5 Model Cascading

Start with a cheap model; escalate to an expensive model only if the cheap model fails or confidence is low.

**Pattern:**
```
Query -> Small Model (e.g., GPT-4o-mini, $0.15/M input)
  |-> Confidence high? Return response
  |-> Confidence low? -> Large Model (e.g., Claude Opus, $5/M input)
```

**Results from research:**
- 90% of queries handled by small model -> 87% cost reduction
- Cascaded LLM orchestration in coding agents: cost drops from $0.931 to $0.054 per task (94% reduction) while maintaining success rates
- The unified routing + cascading paper (arXiv:2410.10347) proves theoretical optimality of cascade routing

### 2.6 Speculative Decoding

A latency optimization (rather than direct cost optimization) that reduces inference time 2-3x without changing output quality.

**How it works:**
1. A small, fast **draft model** generates K speculative tokens quickly
2. The large **target model** verifies all K tokens in a single forward pass
3. Accepted tokens are kept; rejected tokens trigger regeneration from the divergence point
4. Output distribution is mathematically identical to the target model alone

**Production status (2025):**
- vLLM and TensorRT-LLM include native speculative decoding support
- NVIDIA demonstrated 3.6x throughput improvements on H200 GPUs
- Recent advances solved vocabulary compatibility issues between draft and target models
- Blog: https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/

**Cost implication**: Reduces cost per request when self-hosting (same quality, fewer GPU-seconds). Does not directly reduce API costs but improves throughput.

### 2.7 Model Distillation

Transfer knowledge from a large "teacher" model to a small "student" model.

**Key metrics:**
- 2-8x faster inference than teacher models
- Maintains up to 95% of teacher's accuracy
- Llama 3.1 405B -> distilled 8B: 21% better accuracy on NLI tasks than the base 8B model
- Optimal compression order: Pruning -> Distillation -> Quantization (P-KD-Q)

**Tools:**
- Alibaba's EasyDistill (2025) for compressing large models
- NVIDIA TensorRT Model Optimizer for pruning + distillation
- OpenAI's built-in distillation API (fine-tuning with GPT-4 outputs)

**Use case for cost optimization**: Distill a frontier model into a task-specific small model for your most common queries. Handle edge cases with the full frontier model.

### 2.8 Prompt Optimization

- **Prompt compression**: Remove redundant tokens while preserving meaning. Tools: LLMLingua, selective context
- **Structured output**: Request JSON/structured responses to reduce verbose output tokens
- **System prompt optimization**: Minimize system prompt length; move static instructions to cached prefixes
- **Impact**: 15-40% cost reduction with minimal effort

### 2.9 Combined Strategy Impact

| Strategy | Effort | Typical Savings | Latency Impact |
|----------|--------|----------------|----------------|
| Prompt caching | Low | 60-90% on repeated prefixes | Improves |
| Semantic caching | Medium | 20-50% (depends on hit rate) | Improves |
| Batch processing | Low | 50% flat discount | Adds delay (24h) |
| Model cascading | Medium | 70-94% | Slight increase |
| Model routing | Medium | 35-85% | Neutral |
| Speculative decoding | High (self-host) | 2-3x throughput | Improves |
| Distillation | High | 2-8x cheaper inference | Improves |
| Prompt optimization | Low | 15-40% | Improves |

These techniques are **highly complementary** and can deliver compound savings of 60-90%.

---

## 3. OpenRouter Specifics

### 3.1 Architecture

OpenRouter is an LLM API gateway and marketplace. The internal architecture consists of:

1. **Unified endpoint**: `api.openrouter.ai` -- accepts OpenAI-compatible payloads (`/chat/completions`)
2. **Reverse proxy + routing layer**: Translates requests, selects providers, handles failover
3. **Provider pool**: 290+ models from all major providers (OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, etc.)
4. **Latency overhead**: ~25ms in ideal conditions, ~40ms under typical production load

### 3.2 How Routing Works

OpenRouter provides several routing mechanisms:

**Provider Selection:**
- Default: Load-balances based on price, considering uptime
- Configurable via `provider.sort`: "price", "throughput", "latency"
- If `sort` or `order` is set, load balancing is disabled; providers are tried in order

**Model Fallbacks:**
- Specify multiple models in the `models` parameter
- Primary model's providers are tried first (grouped by model)
- Fallbacks triggered by: rate limits, context length errors, moderation flags, downtime
- Set `partition: "none"` to sort globally across all models (ignoring model grouping)
- Billed only for the successful model run

**Routing Shortcuts:**
- `:nitro` suffix -- prioritize providers with highest throughput (tokens/second)
- `:floor` suffix -- always select the lowest-priced provider
- These are equivalent to setting `provider.sort` to "throughput" or "price"

**Auto Router (`openrouter/auto`):**
- Powered by Not Diamond
- Analyzes prompt and selects optimal model from curated set
- No additional fee beyond the selected model's pricing
- Can restrict model pool via `plugins` parameter

**Message Transforms:**
- `middle-out` transform: Truncates messages from the middle when prompt exceeds context window
- Useful for long conversations that might overflow smaller models

### 3.3 Pricing Model

**Platform fees (as of mid-2025):**
- Credit card purchases: 5.5% fee (minimum $0.80)
- Crypto purchases: 5% fee, no minimum

**Model pricing:**
- At or very close to direct API pricing for most models
- Some models available for free (rate-limited: ~20 req/min, 200/day)
- You pay per-token based on the model actually used

**Prompt caching on OpenRouter:**
- Supported for providers that offer it (Anthropic, OpenAI, Google)
- Cache discount passes through to the user
- Docs: https://openrouter.ai/docs/guides/best-practices/prompt-caching

### 3.4 Best Practices for Production Use

1. **Start with explicit model selection** in staging to baseline behavior
2. **Introduce auto routing** (`:nitro` or `:floor`) behind feature flags
3. **Configure fallback chains**: preferred model -> backup provider -> open-source fallback
4. **Simulate provider unavailability** during off-peak hours to test failover
5. **Monitor cost/latency shifts** before promoting routing changes to production
6. **Use prompt caching** for repeated system prompts / tool definitions
7. **Set provider preferences** to control which providers handle your traffic

### 3.5 API Patterns

```python
# Basic OpenRouter usage (OpenAI SDK compatible)
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-..."
)

# Explicit model selection
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Using :nitro for fastest throughput
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4:nitro",
    messages=[{"role": "user", "content": "Hello"}]
)

# Using :floor for cheapest provider
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4:floor",
    messages=[{"role": "user", "content": "Hello"}]
)

# Model fallbacks with provider preferences
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    extra_body={
        "models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-2.5-pro"
        ],
        "provider": {
            "sort": "price",
            "allow_fallbacks": True
        }
    },
    messages=[{"role": "user", "content": "Hello"}]
)

# Auto router
response = client.chat.completions.create(
    model="openrouter/auto",
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## 4. Token Economics

### 4.1 Current Model Pricing (February 2026)

Prices are per million tokens (input / output).

#### Frontier Models

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| Claude Opus 4.5 | $5.00 | $25.00 | Most capable Anthropic model |
| Claude Sonnet 4.5 | $3.00 | $15.00 | $6/$22.50 for >200K context |
| Claude Haiku 4.5 | $1.00 | $5.00 | Fast, cost-efficient |
| GPT-4o | $2.50 | $10.00 | OpenAI flagship |
| GPT-4.1 | $2.00 | $8.00 | Newer, slightly cheaper |
| GPT-4o-mini | $0.15 | $0.60 | Ultra-cheap, surprisingly capable |
| o3-mini | $1.10 | $4.40 | Reasoning model; hidden reasoning tokens billed as output |
| o4-mini | $1.10 | $4.40 | Faster reasoning model |
| Gemini 2.5 Pro | $1.25 | $10.00 | $2.50/$15 for >200K context |
| Gemini 2.0 Flash | $0.10 | $0.40 | Extremely cheap |
| DeepSeek V3 | $0.14 | $0.28 | Cache hit: $0.028/M input |
| DeepSeek R1 | $0.55 | $1.68 | Reasoning model |

#### Pricing Modifiers

| Modifier | Anthropic | OpenAI | Google |
|----------|-----------|--------|--------|
| Prompt cache write | 1.25x input | N/A (automatic) | Varies |
| Prompt cache read | 0.1x input (90% off) | 0.5x input (50% off) | Varies |
| Batch processing | 0.5x (50% off) | 0.5x (50% off) | Available |
| Long context (>200K) | 2x input, 1.5x output | N/A | 2x input, 1.5x output |

#### Important: Reasoning Token Costs

O-series models (o3, o4-mini) use hidden "reasoning tokens" billed as output tokens. A 500-token visible response may consume 2,000+ total tokens. This makes the effective cost much higher than the per-token price suggests for complex reasoning tasks.

### 4.2 Cost Per Task: Real-World Estimates

Based on published data and research:

#### Simple Tasks (classification, extraction, short Q&A)
- ~500 input tokens, ~100 output tokens
- GPT-4o-mini: $0.000135/task ($0.135 per 1,000 tasks)
- Gemini 2.0 Flash: $0.00009/task ($0.09 per 1,000 tasks)
- DeepSeek V3: $0.0001/task ($0.10 per 1,000 tasks)

#### Medium Tasks (summarization, code review, analysis)
- ~2,000 input tokens, ~500 output tokens
- Claude Sonnet 4.5: $0.0135/task ($13.50 per 1,000 tasks)
- GPT-4o: $0.01/task ($10 per 1,000 tasks)
- Gemini 2.5 Pro: $0.0075/task ($7.50 per 1,000 tasks)

#### Complex Coding Agent Tasks
- Research from OpenHands SWE-bench analysis shows agentic coding tasks consume highly variable token counts
- Input tokens dominate: some runs use 10x more tokens than others
- Average agentic run: ~50K-200K input tokens, ~5K-20K output tokens per task
- Claude Sonnet 4.5 at ~150K input + 10K output: ~$0.60/task
- With cascading (94% handled by cheap model): ~$0.054/task

#### Support Ticket Processing (10,000 tickets/day)
- GPT-5.2 Pro: ~$1,300+/day
- GPT-4o-mini: ~$7/day (190x cheaper)
- This illustrates the massive impact of model selection

### 4.3 Cost Trends

- **Epoch AI research**: The cost to achieve GPT-4 performance has fallen ~40x per year
- **Anthropic**: Claude Opus 4.5 at $5/$25 represents a 67% price reduction vs. Opus 4.1 at $15/$75
- **DeepSeek**: Cut all prices ~50% in September 2025; V3.2-Exp at $0.028/M input
- **Industry direction**: By 2026, cost is becoming a chief competitive factor, potentially surpassing raw performance in importance

---

## 5. Multi-Model Architectures

### 5.1 Mixture of Agents (MoA)

**Paper**: "Mixture-of-Agents Enhances Large Language Model Capabilities" (arXiv:2406.04692, ICLR 2025 Spotlight)

**How it works:**
- Layered architecture where each layer has multiple LLM agents
- Each agent receives outputs from ALL agents in the previous layer as auxiliary context
- Final layer has a single "aggregator" model that synthesizes the best response

**Configuration (Together AI's implementation):**
- Proposer models: Qwen1.5-110B, Qwen1.5-72B, WizardLM-8x22B, LLaMA-3-70B, Mixtral-8x22B, DBRX
- 3 MoA layers, same models in each layer
- Aggregator: Qwen1.5-110B-Chat

**Results:**
- 65.1% on AlpacaEval 2.0 (vs. GPT-4o at 57.5%) using only open-source models
- State-of-the-art on AlpacaEval 2.0, MT-Bench, and FLASK
- Consistent, monotonic performance gain after each layer

**MoA-Lite (cost-effective variant):**
- 2 layers instead of 3
- Smaller aggregator (Qwen1.5-72B)
- Still beats GPT-4o by 1.8% on AlpacaEval while being more cost-effective

**Selection criteria:**
- Performance metrics (average win rate)
- Diversity (heterogeneous models contribute more than same-model duplicates)

GitHub: https://github.com/togethercomputer/MoA

### 5.2 Small Model Drafting + Large Model Review

NVIDIA's research on small language models (SLMs) for agentic AI recommends:

- Use SLMs for routine, narrow tasks (tool calling, simple extraction, classification)
- Reserve LLMs for complex reasoning, planning, and review
- This heterogeneous approach is both more capable and more economical
- SLMs are "sufficiently powerful, inherently more suitable, and necessarily more economical for many invocations"

**Practical architecture:**
```
User Request -> Small Model (drafts response)
  |-> Confidence check / quality gate
  |-> If passes: return response (fast + cheap)
  |-> If fails: Large Model (reviews and corrects / regenerates)
```

Paper: https://research.nvidia.com/labs/lpr/slm-agents/

### 5.3 Model Ensembling for Code Generation

**EnsLLM (2025)**:
- Generates candidate programs from multiple LLMs
- Applies CodeBLEU voting mechanism for pairwise comparison
- Ranks solutions by aggregated similarity scores
- Paper: https://arxiv.org/pdf/2503.15838

**Multi-Agent Verification (MAV, 2025)**:
- Scales test-time compute by increasing number of verifiers
- "Aspect Verifiers" -- LLMs prompted to verify specific aspects via True/False
- Combined through simple voting mechanisms
- Key finding: simple majority voting accounts for most gains; adding debate/discussion has marginal additional benefit

**Collaborative Beam Search (CBS)**:
- Generation phase: multiple LLMs generate diverse candidate steps
- Verification phase: collective consensus via average perplexity (no external verifier needed)

**Cost-Aware Majority Voting (CaMVo)**:
- On MMLU: higher accuracy with ~40% lower cost
- On IMDB: 0.17% accuracy drop while halving cost
- Implements cost-weighted voting where cheaper model votes count less

**Key insight from research**: "Simple majority voting accounts for most of the observed gains" in multi-agent systems. Expensive inter-agent communication protocols (debate, negotiation) provide marginal benefit over basic voting.

### 5.4 Cross-Validation for Hallucination Reduction

Multi-agent systems with cross-validation can improve accuracy by up to 40% on complex tasks. Pattern:
1. Multiple agents independently generate responses
2. Agents verify each other's outputs
3. Consensus mechanism selects or synthesizes final answer

---

## 6. Benchmarking Model Quality vs Cost

### 6.1 Chatbot Arena / LM Arena (LMSYS)

**What it is**: Crowdsourced benchmark platform where users compare anonymous LLM responses in head-to-head battles.

**Methodology:**
- Anonymous, randomized pairwise comparisons
- Bradley-Terry model fitted to preferences, producing Elo-like scores with confidence intervals
- Tracks 100+ models from all major providers
- Specialized arenas: text, vision, text-to-video, coding

**Current rankings (February 2026):**
- #1: Gemini-3-Pro (Elo 1492)
- Specialized coding leaderboard available separately

**Arena-Hard pipeline:**
- 500+ challenging prompts derived from live Arena data
- Cheap and fast to run (~$25)
- Better separates true reasoning capability from memorization

**Limitations:**
- General leaderboard can be misleading for specific use cases
- Favors verbose, confident-sounding responses
- Not a direct quality-per-dollar metric

Sources:
- https://lmsys.org/blog/2023-05-03-arena/
- https://lmsys.org/blog/2024-04-19-arena-hard/

### 6.2 Quality-Per-Dollar Frameworks

No single dominant "quality per dollar" benchmark exists, but several approaches:

**RouterArena metrics** (most comprehensive):
- Query-answer accuracy
- Query-answer cost
- Routing optimality (did it pick the cheapest correct model?)
- Robustness to perturbations
- Router overhead latency

**DIY quality-per-dollar calculation:**
```
Quality Score = (accuracy on your eval set) / (cost per 1000 evaluations)
```

**Practical approach:**
1. Build a task-specific eval set (100-500 examples)
2. Run all candidate models
3. Score with automated metrics + human review
4. Calculate cost-normalized performance
5. Select the cheapest model that meets your quality threshold

### 6.3 Programmatic Model Selection

**RouteLLM approach:**
- Train a classifier on preference data
- At inference time, classifier predicts whether the query needs the strong or weak model
- Configurable threshold: higher threshold = more queries to strong model = higher quality + cost

**Not Diamond approach:**
- API call that returns the recommended model for a given prompt
- Optimizes for quality, cost, or latency based on configuration

**OpenRouter Auto approach:**
- Transparent -- just set `model: "openrouter/auto"` and let the router decide
- No additional cost beyond the selected model

### 6.4 Choosing the Right Model Per Task

| Task Type | Recommended Tier | Example Models | Typical Cost/1K tasks |
|-----------|-----------------|----------------|----------------------|
| Classification / Extraction | Cheap | GPT-4o-mini, Gemini 2.0 Flash, DeepSeek V3 | $0.05-0.15 |
| Summarization | Mid | Claude Haiku 4.5, GPT-4.1 | $1-5 |
| Code generation | Mid-High | Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Pro | $5-15 |
| Complex reasoning | High | Claude Opus 4.5, o3, Gemini 2.5 Pro | $10-50 |
| Code review / debugging | Mid | Claude Sonnet 4.5, GPT-4.1 | $5-15 |
| Creative writing | Mid-High | Claude Sonnet 4.5, GPT-4o | $5-15 |
| Math / STEM | Mid (reasoning) | o4-mini, DeepSeek R1 | $2-10 |

---

## 7. Rate Limiting and Reliability

### 7.1 The Reliability Problem

Individual LLM providers rarely exceed 99.7% uptime. Production AI systems demand 99.99% uptime. This gap must be bridged through engineering.

**Common failure modes:**
- 429 Too Many Requests (rate limit exceeded)
- 500-series errors (provider internal errors)
- Timeouts (model inference takes 5-30s for complex requests; standard 10s timeouts kill valid calls)
- Context length exceeded
- Content moderation rejections
- Provider-wide outages

### 7.2 Rate Limiting Strategies

**Traditional vs. Token-Aware:**
- Traditional request-per-second limiting is insufficient for LLMs
- Token-aware rate limiting accounts for actual computational burden per call
- A single request to a 70B model can consume thousands of tokens

**Sliding window approach:**
- Track token consumption over rolling time windows
- More forgiving than fixed-window counting
- Prevents burst-then-starve patterns

**Provider rate limits (typical):**
- OpenAI: Per-minute token limits, varies by model and tier
- Anthropic: Per-minute request and token limits
- OpenRouter free models: ~20 requests/minute, 200/day

### 7.3 Reliability Patterns

#### Retries with Exponential Backoff + Jitter
```
Base pattern:
  wait_time = min(base_delay * (2 ^ attempt) + random_jitter, max_delay)

Studies show 70-80% of transient failures resolve within seconds.
Jitter prevents synchronized retry storms across clients.
```

#### Fallback Chains
```
Primary Model (Claude Sonnet)
  -> [on failure] Fallback 1 (GPT-4o)
  -> [on failure] Fallback 2 (Gemini 2.5 Pro)
  -> [on failure] Fallback 3 (open-source model)
```

Trigger conditions:
- 429 (rate limit) -> instantly reroute to secondary provider
- 500-series -> retry on another provider
- Timeout -> try faster model/provider

#### Circuit Breaker Pattern
Three states:
1. **CLOSED**: Normal operation, requests flow through
2. **OPEN**: After N consecutive failures, block all requests (prevents cascade)
3. **HALF-OPEN**: After cooldown period, allow test requests to check recovery

**Benefits over naive retries:**
- Stops traffic to failing providers proactively
- Prevents overwhelming recovering services
- Triggers fallback earlier, improving user experience

Implementation: https://github.com/gitcommitshow/resilient-llm

### 7.4 LLM Gateway Solutions

| Gateway | Key Features | Scale |
|---------|-------------|-------|
| **Portkey** | Routing to 1600+ models, fallbacks, load balancing, canary testing, circuit breakers | 10B+ requests/month, 99.9999% uptime |
| **OpenRouter** | 290+ models, auto routing, provider fallback, prompt caching | Large-scale production |
| **Helicone** | Observability-focused, cost tracking, caching | Production monitoring |
| **LiteLLM** | Open-source, 100+ providers, OpenAI-compatible | Self-hosted |

**Portkey specifics:**
- Weighted load balancing across providers
- Automatic retries (up to 5x) with exponential backoff
- Canary testing: gradually roll new models to small % of traffic
- Sub-10ms added latency
- GitHub: https://github.com/Portkey-AI/gateway

### 7.5 Production Reliability Checklist

1. **Never depend on a single provider** -- always have fallback chains
2. **Set appropriate timeouts** -- LLM inference needs 30-60s timeouts, not 10s
3. **Implement circuit breakers** -- stop hammering failing providers
4. **Use token-aware rate limiting** -- don't treat all requests equally
5. **Monitor error codes and patterns** -- track 429s, 500s, timeouts per provider
6. **Test failover regularly** -- simulate provider unavailability
7. **Cache aggressively** -- reduce dependency on live API calls
8. **Use streaming** -- get partial results even if connection drops
9. **Implement request hedging** for critical paths -- send to two providers, use first response
10. **Track cost per provider** -- failover to expensive models can blow budgets

### 7.6 Observability

Per Gartner's Hype Cycle for Generative AI 2025, AI gateways emerged as critical infrastructure. Key metrics to track:

- Error rates per provider/model
- Latency percentiles (p50, p95, p99)
- Token consumption and cost per request
- Cache hit rates
- Fallback trigger frequency
- Circuit breaker state changes

---

## 8. Key Takeaways and Recommendations

### For Immediate Implementation (Low Effort, High Impact)

1. **Enable prompt caching** -- If using Anthropic, add `cache_control` to system prompts and tool definitions. 90% savings on cached tokens with zero quality impact.

2. **Use the cheapest model that works** -- GPT-4o-mini at $0.15/M input handles 70%+ of routine tasks. Test your workload against cheap models before defaulting to expensive ones.

3. **Implement model fallbacks** -- Use OpenRouter or Portkey to automatically failover between providers. Eliminates single-provider dependency.

4. **Batch non-urgent work** -- 50% discount for free. Evaluations, data processing, report generation should all use batch APIs.

### For Medium-Term Optimization

5. **Deploy model routing** -- Use RouteLLM or OpenRouter Auto to automatically route simple queries to cheap models. 35-85% cost savings.

6. **Implement cascading** -- Try cheap model first, escalate on failure. 87-94% cost reduction in agentic workflows.

7. **Add semantic caching** for repetitive workloads -- Customer support, FAQ handling, common developer queries.

8. **Build task-specific eval sets** -- You cannot optimize what you cannot measure. 100-500 examples per task type.

### For Long-Term Architecture

9. **Distill frontier models** for high-volume tasks -- If you have a task processing millions of requests, a distilled 8B model can deliver 95% of frontier quality at 10x lower cost.

10. **Consider Mixture of Agents** for quality-critical outputs -- MoA with open-source models beats GPT-4o. Cost is higher per query but may be worth it for high-stakes outputs.

11. **Invest in observability** -- Track cost, latency, error rates, and quality per model/provider. This data drives all optimization decisions.

### Cost Optimization Priority Matrix

```
                    High Impact
                        |
  Prompt Caching ------+------ Model Routing
  Model Selection      |       Cascading
                       |
  Low Effort ----------+---------- High Effort
                       |
  Prompt Optimization  |       Distillation
  Batch Processing     |       MoA / Ensembling
                       |
                    Low Impact
```

---

## 9. References

### Model Routing

- [RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing (LMSYS)](https://lmsys.org/blog/2024-07-01-routellm/)
- [RouteLLM Paper (arXiv:2406.18665)](https://arxiv.org/abs/2406.18665)
- [RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
- [RouterArena Paper (arXiv:2510.00202)](https://arxiv.org/abs/2510.00202)
- [A Unified Approach to Routing and Cascading (arXiv:2410.10347)](https://arxiv.org/abs/2410.10347)
- [IRT-Router: Multi-LLM Routing via Item Response Theory (ACL 2025)](https://aclanthology.org/2025.acl-long.761.pdf)
- [Router-R1: Multi-Round Routing via Reinforcement Learning](https://champaignmagazine.com/2025/10/16/router-r1-and-llm-routing-research/)
- [Not Diamond Documentation](https://docs.notdiamond.ai/docs/what-is-model-routing)
- [Awesome AI Model Routing (curated list)](https://github.com/Not-Diamond/awesome-ai-model-routing)
- [Why Accenture and Martian See Model Routing as Key (VentureBeat)](https://venturebeat.com/ai/why-accenture-and-martian-see-model-routing-as-key-to-enterprise-ai-success)

### Cost Optimization

- [LLM Cost Optimization: Reducing AI Expenses by 80% (Koombea)](https://ai.koombea.com/blog/llm-cost-optimization)
- [Prompt Caching: 60% Cost Reduction (Thomson Reuters Labs)](https://medium.com/tr-labs-ml-engineering-blog/prompt-caching-the-secret-to-60-cost-reduction-in-llm-applications-6c792a0ac29b)
- [Anthropic Prompt Caching Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [GPTCache (Semantic Caching)](https://github.com/zilliztech/GPTCache)
- [AWS: Optimize LLM Costs with Effective Caching](https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/)
- [Save 50% with OpenAI Batch Requests](https://engineering.miko.ai/save-50-on-openai-api-costs-using-batch-requests-6ad41214b4ac)
- [Token Optimization Strategies (Rost Glukhov)](https://www.glukhov.org/post/2025/11/cost-effective-llm-applications/)

### OpenRouter

- [OpenRouter Documentation](https://openrouter.ai/docs/quickstart)
- [OpenRouter Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [OpenRouter Model Fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks)
- [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)
- [OpenRouter Nitro and Floor Shortcuts](https://openrouter.ai/announcements/introducing-nitro-and-floor-price-shortcuts)
- [OpenRouter Prompt Caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
- [A Practical Guide to OpenRouter (Medium)](https://medium.com/@milesk_33/a-practical-guide-to-openrouter-unified-llm-apis-model-routing-and-real-world-use-d3c4c07ed170)
- [OpenRouter Review 2025 (Skywork)](https://skywork.ai/blog/openrouter-review-2025/)

### Pricing and Token Economics

- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Anthropic Claude Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Google Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [LLM API Pricing Comparison 2025 (IntuitionLabs)](https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025)
- [Complete LLM Pricing Comparison 2026 (CloudIDR)](https://www.cloudidr.com/blog/llm-pricing-comparison-2026)
- [LLM Pricing Calculator](https://www.llm-prices.com/)
- [Helicone LLM Cost Calculator](https://www.helicone.ai/llm-cost)
- [How Coding Agents Spend Your Money (OpenReview)](https://openreview.net/forum?id=1bUeVB3fov)

### Multi-Model Architectures

- [Mixture-of-Agents Paper (arXiv:2406.04692)](https://arxiv.org/abs/2406.04692)
- [Together AI MoA Implementation](https://github.com/togethercomputer/MoA)
- [Together MoA Blog](https://www.together.ai/blog/together-moa)
- [Small Language Models for Agentic AI (NVIDIA)](https://research.nvidia.com/labs/lpr/slm-agents/)
- [Enhancing LLM Code Generation with Ensembles](https://arxiv.org/pdf/2503.15838)
- [Awesome LLM Ensemble (curated list)](https://github.com/junchenzhi/Awesome-LLM-Ensemble)
- [LLM Ensembles and MoA Explained](https://bdtechtalks.com/2025/02/17/llm-ensembels-mixture-of-agents/)
- [Multi-Agent LLM Architecture Guide 2025 (Collabnix)](https://collabnix.com/multi-agent-and-multi-llm-architecture-complete-guide-for-2025/)

### Benchmarking

- [Chatbot Arena (LMSYS)](https://lmsys.org/blog/2023-05-03-arena/)
- [Arena-Hard Pipeline](https://lmsys.org/blog/2024-04-19-arena-hard/)
- [LMSYS Chatbot Arena Leaderboard Feb 2026](https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html)
- [LMSYS Coding Leaderboard Feb 2026](https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-coding-leaderboard-2026.html)
- [LLM Leaderboards Explained (KeywordsAI)](https://www.keywordsai.co/blog/llm-leaderboards-explained)

### Speculative Decoding and Distillation

- [Speculative Decoding (NVIDIA)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Looking Back at Speculative Decoding (Google Research)](https://research.google/blog/looking-back-at-speculative-decoding/)
- [Speculative Decoding Survey](https://blog.codingconfessions.com/p/a-selective-survey-of-speculative-decoding)
- [Model Distillation for LLMs (Redis)](https://redis.io/blog/model-distillation-llm-guide/)
- [Distillation: Smaller Models as High-Performance Solutions (Microsoft)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029)
- [LLM Distillation Demystified (Snorkel AI)](https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/)

### Rate Limiting and Reliability

- [Retries, Fallbacks, and Circuit Breakers in LLM Apps (Portkey)](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [Failover Routing Strategies for LLMs (Portkey)](https://portkey.ai/blog/failover-routing-strategies-for-llms-in-production/)
- [Portkey AI Gateway (GitHub)](https://github.com/Portkey-AI/gateway)
- [Resilient-LLM (GitHub)](https://github.com/gitcommitshow/resilient-llm)
- [Building Reliable LLM Pipelines](https://ilovedevops.substack.com/p/building-reliable-llm-pipelines-error)
- [Rate Limiting in AI Gateway (TrueFoundry)](https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway)
- [Top 5 LLM Gateways in 2025 (Helicone)](https://www.helicone.ai/blog/top-llm-gateways-comparison-2025)
- [LLM Observability Guide 2026 (Portkey)](https://portkey.ai/blog/the-complete-guide-to-llm-observability/)
