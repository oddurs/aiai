# Model Routing

The technical design of aiai's cost-optimized model selection system.

## Why Model Routing Matters

The difference between "use Opus for everything" and "use the right model for each task" is roughly 50-100x in cost for equivalent total output quality. A system that can't manage model selection can't be cost-effective infrastructure.

```
Scenario: 1000 tasks/day

Naive (Opus for everything):
  1000 × ~$0.10 avg = $100/day = $3,000/month

Cost-optimized:
  700 trivial/simple × ~$0.001 avg  = $0.70
  200 medium         × ~$0.01 avg   = $2.00
   80 complex        × ~$0.05 avg   = $4.00
   20 critical       × ~$0.15 avg   = $3.00
                              Total  = $9.70/day = $291/month

Savings: ~90%
```

The quality difference is negligible for the trivial/simple tasks. Haiku can rename a variable as well as Opus. The savings come from not wasting expensive compute on cheap tasks.

## Architecture

```
┌───────────────────────────────────────────┐
│              Agent Request                 │
│                                           │
│  task: "rename variable in router.py"     │
│  complexity: "trivial"                    │
│  max_tokens: 500                          │
│  context: [file contents]                 │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│           Complexity Router                │
│                                           │
│  1. Read config/models.yaml               │
│  2. Map complexity → tier                 │
│  3. Select first available model in tier  │
│  4. Apply cost/token constraints          │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│           OpenRouter Client                │
│                                           │
│  1. Format request for OpenRouter API     │
│  2. Send request                          │
│  3. Handle response / errors              │
│  4. Log cost and performance              │
└───────────────┬───────────────────────────┘
                │
           ┌────┴────┐
           │         │
        success    failure
           │         │
           ▼         ▼
        return    fallback
        result    to next model
                  in tier, then
                  escalate to
                  next tier
```

## Complexity Levels

### Trivial
- Single-line changes, renames, formatting
- No reasoning required, just execution
- Model: nano tier (Haiku, Flash-Lite)
- Budget: < $0.005 per request

### Simple
- Single-file edits, straightforward implementations
- Clear requirements, no ambiguity
- Model: fast tier (Sonnet, Flash)
- Budget: < $0.02 per request

### Medium
- Multi-file changes, moderate reasoning
- Some ambiguity or design decisions
- Model: balanced tier (Sonnet, GPT-4o-mini)
- Budget: < $0.10 per request

### Complex
- Architecture decisions, hard bugs, novel implementations
- Requires deep reasoning, multiple considerations
- Model: powerful tier (Opus, GPT-4, DeepSeek-R1)
- Budget: < $0.50 per request

### Critical
- Safety-critical changes, system-wide modifications
- Highest stakes, needs best available reasoning
- Model: max tier (Opus, o1-pro)
- Budget: < $2.00 per request

## Self-Assessed Complexity

Agents assess their own task complexity. This is a deliberate design choice:

**Why self-assessment**: The agent knows the most about the task it's performing. External assessment would require a separate model call, adding cost and latency.

**Risk**: Agents might over-estimate complexity (wasting money on expensive models) or under-estimate (producing low-quality output with cheap models).

**Mitigation**: The evolution engine tracks the relationship between declared complexity and actual outcomes. If an agent consistently declares "complex" for tasks that Haiku could handle, the pattern is identified and a routing adjustment is proposed.

### Complexity Heuristics

Agents should assess complexity based on:

| Factor | Lower complexity | Higher complexity |
|--------|-----------------|-------------------|
| Files affected | 1 file | Multiple files, multiple systems |
| Reasoning depth | Direct, obvious | Requires tradeoffs, design decisions |
| Novelty | Similar to past tasks | Never done before |
| Risk | Easily testable, reversible | Hard to test, affects other systems |
| Context needed | Minimal | Requires understanding multiple modules |

## Fallback Chains

When a model fails (API error, rate limit, quality issue), the router falls back:

```
Tier: fast
  Try: claude-sonnet-4
  ↓ fail
  Try: gemini-2.0-flash
  ↓ fail
  Try: gpt-4o-mini
  ↓ fail
  Escalate to next tier: balanced
    Try: claude-sonnet-4 (with higher token limit)
    ↓ fail
    Try: deepseek-r1
    ...
```

### Failure types

- **API error (5xx)**: Retry once, then fall back to next model
- **Rate limit (429)**: Wait and retry, or fall back to different provider
- **Timeout**: Fall back to next model (don't retry same model)
- **Quality failure**: Escalate to next tier (current tier isn't capable enough)
- **Budget exceeded**: Block the request, log a warning

### Quality failure detection

How does the router know if the output quality is insufficient? Several signals:

1. **Syntax errors**: If the output is code and it doesn't parse, quality is insufficient
2. **Test failures**: If the output is a code change and tests fail, quality may be insufficient
3. **Self-reported failure**: Agent declares the model's output wasn't good enough
4. **Length mismatch**: Output is suspiciously short or truncated

When quality failure is detected, the router escalates to a more capable model and logs the event for the evolution engine.

## Cost Controls

### Per-request warnings
If a single request is estimated to cost more than the configured threshold (default: $1.00), the agent is warned before the request is sent.

### Daily budget
When the daily spend exceeds the configured budget (default: $50.00), all requests are blocked until the next day or a human raises the limit.

### Cost logging
Every request is logged to `logs/model-costs.jsonl` with full details:
- Model used, tokens in/out, cost
- Task ID, agent ID, complexity level
- Timestamp, latency

### Cost reporting
Periodic reports (daily, weekly) summarize:
- Total spend by model, agent, task type
- Average cost per task type
- Trends (is cost going up or down?)
- Top expensive tasks
- Wasted spend (escalations, retries)

## Configuration

Model routing is configured in `config/models.yaml`:

```yaml
tiers:
  nano:
    models:
      - anthropic/claude-haiku-4-5
      - google/gemini-2.0-flash-lite
    max_tokens: 1024

  fast:
    models:
      - anthropic/claude-sonnet-4
      - google/gemini-2.0-flash
      - openai/gpt-4o-mini
    max_tokens: 4096

  # ... etc
```

This file is **approval-gated**: changes require human PR approval. This prevents agents from routing themselves to expensive models without oversight.

## Future: Learned Routing

The initial router is rule-based: complexity → tier → model. Over time, the evolution engine can improve routing based on observed data:

- "DeepSeek-R1 handles refactoring tasks 20% better than Sonnet at similar cost" → adjust tier preferences for refactoring
- "Haiku fails on tasks with >3 files 45% of the time" → adjust complexity heuristics to escalate multi-file tasks
- "Build tasks rarely need more than 2000 output tokens" → lower max_tokens for build tasks to save cost

These improvements go through the standard evolution cycle: observe → analyze → hypothesize → implement → validate → gate.
