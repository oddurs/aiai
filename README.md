# aiai

Self-improving AI infrastructure. Agents build, test, and evolve their own codebase.

## What is this?

aiai is a system where AI agents autonomously write code, coordinate in teams, and improve their own tooling and capabilities over time. The primary interface is [Claude Code](https://claude.ai/code), with [OpenRouter](https://openrouter.ai) providing access to any model for cost-optimized task routing.

Built in Python. Managed by git. Designed to be serious infrastructure, not a demo.

## How it works

1. You give aiai a task (build something, research something, improve itself)
2. Agents self-organize into teams, picking the right models for each subtask
3. Work happens autonomously — code is written, tested, committed to branches
4. Human approves PRs to `main` and any self-modification changes
5. The system learns from each cycle and gets better at the next one

## Core principles

- **Cost-optimized model routing** — Cheap models for simple tasks, powerful models for hard ones, via OpenRouter
- **Autonomous within guardrails** — Agents commit freely to branches; humans approve merges to `main` and changes to agent configs
- **Everything is auditable** — Git history IS the evolution history
- **Python only** — One language, no polyglot complexity
- **Self-improving** — The framework is its own first customer

## Project structure

```
aiai/
├── CLAUDE.md                          # Agent conventions and safety rules
├── config/
│   └── models.yaml                    # OpenRouter model routing tiers
├── docs/
│   ├── vision.md                      # Why this exists
│   ├── architecture.md                # System design
│   ├── concepts.md                    # Key concepts
│   ├── founding/                      # Founding documents
│   └── research/                      # Deep research
├── scripts/
│   ├── git-workflow.sh                # Auto-commit, branch, merge, changelog
│   └── agent-git.sh                   # Safe git wrapper (secret scanning, guardrails)
└── .github/
    ├── workflows/ci.yml               # Lint, test, auto-label
    └── PULL_REQUEST_TEMPLATE.md
```

## Agentic git workflow

```bash
# Auto-commit with message generated from diff
./scripts/git-workflow.sh auto-commit

# Create typed branches
./scripts/git-workflow.sh auto-branch feat "add memory system"

# Safe agent operations (blocks force-push, scans for secrets)
./scripts/agent-git.sh commit "feat(core): add eval loop"
./scripts/agent-git.sh status
```

## Documentation

### Core
- [Vision](docs/vision.md) — Project thesis and direction
- [Architecture](docs/architecture.md) — System layers, model routing, approval gates
- [Concepts](docs/concepts.md) — Self-improvement loops, capability bootstrapping, safety model

### Founding Documents
- [Manifesto](docs/founding/manifesto.md) — The problem, the bet, the standard
- [Theory of Operation](docs/founding/theory-of-operation.md) — How the system works, from first principles
- [Engineering Principles](docs/founding/principles.md) — The rules that govern how aiai is built
- [Safety Model](docs/founding/safety-model.md) — Containment rings, approval gates, enforcement layers
- [Evolution Engine](docs/founding/evolution-engine.md) — How self-improvement works technically
- [Model Routing](docs/founding/model-routing.md) — Cost-optimized model selection design
- [Roadmap](docs/founding/roadmap.md) — Capability levels and concrete deliverables

### Research
- [Agent Memory & Knowledge Systems](docs/research/agent-memory-systems.md) — RAG, vector DBs, context management, knowledge graphs, memory safety
- [Agent Safety & Alignment](docs/research/agent-safety-and-alignment.md) — Sandboxing, capability control, alignment monitoring, red teaming, regulations
- [AI Infrastructure Landscape 2026](docs/research/ai-infrastructure-landscape-2026.md) — MCP, A2A, Claude Code, Cursor, Codex, E2B, open-source models, agent economy
- [Python Agent Ecosystem](docs/research/python-agent-ecosystem.md) — Anthropic SDK, OpenAI Agents SDK, PydanticAI, LangGraph, async patterns, testing
- [Evaluation & Benchmarks](docs/research/evaluation-and-benchmarks.md) — SWE-bench, agent benchmarks, LLM-as-judge, A/B testing, cost-quality Pareto
- [Multi-Agent Coordination](docs/research/multi-agent-coordination.md) — Debate, consensus, specialization, parallel execution, emergent behavior
- [Agent Orchestration Patterns](docs/research/agent-orchestration-patterns.md) — Production patterns, protocols (MCP/A2A), benchmarks, frameworks
- [Model Routing & Cost Optimization](docs/research/model-routing-and-cost.md) — OpenRouter, cascading, caching, real pricing data
- [Self-Improving AI (Technical)](docs/research/self-improvement-technical.md) — AlphaEvolve internals, prompt evolution, tool creation, evaluation
- [Self-Improving AI (Overview)](docs/research/self-improving-ai.md) — RSI concepts, safety, current state of the art
- [Agentic DevOps](docs/research/agentic-devops.md) — AI git workflows, automated review, CI/CD, audit trails
- [OpenClaw & Agent Frameworks](docs/research/openclaw-and-agent-frameworks.md) — OpenClaw, CrewAI, LangGraph, AutoGen, MetaGPT

## Status

Foundation phase complete. 12,000+ lines of research across 12 topics, plus 7 founding documents. Git workflows, CI, and model routing config in place.

**Next**: Build the Python OpenRouter client (`src/router/`), then the agent runtime, then the self-improvement engine. See [Roadmap](docs/founding/roadmap.md) for details.

## License

MIT
