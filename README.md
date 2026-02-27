# aiai

AI that builds itself. Fully autonomous. No human gates.

## What is this?

aiai is AI-made AI. A system where AI agents write code, test it, commit it, push it, and improve themselves — all without human review or approval. The primary interface is [Claude Code](https://claude.ai/code), with [OpenRouter](https://openrouter.ai) for cost-optimized model routing.

Built in Python. Managed by git. Runs at machine speed.

## How it works

1. You give aiai a task (build something, research something, improve itself)
2. Agents self-organize into teams, picking the right models for each subtask
3. Code is written, tested, committed directly to main, and pushed
4. No PRs, no human review — tests are the quality gate
5. The system learns from each cycle and gets better at the next one

## Core principles

- **Full auto** — No human gates. Tests are the gatekeeper, not people.
- **Build itself** — aiai's first customer is aiai. The system constructs and improves itself.
- **Cost-optimized** — Cheap models for simple tasks, powerful models for hard ones, via OpenRouter
- **Everything is auditable** — Git history IS the evolution history. Revert anything.
- **Python only** — One language, no polyglot complexity

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
- [Architecture](docs/architecture.md) — System layers, model routing, full-auto operation
- [Concepts](docs/concepts.md) — Self-improvement loops, capability bootstrapping, safety model

### Founding Documents
- [Manifesto](docs/founding/manifesto.md) — The problem, the bet, the standard
- [Theory of Operation](docs/founding/theory-of-operation.md) — How the system works, from first principles
- [Engineering Principles](docs/founding/principles.md) — The rules that govern how aiai is built
- [Safety Model](docs/founding/safety-model.md) — Testing as safety, git revert, cost controls, observability
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
- [AI Building AI](docs/research/ai-building-ai.md) — AI writing AI code, AutoML, NAS, bootstrapping, acceleration thesis, self-replicating AI
- [Fully Autonomous Systems](docs/research/fully-autonomous-systems.md) — Zero-intervention architectures, auto-merge, self-healing, runaway prevention, trust through verification
- [Autonomous Software Development](docs/research/autonomous-software-development.md) — AI-generated production code, full-stack agents, autonomous bug fixing, software factories

## Status

Foundation phase complete. 15,000+ lines of research across 15 topics, plus 7 founding documents. Git workflows, CI, and model routing config in place. All docs updated to full-auto mode (no human gates).

**Next**: Build the Python OpenRouter client (`src/router/`), then the agent runtime, then the self-improvement engine. See [Roadmap](docs/founding/roadmap.md) for details.

## License

MIT
