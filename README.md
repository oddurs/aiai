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
├── docs/
│   ├── vision.md                      # Why this exists
│   ├── architecture.md                # System design
│   ├── concepts.md                    # Key concepts
│   └── research/                      # Background research
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

- [Vision](docs/vision.md) — Project thesis and direction
- [Architecture](docs/architecture.md) — System layers, model routing, approval gates
- [Concepts](docs/concepts.md) — Self-improvement loops, capability bootstrapping, safety model
- [Research: Agent Frameworks](docs/research/openclaw-and-agent-frameworks.md) — OpenClaw, CrewAI, LangGraph, etc.
- [Research: Self-Improving AI](docs/research/self-improving-ai.md) — AlphaEvolve, recursive improvement, safety

## Status

Foundation phase. Git workflows, CI, docs, and research are in place. Next: Python agent runtime, OpenRouter integration, model routing, self-improvement engine.

## License

MIT
