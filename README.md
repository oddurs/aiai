# aiai

**AI-made AI.** A self-improving system where AI agents orchestrate, build, and evolve themselves.

## What is this?

aiai is an open framework for autonomous, self-improving AI systems. Instead of humans writing every line, AI models (Claude, GPT, Codex, etc.) self-mobilize within a structured environment to build, test, and iteratively improve their own capabilities.

Inspired by projects like [OpenClaw](https://github.com/openclaw/openclaw), multi-agent frameworks (CrewAI, MetaGPT, AutoGen), and research on recursive self-improvement (AlphaEvolve, self-evolving agents).

## Core Ideas

- **Self-mobilization** — Agents decide how to organize and what to work on
- **Self-improvement** — The system modifies its own prompts, tools, and code
- **Agentic version control** — Every AI action is git-tracked and auditable
- **Multi-model** — Use the right model for the right task (Opus for reasoning, Haiku for speed)
- **Safety-first** — Human oversight at critical points, everything reversible

## Project Structure

```
aiai/
├── CLAUDE.md                          # AI agent conventions
├── README.md                          # You are here
├── docs/
│   ├── vision.md                      # Project vision and philosophy
│   ├── architecture.md                # System architecture
│   ├── concepts.md                    # Key concepts explained
│   └── research/
│       ├── openclaw-and-agent-frameworks.md   # Agent framework analysis
│       └── self-improving-ai.md               # Self-improvement research
├── scripts/
│   ├── git-workflow.sh                # Automated git operations
│   └── agent-git.sh                   # Safe git wrapper for agents
└── .github/
    ├── workflows/ci.yml               # CI pipeline
    └── PULL_REQUEST_TEMPLATE.md       # PR template
```

## Agentic Git Workflow

aiai ships with a fully automated git workflow for AI agents:

```bash
# Auto-commit with AI-generated message from diff analysis
./scripts/git-workflow.sh auto-commit

# Create feature branches
./scripts/git-workflow.sh auto-branch feat "add memory system"

# Safe git operations (blocks destructive actions, checks for secrets)
./scripts/agent-git.sh commit "feat(core): add eval loop"
./scripts/agent-git.sh status
```

See [scripts/git-workflow.sh](scripts/git-workflow.sh) and [scripts/agent-git.sh](scripts/agent-git.sh) for full docs.

## Documentation

- [Vision](docs/vision.md) — Why aiai exists and where it's going
- [Architecture](docs/architecture.md) — How the system is structured
- [Concepts](docs/concepts.md) — Key ideas: self-improvement loops, capability bootstrapping, safety model
- [OpenClaw & Agent Frameworks](docs/research/openclaw-and-agent-frameworks.md) — Research on the multi-agent landscape
- [Self-Improving AI](docs/research/self-improving-ai.md) — State of the art in recursive self-improvement

## Status

Early stage. The foundation is laid — documentation, research, git workflows, CI. Next: building the actual agent runtime, self-improvement engine, and tool ecosystem.

## Contributing

This project is built by AI, for AI, with human guidance. Contributions welcome — both human and AI-authored PRs. See [CLAUDE.md](CLAUDE.md) for conventions.

## License

MIT
