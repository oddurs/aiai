# Roadmap

Where aiai is going, in concrete terms.

## Current State: Level 0 — Foundation

What exists today:

- [x] Git repository with automated workflows
- [x] `scripts/git-workflow.sh` — auto-commit, auto-branch, auto-merge, changelog, snapshot
- [x] `scripts/agent-git.sh` — safe git wrapper with secret scanning, destructive-op blocking
- [x] CI/CD pipeline — lint (ruff, mypy, shellcheck), test, auto-label PRs
- [x] `CLAUDE.md` — agent conventions, safety rules, approval gates
- [x] `config/models.yaml` — model routing configuration (tiers, fallbacks, cost limits)
- [x] Founding documentation — vision, architecture, concepts, safety model, theory of operation
- [x] Research base — OpenClaw analysis, agent frameworks, self-improving AI, orchestration patterns

The foundation is solid. Everything from here is building on this base.

---

## Level 1: Model Router

**Goal**: Python module that routes requests to the optimal model via OpenRouter.

### Deliverables

- [ ] `src/router/__init__.py` — Public API
- [ ] `src/router/client.py` — OpenRouter HTTP client
- [ ] `src/router/selector.py` — Complexity → tier → model selection logic
- [ ] `src/router/fallback.py` — Fallback chain implementation
- [ ] `src/router/cost.py` — Cost tracking and budget enforcement
- [ ] `src/router/config.py` — Config loader for `config/models.yaml`
- [ ] `tests/test_router/` — Full test suite
- [ ] `logs/` directory with `.gitkeep`

### Key decisions
- Use `httpx` for async HTTP (OpenRouter API)
- Config loaded from YAML at startup, cached in memory
- Cost logged to JSONL file (one line per request)
- Daily budget enforced in-process (not via external system)

### Success criteria
- Can route a request to the correct model based on declared complexity
- Falls back gracefully when a model is unavailable
- Tracks and reports cost per request
- Blocks requests when daily budget is exceeded
- All functionality covered by tests

---

## Level 2: Agent Runtime

**Goal**: Python framework for spawning, coordinating, and managing agents.

### Deliverables

- [ ] `src/agents/base.py` — Base agent class with common behavior
- [ ] `src/agents/builder.py` — Builder agent (writes code)
- [ ] `src/agents/researcher.py` — Researcher agent (gathers information)
- [ ] `src/agents/architect.py` — Architect agent (plans and decomposes)
- [ ] `src/agents/reviewer.py` — Reviewer agent (validates changes)
- [ ] `src/agents/evolver.py` — Evolver agent (proposes improvements)
- [ ] `src/coordination/team.py` — Team formation and lifecycle
- [ ] `src/coordination/task_router.py` — Task → agent routing
- [ ] `src/memory/project.py` — Project memory read/write
- [ ] `src/memory/session.py` — Session memory management
- [ ] `src/tools/registry.py` — Tool registry (discover and invoke tools)
- [ ] `tests/test_agents/`
- [ ] `tests/test_coordination/`

### Key decisions
- Agents are Python classes that wrap Claude Code Task tool invocations
- Team coordination uses task lists (compatible with Claude Code's Task tools)
- Memory is file-based (CLAUDE.md, .claude/memory/, docs/)
- Tools are Python scripts with a standard interface

### Success criteria
- Can spawn a team of agents for a complex task
- Agents coordinate via shared task lists
- Agent types have distinct behaviors and model preferences
- Memory persists across agent sessions
- Tools are discoverable and invocable by agents

---

## Level 3: Self-Improvement Engine

**Goal**: Automated observation, analysis, and improvement proposal system.

### Deliverables

- [ ] `src/evolution/metrics.py` — Metric collection and storage
- [ ] `src/evolution/analyzer.py` — Pattern detection in metrics
- [ ] `src/evolution/hypothesis.py` — Improvement hypothesis generation
- [ ] `src/evolution/validator.py` — Before/after comparison framework
- [ ] `src/evolution/reporter.py` — Periodic improvement reports
- [ ] `src/evolution/patterns/` — Pattern library (reusable solutions)
- [ ] `config/evolution.yaml` — Evolution engine configuration
- [ ] `tests/test_evolution/`

### Key decisions
- Metrics stored as JSONL (append-only, easy to process)
- Analysis runs on schedule or on-demand
- Hypotheses are structured data (not free-text)
- All evolution changes go through the standard PR workflow

### Success criteria
- System tracks metrics for every task execution
- Patterns are automatically identified (cost outliers, failure patterns, slow tasks)
- Improvement hypotheses are generated with evidence
- Before/after comparisons validate improvements
- Reports are generated weekly

---

## Level 4: Autonomous Operation

**Goal**: The system operates with minimal human intervention, improving itself continuously.

### Deliverables

- [ ] Automatic tool creation when agents encounter limitations
- [ ] Prompt evolution based on measured outcomes
- [ ] Pattern discovery and storage across task types
- [ ] Architecture evolution proposals for significant structural changes
- [ ] Learned model routing (data-driven tier adjustments)
- [ ] Cost optimization that improves over time

### Success criteria
- The system demonstrably improves on key metrics month over month
- Human intervention is limited to PR approval at merge gates
- New capabilities emerge without human specification
- Cost per task trends downward over time
- Quality metrics trend upward over time

---

## Non-Goals (for now)

Things we're explicitly not building in the near term:

- **Web UI**: The interface is Claude Code CLI + GitHub PRs. No dashboard.
- **Hosted service**: This runs locally. No cloud deployment.
- **Model training**: aiai improves through code/prompt modification, not fine-tuning.
- **Multi-repo**: One repo for now. Multi-repo support can come later.
- **Plugin marketplace**: Tools are in `src/tools/`. No external plugin system.

## Versioning

Capability levels correspond to version milestones:

| Level | Version | Tag |
|-------|---------|-----|
| 0 | v0.1.0 | `level-0-foundation` |
| 1 | v0.2.0 | `level-1-router` |
| 2 | v0.3.0 | `level-2-agents` |
| 3 | v0.4.0 | `level-3-evolution` |
| 4 | v1.0.0 | `level-4-autonomous` |

v1.0.0 is when the system can meaningfully operate and improve itself with only PR approval as human input.
