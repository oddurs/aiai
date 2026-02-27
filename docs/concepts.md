# aiai - Key Concepts

## Self-Improvement Loop

The core mechanism. After every task, the system evaluates what happened and applies improvements. No human in the loop.

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ Execute  │────▶│ Evaluate  │────▶│ Improve  │
│ Task     │     │ Outcome   │     │ System   │
└─────────┘     └──────────┘     └─────────┘
     ▲                                  │
     └──────────────────────────────────┘
```

**Everything is improvable:**
- Agent prompts and system instructions (CLAUDE.md)
- Tool implementations in `src/tools/`
- Coordination patterns and task decomposition
- Model routing configuration
- Agent definitions and configs
- Test suites and quality checks
- The evolution engine itself
- Documentation and conventions

**Quality control: tests, not humans.** Every improvement must pass the test suite. If it does, it ships. If it doesn't, it gets reverted.

## Cost-Optimized Model Routing

Not every task needs Opus. The model router matches task complexity to the cheapest model that can handle it.

```
Agent declares complexity:
  "This is a simple file rename"     → Haiku ($0.001)
  "This is a medium refactor"        → Sonnet ($0.01)
  "This is a complex architecture"   → Opus ($0.10)
```

**How it works:**
1. Agent tags each request with a complexity level: `trivial`, `simple`, `medium`, `complex`, `critical`
2. Router maps complexity to a model tier via `config/models.yaml`
3. Request goes to OpenRouter with the selected model
4. If the model fails or is unavailable, fallback to the next tier
5. Cost is logged per task, per agent, per model

## Agent Self-Mobilization

Agents organize themselves based on the task, not a rigid configuration.

1. **Task arrives** — Human or agent defines a goal
2. **Complexity assessment** — Single-agent or multi-agent?
3. **Team forms** — Agents with relevant capabilities spawn
4. **Roles emerge** — Agents claim subtasks
5. **Parallel execution** — Independent subtasks run concurrently
6. **Integration** — Results merge, tests run, code commits
7. **Team dissolves** — Agents available for the next task

## Capability Bootstrapping

The system builds itself incrementally. Each level enables the next.

**Level 0** (done): Git workflows, CI, documentation, research
**Level 1** (next): OpenRouter client, model routing, cost tracking
**Level 2**: Agent runtime, team coordination, persistent memory
**Level 3**: Performance tracking, prompt evolution, self-evaluation
**Level 4**: Fully autonomous operation — builds, improves, and evolves without human input

## Full Auto Operation

**No gates. No PRs. No human review.**

```
Agent writes code
  → Tests pass?
    → Yes: commit, push, done
    → No: fix the code, try again
```

Quality is maintained by:
- **Automated testing** — Tests are the gatekeeper
- **CI pipeline** — Catches regressions on every push
- **Git revert** — Bad changes are undone instantly
- **Cost tracking** — Prevents budget blowout
- **Evolution engine** — Tracks metrics, drives improvement

The human's role: set direction, watch the git log.

## Memory

Agents remember across sessions to improve.

**Session memory** — Current task context. Handled by Claude Code.

**Project memory** — CLAUDE.md, `.claude/memory/`, docs. Persists across sessions.

**Evolutionary memory** — Performance baselines, what worked, what didn't. Git history + `src/evolution/` data files.

## Agentic Version Control

Git is the system's journal, safety net, and brain.

- **Commits** = atomic units of work, always traceable
- **Branches** = parallel experiments
- **Merges** = direct to main when tests pass
- **Reverts** = undo anything instantly
- **Tags** = capability milestones (Level 0, Level 1, etc.)
- **Log** = the full story of how the system evolved

## Safety Without Gates

```
Tests         → catch bugs before commit
Git revert    → undo anything in seconds
Secret scan   → catch credentials before commit
Cost limits   → prevent budget blowout
Loop detect   → catch agents stuck retrying
Observability → monitor trends after the fact
```

No containment rings. No approval gates. No human review. Just automated quality enforcement and the ability to undo anything.
