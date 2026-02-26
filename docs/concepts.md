# aiai - Key Concepts

## Self-Improvement Loop

The core mechanism. After every task, the system evaluates what happened and applies improvements.

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ Execute  │────▶│ Evaluate  │────▶│ Improve  │
│ Task     │     │ Outcome   │     │ System   │
└─────────┘     └──────────┘     └─────────┘
     ▲                                  │
     └──────────────────────────────────┘
```

**What agents can improve autonomously:**
- Tool implementations in `src/tools/`
- Coordination patterns and task decomposition strategies
- Test coverage and quality checks

**What requires human approval (PR to main):**
- Agent prompts and system instructions (CLAUDE.md)
- Model routing configuration (`config/models.yaml`)
- Agent definitions (`.claude/agents/`)
- The self-improvement engine itself (`src/evolution/`)

This split keeps the system productive while ensuring humans control how agents think and what models they use.

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

**Why this matters:** A system that runs Opus for everything is expensive and slow. A system that runs Haiku for everything is cheap but makes mistakes on hard tasks. Cost-optimized routing gets both right.

## Agent Self-Mobilization

Agents organize themselves based on the task, not a rigid configuration.

1. **Task arrives** — Human or agent defines a goal
2. **Complexity assessment** — Is this a single-agent or multi-agent task?
3. **Team forms** — For complex tasks, agents with relevant capabilities spawn
4. **Roles emerge** — Agents claim subtasks based on their type (Builder, Researcher, etc.)
5. **Parallel execution** — Independent subtasks run concurrently
6. **Integration** — Results merge, tests run, code commits to a branch
7. **Team dissolves** — Agents become available for the next task

## Capability Bootstrapping

The system builds capabilities incrementally. Each level enables the next.

**Level 0** (done): Git workflows, CI, documentation, conventions
**Level 1** (next): OpenRouter client, model routing, cost tracking
**Level 2**: Agent runtime, team coordination, persistent memory
**Level 3**: Performance tracking, prompt evolution, self-evaluation
**Level 4**: Autonomous tool creation, pattern discovery, architecture evolution

The system is currently at Level 0, working toward Level 1.

## Approval Gates

The autonomy model is "free within branches, gated at merge."

```
Autonomous (no approval needed):
├── Write code to feature branches
├── Run tests
├── Create/delete feature branches
├── Spawn agent teams
├── Build tools in src/tools/
├── Research and analyze
└── Commit to any non-protected branch

Human approval required:
├── PR merge to main
├── Modify CLAUDE.md
├── Modify agent configs
├── Modify model routing config
├── Modify self-improvement engine
└── Delete data or protected branches
```

This is enforced by:
- GitHub branch protection rules on `main`
- `scripts/agent-git.sh` blocking destructive operations
- Convention in CLAUDE.md (agents are instructed to follow these rules)

## Memory

Agents need to remember things across sessions to improve.

**Session memory** — Current task context, conversation history. Handled by Claude Code.

**Project memory** — CLAUDE.md conventions, `.claude/memory/` files, past decisions. Persists across sessions.

**Evolutionary memory** — Performance baselines, modification history, successful patterns. Stored in `src/evolution/` data files and git history.

Git history is itself a form of memory — agents can read past commits, diffs, and PR discussions to understand why decisions were made.

## Agentic Version Control

Git is the system's journal. Every agent action is a git operation.

- **Branches** = workspaces where agents operate freely
- **Commits** = atomic units of work, always traceable to an agent
- **PRs** = the approval gate between autonomous work and `main`
- **Merges** = successful improvements integrated into the system
- **Reverts** = failed experiments rolled back cleanly
- **Tags** = capability milestones (Level 0, Level 1, etc.)

The git log tells the story of how the system evolved.

## Safety Model

### Containment Rings

```
Ring 0 (Free):      Read files, search code, analyze, research
Ring 1 (Logged):    Write files, run tests, commit to branches
Ring 2 (Gated):     PR to main, modify agent configs, change model routing
Ring 3 (Blocked):   Force-push to main, delete repos, disable safety checks
```

### Invariants
1. Agents cannot merge to `main` without human approval
2. Agents cannot modify their own rules (CLAUDE.md, agent configs) without human approval
3. All changes are reversible via git revert
4. `scripts/agent-git.sh` enforces Ring 3 blocks at the tool level
5. Cost spend is tracked and logged — no unbounded API calls
