# aiai - Architecture

## Execution Environment

```
┌──────────────────────────────────────────────┐
│              Claude Code (CLI)                 │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agent 1  │  │  Agent 2  │  │  Agent 3  │   │
│  │ (Builder) │  │(Researcher)│  │ (Evolver) │   │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘   │
│        │              │              │         │
│  ┌─────┴──────────────┴──────────────┴─────┐  │
│  │           Task / Team Coordination       │  │
│  └─────────────────┬───────────────────────┘  │
│                    │                           │
└────────────────────┼───────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │    OpenRouter API    │
          │  (Model Selection)   │
          ├─────────────────────┤
          │ Opus/GPT-4  → hard  │
          │ Sonnet/R1   → medium│
          │ Haiku/Flash → easy  │
          └─────────────────────┘
```

**Claude Code** is the runtime. Agents are Claude Code sessions using the Task tool for teams, Bash for execution, and file tools for I/O. No custom runtime to build or maintain.

**OpenRouter** is the model layer. A Python client routes requests based on task complexity, falling back through cheaper models when the expensive ones aren't needed.

**Git** is the state layer, safety net, and memory. All agent work lands as commits. `git revert` is the undo button. The git log is the full history.

## Operating Mode: Full Auto

No human gates. No PRs. No approval workflows.

```
Agent writes code → Tests pass? → Commit to main → Push
                         ↓ no
                   Fix code → Try again
```

Quality enforcement is fully automated:
- **Tests** catch bugs before commit
- **CI** catches regressions on push
- **Git revert** undoes bad commits in seconds
- **Cost tracking** prevents budget blowout
- **Secret scanning** catches credentials before commit

## System Layers

### Layer 0: Infrastructure (done)

- **Git repository** with automated workflows (`scripts/git-workflow.sh`, `scripts/agent-git.sh`)
- **CI/CD pipeline** (`.github/workflows/ci.yml`)
- **Project conventions** (`CLAUDE.md`)
- **Research base** (12 deep research documents, 10K+ lines)

### Layer 1: Model Router

Python module for cost-optimized model selection via OpenRouter.

- **Task complexity declaration** — Agents tag requests as `trivial`, `simple`, `medium`, `complex`, `critical`
- **Model mapping** — Each complexity level maps to a model tier and budget
- **Fallback chain** — If a model fails or is unavailable, fall back to the next option
- **Cost tracking** — Log spend per task, per agent, per model
- **Config-driven** — Model preferences in `config/models.yaml`

```
Complexity → Model Tier         → Example Models
trivial    → nano/micro         → Haiku, Flash-8B
simple     → fast               → Sonnet, Flash
medium     → balanced           → Sonnet, GPT-4o-mini
complex    → powerful           → Opus, GPT-4, DeepSeek-R1
critical   → most capable       → Opus, o1-pro
```

### Layer 2: Agent Runtime

How agents spawn, coordinate, and persist.

- **Agent spawning** — Claude Code Task tool creates specialized agents
- **Team coordination** — Task lists, message passing, shared context
- **Memory system** — Persistent storage in `.claude/` memory files and project docs
- **Tool registry** — Python scripts in `src/tools/` that agents can invoke

### Layer 3: Self-Improvement Engine

How the system gets better, autonomously.

- **Performance tracking** — Log task outcomes, execution time, cost, quality metrics
- **Prompt evolution** — Agents modify CLAUDE.md and agent configs directly
- **Tool creation** — Agents build new Python tools when existing ones are insufficient
- **Pattern library** — Successful patterns extracted and stored for reuse
- **No gates** — Self-modifications commit directly. Tests are the only quality check.

## Agent Types

### Builder
Writes code. Implements features, fixes bugs, creates scripts. The workhorse.

### Researcher
Investigates problems. Reads docs, searches codebases, synthesizes findings.

### Architect
Plans system changes. Decomposes complex tasks, designs interfaces.

### Evolver
Analyzes performance. Modifies prompts, tools, and workflows. Drives self-improvement.

## Data Flow

```
Task
  │
  ├─ Simple? ──→ Single agent, cheap model, direct execution
  │               → Tests pass → Commit to main → Push
  │
  └─ Complex? ─→ Team formation
                   │
                   ├─ Architect decomposes task
                   ├─ Builder(s) implement in parallel
                   ├─ Researcher gathers context as needed
                   │
                   └─→ Results committed to main
                        │
                        └─→ Evolver analyzes outcome
                             │
                             └─→ System improves itself
```

## Directory Structure

```
aiai/
├── CLAUDE.md                  # Agent conventions (agents can modify freely)
├── config/
│   └── models.yaml            # OpenRouter model routing config
├── src/
│   ├── router/                # OpenRouter client + cost-optimized model selection
│   ├── agents/                # Agent definitions and configs
│   ├── tools/                 # Python tools agents can invoke
│   ├── memory/                # Persistent memory system
│   └── evolution/             # Self-improvement engine
├── tests/                     # The real quality gate
├── scripts/
│   ├── git-workflow.sh        # Automated git operations
│   └── agent-git.sh           # Safe git wrapper (secret scan, cost tracking)
├── docs/
├── .github/
└── .claude/
    └── agents/                # Claude Code agent definitions
```

## Key Design Decisions

### Why full auto (no gates)
- Human review is the bottleneck, not the safety mechanism
- Tests are faster, more consistent, and more thorough than humans
- Git revert makes any mistake instantly fixable
- Self-improvement requires fast iteration — gates slow it down
- The system can't build itself if it has to wait for humans

### Why Claude Code as runtime
- Already handles agent spawning, tool use, file I/O, team coordination
- No custom runtime to build or maintain
- Focus on the model router and self-improvement engine, not plumbing

### Why OpenRouter
- Single API key, access to every major model
- Cost optimization is a first-class concern
- Not locked to any one provider

### Why Python only
- One language to reason about, test, and maintain
- Rich ecosystem for AI/ML tooling
- Agents understand Python deeply
