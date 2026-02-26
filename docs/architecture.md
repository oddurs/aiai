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

**Git** is the state layer. All agent work lands as commits. Branches are workspaces. The git log is the audit trail.

## System Layers

### Layer 0: Infrastructure
What's already built.

- **Git repository** with automated workflows (`scripts/git-workflow.sh`, `scripts/agent-git.sh`)
- **CI/CD pipeline** (`.github/workflows/ci.yml`)
- **Project conventions** (`CLAUDE.md`)

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
How the system gets better.

- **Performance tracking** — Log task outcomes, execution time, cost, quality metrics
- **Prompt evolution** — Agents propose modifications to CLAUDE.md and agent configs
- **Tool creation** — Agents build new Python tools when existing ones are insufficient
- **Pattern library** — Successful patterns extracted and stored for reuse
- **Approval gate** — All self-modifications require human PR approval before merging

## Approval Gates

```
┌───────────────────────────────────────────┐
│              Autonomous Zone               │
│                                           │
│  Write code, run tests, commit to         │
│  branches, create tools, coordinate       │
│  teams, research, analyze                 │
│                                           │
├───────────────────────────────────────────┤
│           Human Approval Required          │
│                                           │
│  ✋ Merge PR to main                       │
│  ✋ Modify CLAUDE.md                       │
│  ✋ Modify agent configs (.claude/agents/) │
│  ✋ Modify self-improvement system         │
│  ✋ Change model routing config            │
│  ✋ Delete branches or data                │
│                                           │
└───────────────────────────────────────────┘
```

## Agent Types

### Builder
Writes code. Implements features, fixes bugs, creates scripts. The workhorse.

### Researcher
Investigates problems. Reads docs, searches codebases, synthesizes findings. Informs decisions.

### Architect
Plans system changes. Decomposes complex tasks, designs interfaces, reviews modifications.

### Reviewer
Reviews code for quality and safety. Validates changes don't break things. Checks for security issues.

### Evolver
Analyzes performance. Proposes improvements to prompts, tools, and workflows. Drives self-improvement.

## Data Flow

```
Task
  │
  ├─ Simple? ──→ Single agent, cheap model, direct execution
  │
  └─ Complex? ─→ Team formation
                   │
                   ├─ Architect decomposes task
                   ├─ Builder(s) implement in parallel
                   ├─ Researcher gathers context as needed
                   ├─ Reviewer validates output
                   │
                   └─→ Results committed to branch
                        │
                        └─→ PR created → Human approves → Merge to main
                                                            │
                                                            └─→ Evolver analyzes outcome
                                                                 │
                                                                 └─→ System improves
```

## Directory Structure

```
aiai/
├── CLAUDE.md                  # Agent conventions (approval-gated)
├── config/
│   └── models.yaml            # OpenRouter model routing config (approval-gated)
├── src/
│   ├── router/                # OpenRouter client + cost-optimized model selection
│   ├── agents/                # Agent definitions and configs
│   ├── tools/                 # Python tools agents can invoke
│   ├── memory/                # Persistent memory system
│   └── evolution/             # Self-improvement engine
├── tests/
├── scripts/
│   ├── git-workflow.sh        # Automated git operations
│   └── agent-git.sh           # Safe git wrapper for agents
├── docs/
├── .github/
└── .claude/
    └── agents/                # Claude Code agent definitions (approval-gated)
```

## Key Design Decisions

### Why Claude Code as runtime
- Already handles agent spawning, tool use, file I/O, team coordination
- No custom runtime to build or maintain
- Agents get shell access, web access, and the full Claude toolchain
- Focus effort on the model router and self-improvement engine, not plumbing

### Why OpenRouter
- Single API key, access to every major model
- Cost optimization is a first-class concern
- Not locked to any one provider
- Easy to swap models as new ones ship

### Why Python only
- One language to reason about, test, and maintain
- Rich ecosystem for AI/ML tooling
- Agents understand Python deeply
- Bash for scripts is fine; no need for TypeScript/Node
