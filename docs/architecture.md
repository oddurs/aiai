# aiai - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                    aiai Core Loop                     │
│                                                       │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│   │  Observe   │───▶│  Analyze   │───▶│  Modify    │   │
│   └───────────┘    └───────────┘    └───────────┘   │
│         ▲                                   │         │
│         │           ┌───────────┐           │         │
│         └───────────│ Integrate  │◀──────────┘         │
│                     └───────────┘                     │
│                          │                            │
│                     ┌───────────┐                     │
│                     │   Test     │                     │
│                     └───────────┘                     │
└─────────────────────────────────────────────────────┘
```

## Layer Architecture

### Layer 0: Infrastructure
The foundation that everything runs on.

- **Git repository** — All state is version-controlled
- **Agentic git workflow** — Scripts for autonomous commits, branches, merges
- **CI/CD pipeline** — Automated testing and validation on every change
- **Model access** — API connections to Claude, GPT, or any capable model

### Layer 1: Agent Runtime
The environment where agents execute.

- **Agent spawning** — Create specialized agents on demand
- **Tool registry** — Available tools agents can use (shell, file I/O, web, APIs)
- **Memory system** — Persistent storage across agent sessions
- **Communication bus** — Agents send messages, share findings, coordinate

### Layer 2: Self-Improvement Engine
The mechanism by which the system improves itself.

- **Performance observer** — Tracks what works and what doesn't
- **Prompt evolver** — Modifies system prompts based on outcomes
- **Tool builder** — Agents create new tools when existing ones are insufficient
- **Pattern library** — Stores successful patterns for reuse

### Layer 3: Task Execution
Where actual work gets done.

- **Team formation** — Agents self-organize into teams for complex tasks
- **Task decomposition** — Break problems into parallelizable sub-tasks
- **Result synthesis** — Merge outputs from multiple agents
- **Quality gates** — Automated checks before integration

## Agent Types

### Architect Agent
- Plans system changes and new features
- Decomposes complex tasks
- Reviews and approves modifications

### Builder Agent
- Writes code, scripts, and configurations
- Implements features and fixes
- Creates new tools

### Researcher Agent
- Investigates problems and gathers information
- Reads documentation and codebases
- Synthesizes findings into actionable insights

### Reviewer Agent
- Reviews code changes for quality and safety
- Validates that modifications don't break existing functionality
- Checks for security issues

### Evolver Agent
- Analyzes system performance metrics
- Proposes improvements to prompts, tools, and workflows
- Runs A/B tests on system modifications

## Data Flow

```
Task Input
    │
    ▼
┌──────────────┐
│ Task Router   │ ── Decides which agents to spawn
└──────────────┘
    │
    ▼
┌──────────────┐
│ Agent Team    │ ── Agents work in parallel
│ ┌──┐┌──┐┌──┐│
│ │A1││A2││A3││
│ └──┘└──┘└──┘│
└──────────────┘
    │
    ▼
┌──────────────┐
│ Integration   │ ── Results merged, tested, committed
└──────────────┘
    │
    ▼
┌──────────────┐
│ Self-Review   │ ── System evaluates its own performance
└──────────────┘
    │
    ▼
┌──────────────┐
│ Evolution     │ ── Improvements applied to the system itself
└──────────────┘
```

## Directory Structure

```
aiai/
├── CLAUDE.md              # AI agent conventions and instructions
├── README.md              # Project overview
├── docs/
│   ├── vision.md          # Project vision and philosophy
│   ├── architecture.md    # This file
│   ├── concepts.md        # Key concepts explained
│   ├── research/          # Research on related projects
│   └── guides/            # How-to guides
├── scripts/
│   ├── git-workflow.sh    # Automated git operations
│   └── agent-git.sh       # Safe git wrapper for agents
├── src/                   # Core source code (future)
│   ├── agents/            # Agent definitions and configs
│   ├── tools/             # Tools available to agents
│   ├── memory/            # Persistent memory system
│   └── evolution/         # Self-improvement engine
├── tests/                 # Test suite
├── .github/
│   ├── workflows/         # CI/CD pipelines
│   └── PULL_REQUEST_TEMPLATE.md
└── .claude/
    └── agents/            # Custom agent definitions
```

## Key Design Decisions

### Why Git as the Backbone
- Every change is auditable
- Branching enables parallel experimentation
- Rollback is always possible
- Diffs make changes reviewable
- Standard tooling that AI models already understand

### Why Multi-Model
- Different models have different strengths
- Opus for complex reasoning and architecture
- Sonnet/Haiku for fast, routine tasks
- Future models slot in without redesign
- Cost optimization through model selection

### Why Self-Modifying
- The biggest bottleneck in AI systems is human iteration speed
- AI can test modifications faster than humans can review PRs
- Compound improvements: each improvement enables the next
- The system converges on what actually works, not what we think should work
