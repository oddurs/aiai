# Theory of Operation

How aiai works, from first principles.

## The Execution Model

### Everything is a task

Every action in aiai begins with a task. A task is a natural language description of a desired outcome:
- "Implement the OpenRouter client"
- "Fix the failing test in test_router.py"
- "Research how Voyager stores learned skills"
- "Improve the commit message generator"

Tasks come from two sources: humans and agents. A human creates the initial task. Agents decompose it into subtasks, and those subtasks may spawn further subtasks. The task tree is the system's execution plan.

### Tasks route to agents

When a task arrives, the system assesses its complexity:

```
Task: "Rename a variable in router.py"
Complexity: trivial
Route: single agent, nano-tier model

Task: "Build the OpenRouter client with fallback chains"
Complexity: complex
Route: team of agents (architect + builder + reviewer), powerful-tier models

Task: "Redesign the memory system architecture"
Complexity: critical
Route: architect agent with max-tier model, human approval before execution
```

The routing decision considers:
- **Scope**: How many files? How many systems affected?
- **Novelty**: Has the system done something similar before?
- **Risk**: Could this break existing functionality?
- **Cost**: What's the budget for this task?

### Agents execute within branches

Every non-trivial task gets a git branch. The branch is the agent's workspace. Within that branch, the agent has full autonomy: write files, run commands, create commits, iterate.

```
main ─────────────────────────────────────────── (protected)
  │
  ├── feat/openrouter-client ──── agent works here freely
  │     ├── commit: "add base client"
  │     ├── commit: "add fallback logic"
  │     ├── commit: "add tests"
  │     └── PR → main (requires human approval)
  │
  ├── fix/test-router ──── agent works here freely
  │     ├── commit: "fix assertion"
  │     └── PR → main (requires human approval)
```

This model gives agents the freedom to experiment, make mistakes, and iterate — all without affecting the mainline codebase. Failed branches are simply deleted. Successful branches merge through PRs.

### Results merge through PRs

The PR is the contract between autonomous agents and human oversight. Every PR includes:
- What changed and why (generated from commits)
- Test results
- Cost of execution (models used, tokens consumed)
- Whether the change was AI-authored

The human decides: merge, request changes, or close.

## The Self-Improvement Model

### What improves

The system has five improvable surfaces:

1. **Tools** (`src/tools/`): Python scripts that agents invoke. Agents can create new tools or improve existing ones when they encounter limitations.

2. **Prompts** (`CLAUDE.md`, `.claude/agents/`): The instructions that shape agent behavior. When an agent finds a better way to phrase instructions, it proposes the change.

3. **Workflows** (`scripts/`): The automation scripts for git, CI, and operations. If a workflow is slow or broken, agents fix it.

4. **Patterns** (`src/evolution/patterns/`): Reusable solutions to common problems. When an agent solves a novel problem, the solution is extracted and stored for reuse.

5. **Configuration** (`config/`): Model routing, cost limits, agent settings. Optimization of these parameters improves system performance and efficiency.

### How improvement happens

```
Step 1: Execute
  Agent performs a task. Everything is logged:
  execution time, model used, tokens consumed, outcome.

Step 2: Evaluate
  After the task, the system evaluates:
  - Did the task succeed? (tests pass, output correct)
  - How long did it take?
  - How much did it cost?
  - Were there errors or retries?

Step 3: Identify
  Compare against baselines:
  - Is this task type getting faster or slower?
  - Is cost per task trending down?
  - Are error rates increasing?
  - Are there patterns in what fails?

Step 4: Propose
  If an improvement is identified, an agent creates a branch:
  evolve/optimize-commit-messages
  evolve/add-caching-to-router
  evolve/improve-test-coverage

Step 5: Validate
  The improvement is tested:
  - Run the existing test suite
  - Compare metrics before and after
  - Check for regressions

Step 6: Gate
  If the change modifies agent behavior (prompts, configs, evolution system):
  → PR requires human approval

  If the change is to tools, tests, or non-gated code:
  → Can merge after CI passes (still via PR for auditability)
```

### What stays fixed

Some things are not subject to self-improvement:

- **The safety model**: Agents cannot modify approval gates, containment rings, or safety invariants without human approval.
- **The audit system**: Logging is append-only. Agents cannot delete or modify historical records.
- **The rollback mechanism**: Git revert is always available. Agents cannot make irreversible changes to main.
- **Cost bounds**: Daily budget limits and per-request warnings cannot be increased by agents.

These are the system's constitutional constraints. They define the envelope within which self-improvement operates.

## The Cost Model

### Why cost matters

AI agent systems can burn through API budgets fast. A naive system that sends every request to the most capable model will spend $100 doing work that could cost $5. At scale, the difference is existential.

aiai treats cost as a first-class constraint, not a side effect.

### How routing works

```
┌─────────────────┐
│   Agent Request   │
│  + complexity tag │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Model Router    │────▶│  config/models.yaml │
│                  │     │  (tier mapping)     │
└────────┬─────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Select cheapest  │
│  available model  │
│  in target tier   │
└────────┬─────────┘
         │
         ├── success ──→ return result, log cost
         │
         └── failure ──→ try next model in tier
                              │
                              └── all failed ──→ escalate to next tier
```

### Cost tracking

Every API call is logged:
```json
{
  "timestamp": "2026-02-26T14:30:00Z",
  "task_id": "feat-openrouter-client",
  "agent": "builder-1",
  "model": "anthropic/claude-haiku-4-5",
  "complexity": "simple",
  "tokens_in": 1200,
  "tokens_out": 450,
  "cost_usd": 0.0008,
  "latency_ms": 340
}
```

Aggregated reports show:
- Cost per task type
- Cost per agent
- Cost per model
- Cost trends over time
- Budget utilization

### Escalation vs waste

The system defaults to the cheapest model. If a task fails or produces low-quality output, it escalates to a more capable model. This "try cheap first" approach means most requests are handled by inexpensive models, with expensive models reserved for the tasks that actually need them.

The alternative — always using the most capable model — is simpler but wasteful. A system that can't manage its own costs can't be trusted as infrastructure.

## The Coordination Model

### Single agent tasks

Most tasks don't need a team. A single agent receives the task, works on a branch, creates a PR. This is the default path and should handle the majority of work.

### Multi-agent teams

Complex tasks decompose into parallel subtasks. A team forms:

```
Task: "Build the OpenRouter client"
  │
  ├── Architect: designs the API, defines modules
  │
  ├── Builder 1: implements the client core
  ├── Builder 2: implements the fallback logic
  ├── Builder 3: implements cost tracking
  │     (parallel execution)
  │
  ├── Researcher: checks OpenRouter API docs for edge cases
  │
  └── Reviewer: reviews all output before PR
```

Teams coordinate through:
- **Shared task list**: Each agent sees what others are working on
- **Message passing**: Agents send messages when they need input or have findings
- **Branch convention**: All agents work on branches under the same prefix
- **Integration agent**: One agent is responsible for merging subtask outputs

### Team lifecycle

1. **Formation**: Task arrives, architect decomposes, agents spawn
2. **Execution**: Agents work in parallel on their subtasks
3. **Integration**: Results are merged, tests run, conflicts resolved
4. **Review**: A reviewer agent validates the combined output
5. **Submission**: PR created for human approval
6. **Dissolution**: Team disbands, agents available for next task

Teams are ephemeral. They form for a task and dissolve when it's done. There are no permanent teams.

## The Memory Model

### Three tiers of memory

**Tier 1: Session** (ephemeral)
- Conversation history within a single Claude Code session
- Current task context and working state
- Garbage collected when the session ends

**Tier 2: Project** (persistent, human-readable)
- `CLAUDE.md`: Conventions, safety rules, operating procedures
- `.claude/memory/`: Learned patterns, past decisions, project-specific knowledge
- `docs/`: Architecture, research, guides
- Git history: Every past decision and its rationale

**Tier 3: Evolutionary** (persistent, machine-readable)
- Performance baselines: How fast/cheap/accurate is each task type?
- Modification history: What was changed, what was the outcome?
- Pattern library: Reusable solutions indexed by problem type
- Model performance data: Which models work best for which task types?

### Memory as competitive advantage

An agent system without persistent memory repeats the same mistakes, rediscovers the same solutions, and never improves. Memory is what turns a stateless tool into a system that compounds.

The git repository is the ultimate memory: every decision, every experiment, every improvement is recorded with full context (commit messages, PR descriptions, code diffs). An agent that can read git history has access to the complete reasoning chain of the system's evolution.
