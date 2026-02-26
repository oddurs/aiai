# Evolution Engine

The technical design of aiai's self-improvement system.

## What This Document Covers

The evolution engine is the mechanism by which aiai improves itself over time. This document defines how improvements are identified, proposed, validated, and integrated. It's the most important — and most dangerous — subsystem in aiai.

## The Improvement Cycle

```
                    ┌─────────────┐
            ┌──────▶│   Observe    │
            │       │  (metrics)   │
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            │       │   Analyze    │
            │       │  (patterns)  │
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            │       │  Hypothesize │
            │       │ (improvement)│
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            │       │  Implement   │
            │       │  (on branch) │
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            │       │   Validate   │
            │       │  (test+diff) │
            │       └──────┬──────┘
            │              │
            │              ▼
            │       ┌─────────────┐
            │       │    Gate      │
            │       │  (PR review) │
            │       └──────┬──────┘
            │              │
            │         merge │ reject
            │              │    │
            │              ▼    └──→ learn from rejection
            │       ┌─────────────┐
            └───────│   Measure    │
                    │ (new baseline)│
                    └─────────────┘
```

## Observation: What Gets Measured

The evolution engine needs data to identify improvements. Every task execution generates metrics:

### Execution Metrics
```python
@dataclass
class TaskMetrics:
    task_id: str
    task_type: str           # "build", "research", "fix", "evolve", "review"
    complexity: str          # "trivial", "simple", "medium", "complex", "critical"

    # Performance
    duration_seconds: float
    agent_count: int
    retry_count: int

    # Cost
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    models_used: list[str]

    # Outcome
    success: bool
    tests_passed: int
    tests_failed: int
    files_changed: int
    lines_added: int
    lines_removed: int

    # Quality signals
    ci_passed: bool
    review_requested_changes: bool
    human_approved: bool
```

### Aggregate Metrics (computed over time windows)
- **Task success rate**: % of tasks that succeed on first attempt
- **Cost per task type**: Average cost for build/research/fix/evolve tasks
- **Model efficiency**: Quality per dollar for each model on each task type
- **Time to merge**: How long from task start to PR merge
- **Retry rate**: How often agents need to retry failed operations
- **Self-improvement acceptance rate**: % of evolve PRs that humans approve

## Analysis: Identifying Opportunities

The evolution engine periodically analyzes metrics to find improvement opportunities.

### Pattern: Cost Outliers
```
Observation: "build" tasks using Opus cost 10x more than Sonnet,
             but success rate is only 5% higher.
Hypothesis:  Sonnet is adequate for most build tasks.
Action:      Propose routing change: build tasks default to Sonnet,
             escalate to Opus only on failure.
```

### Pattern: Repeated Failures
```
Observation: test_router.py fails 40% of the time after agent edits.
Hypothesis:  Agents don't understand the test fixtures.
Action:      Add documentation to the test file explaining fixtures,
             or add a pre-commit check.
```

### Pattern: Slow Tasks
```
Observation: "research" tasks take 3x longer than they should.
Hypothesis:  Agents are searching too broadly.
Action:      Propose a research template that structures searches
             (define scope → search → synthesize → cite).
```

### Pattern: Tool Gaps
```
Observation: Agents manually parse JSON in 15 different tasks.
Hypothesis:  A JSON utility tool would save time and reduce errors.
Action:      Create src/tools/json_utils.py with common operations.
```

### Pattern: Prompt Inefficiency
```
Observation: Agents frequently misunderstand the commit message format.
Hypothesis:  The CLAUDE.md instructions are ambiguous.
Action:      Rewrite the commit message section with explicit examples.
```

## Hypothesis Generation

When an improvement opportunity is identified, the evolution engine generates a concrete hypothesis:

```
Hypothesis: {
  id: "hyp-2026-02-26-001",
  observation: "Build tasks using nano-tier models fail 60% of the time",
  proposed_change: "Set minimum tier for build tasks to 'fast'",
  expected_outcome: "Build task success rate increases from 40% to 85%+",
  risk: "Cost per build task increases ~5x (from $0.001 to $0.005)",
  reversible: true,
  gate_required: true,  // changes model routing config
  files_affected: ["config/models.yaml"]
}
```

Hypotheses are stored in `src/evolution/hypotheses/` as structured data. This creates an audit trail of what the system considered improving and why.

## Implementation

Each hypothesis becomes a branch:

```
evolve/hyp-2026-02-26-001-build-min-tier
```

The evolver agent:
1. Creates the branch
2. Implements the change
3. Writes or updates tests
4. Runs the test suite
5. Generates a before/after comparison
6. Creates a PR with full context

### PR Format for Evolution Changes

```markdown
## Evolution: [hypothesis ID]

### Observation
Build tasks using nano-tier models fail 60% of the time.

### Change
Set minimum tier for build tasks to `fast` in config/models.yaml.

### Expected Outcome
Build task success rate: 40% → 85%+
Cost per build task: $0.001 → $0.005

### Evidence
[metrics, logs, specific failure examples]

### Risk Assessment
- Cost increase is bounded and small in absolute terms
- Change is reversible (revert the config line)
- No impact on non-build tasks

### Rollback Plan
git revert [commit hash]
```

## Validation

Before a PR is created, the evolution engine validates the change:

### Automated Checks
1. **Test suite passes**: All existing tests must still pass
2. **No safety regressions**: Safety invariants are verified
3. **Cost impact estimated**: The expected cost change is calculated
4. **Scope check**: The change only modifies files relevant to the hypothesis

### Comparison Metrics
When possible, the system runs the same task with and without the change:
- Execute a sample of recent tasks with the current configuration
- Execute the same tasks with the proposed change
- Compare success rate, cost, and quality

This isn't always possible (some changes are structural), but when it is, it provides the strongest evidence.

## Gating

### Changes that require human approval
- Any modification to CLAUDE.md
- Any modification to agent configs
- Any modification to model routing config
- Any modification to the evolution engine itself
- Any change that increases cost bounds

### Changes that can auto-merge (after CI)
- New tools in `src/tools/`
- Test improvements
- Documentation updates
- Non-config code changes

Even auto-merge changes go through PRs for auditability. The difference is whether a human must explicitly approve.

## Measuring Outcomes

After a change merges, the evolution engine tracks whether the hypothesis was correct:

```
Hypothesis: "Build task success rate increases from 40% to 85%+"
Measurement window: 7 days post-merge
Actual outcome: Success rate went from 40% to 82%
Verdict: Hypothesis partially confirmed (close to target)
```

Outcomes are stored alongside hypotheses, creating a feedback loop:
- Hypotheses that were confirmed → increases confidence in similar future hypotheses
- Hypotheses that were wrong → informs future analysis (what was missed?)
- Rejected PRs → understanding what humans don't want changed

## Evolution of the Evolution Engine

The evolution engine can improve itself, but with the highest level of scrutiny:

1. Any change to `src/evolution/` requires human PR approval
2. Changes must include comprehensive justification
3. The previous version's metrics serve as the baseline
4. Rollback plan is mandatory

This is the system's most sensitive surface. A bug in the evolution engine could cause cascading bad modifications. Changes here are treated with corresponding care.

## Bootstrapping

The evolution engine doesn't exist yet. It will be built incrementally:

### Phase 1: Manual metrics
Agents log metrics to files. Humans review and identify patterns manually.

### Phase 2: Automated observation
Python scripts aggregate metrics and generate reports. Patterns are surfaced automatically.

### Phase 3: Hypothesis generation
Agents analyze reports and propose improvements. Humans review all proposals.

### Phase 4: Automated validation
The system can run before/after comparisons. Humans still approve all changes.

### Phase 5: Selective auto-merge
Low-risk improvements (new tools, test improvements) can auto-merge after CI. High-risk improvements still require human approval.

Each phase builds on the previous one. The system earns more autonomy by demonstrating reliability at each level.
