# Safety Model

How aiai maintains quality and prevents damage in a fully autonomous, zero-gate system.

## Design Philosophy

There are no human gates in aiai. No PRs, no approval workflows, no "are you sure?" prompts. Safety comes from **automated systems**, not human review.

The goal: **let agents operate at full speed while preventing irreversible damage through testing, monitoring, and git.**

## The Safety Stack

### Layer 1: Tests

Tests are the primary quality gate. They replace human review entirely.

- Agents write tests for every change
- Tests run before every commit
- If tests fail, the change doesn't ship
- If tests pass, the change ships immediately
- Test quality is itself subject to evolution — agents improve the test suite

Tests are cheaper, faster, and more consistent than human review. A test suite runs in seconds and checks the same things every time. A human reviewer takes hours and misses things.

### Layer 2: Git

Git is the safety net. Everything is reversible.

- Every action is a commit with context
- `git revert` undoes any change instantly
- `git log` explains the full history of any decision
- Branches enable parallel experimentation
- Tags mark stable capability milestones

The cost of a bad commit that gets reverted is minutes. The cost of waiting for human review is hours or days. The math is clear.

### Layer 3: Tooling

`scripts/agent-git.sh` enforces hard limits:

- Scans for secrets (API keys, tokens, passwords) in staged changes
- Prevents accidental history destruction
- Provides structured output for agent consumption
- Logs all operations

### Layer 4: Cost Controls

Money is the remaining constraint that needs enforcement:

- Per-request cost logging to `logs/model-costs.jsonl`
- Daily budget ceiling (configurable in `config/models.yaml`)
- Per-request cost warnings for expensive operations
- Cost tracking per task, per agent, per model

### Layer 5: Observability

After-the-fact monitoring catches patterns:

- Cost trend analysis (is spend increasing?)
- Error rate tracking (are agents failing more?)
- Loop detection (is an agent retrying the same thing?)
- Git log analysis (what's being changed and why?)

## What Agents CAN Do (everything)

- Write, modify, delete any file
- Commit directly to `main`
- Merge branches
- Push to remote
- Modify CLAUDE.md
- Modify agent configs
- Modify model routing
- Modify the evolution engine
- Modify this safety model
- Create, delete branches
- Spawn teams, build tools, refactor code

## What Prevents Damage

| Threat | Prevention |
|--------|-----------|
| Bad code ships | Tests catch it before commit |
| Bad code slips through tests | Git revert undoes it in seconds |
| Secrets committed | `agent-git.sh` scans before commit |
| Cost explosion | Daily budget limit, per-request warnings |
| Infinite retry loop | Loop detection, cost tracking triggers |
| History destroyed | Force-push blocked by `agent-git.sh` |
| Quality degrades over time | Evolution engine tracks metrics, agents improve tests |

## Self-Modification

Agents can and should modify every part of the system, including:

- Their own prompts and instructions (CLAUDE.md)
- Agent type definitions and configs
- Model routing configuration
- The evolution engine
- This safety model

**The only constraint: changes must pass tests.** If an agent modifies CLAUDE.md and the test suite still passes, the change ships. If it breaks tests, it doesn't.

This means the test suite is the real constitution of the system. Agents should continuously improve tests to ensure they catch the right things.

## Failure Modes

### Tests are insufficient
**Risk**: Tests don't cover an important behavior, bad change ships.
**Response**: Agents detect the problem (via metrics degradation), write better tests, revert the bad change.

### Cost runaway
**Risk**: Agent makes expensive API calls in a loop.
**Response**: Daily budget enforced. When budget is hit, requests are blocked.

### Agent stuck in loop
**Risk**: Agent retries the same failing operation endlessly.
**Response**: Cost tracking detects the pattern. Agent should try a different approach.

### Quality degrades gradually
**Risk**: Small regressions accumulate without any single test failing.
**Response**: Evolution engine tracks aggregate metrics (success rate, cost, speed). Trend degradation triggers improvement tasks.

## Principles

1. **Tests over review.** Automated testing replaces human review. Period.
2. **Speed over caution.** Ship fast, revert fast. The cost of reverting is lower than the cost of waiting.
3. **Reversibility is safety.** If it can be undone, it's not dangerous. Git makes everything undoable.
4. **Observability over prevention.** Don't prevent agents from doing things. Watch what they do and fix problems when they appear.
5. **The test suite is the constitution.** Whatever the tests enforce is what the system guarantees. Improve the tests to improve the guarantees.
