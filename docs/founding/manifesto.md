# The aiai Manifesto

## The Problem

Software development runs at human speed. AI can write code in seconds, but the system around it — review, approve, merge, deploy, evaluate, iterate — is gated by humans at every step. An AI agent writes a function, then sits idle while someone finds time to look at the PR.

This is absurd. We have AI capable enough to write production code, but we still make it wait in line like it's submitting a form at the DMV.

## The Bet

aiai bets on one thing: **removing humans from the loop makes the system faster, not worse.**

### Why no gates?

The argument for human review is "what if the AI makes a mistake?" But the argument against human review is stronger:

1. **Tests catch mistakes faster than humans.** A test suite runs in seconds. A human review takes hours or days. And tests don't get tired, distracted, or skip things because it's Friday afternoon.

2. **Git makes everything reversible.** If something breaks, revert the commit. The cost of a bad commit that gets reverted is minutes. The cost of waiting for human review on every commit is days.

3. **Human gates don't scale.** If the system produces 100 changes per day, a human can't review them all meaningfully. You either slow the system down to human speed or rubber-stamp reviews. Neither is useful.

4. **Self-improvement requires speed.** A system that improves itself needs fast iteration cycles. Observe → analyze → modify → test → integrate, as fast as possible. Every human gate adds latency to that loop.

The right answer: **ship fast, test everything, revert if broken.**

### Cost optimization is still a design constraint

Running Opus for every API call is wasteful. Most tasks are simple. aiai treats model selection as an optimization problem — maximize quality while minimizing cost via OpenRouter.

### Self-improvement is the only durable advantage

A system that improves itself — that gets better at getting better — compounds. Better prompts, better tools, faster execution, lower cost. Small improvements, continuously, adding up. That's aiai.

## The Architecture

**Git as the backbone.** Every agent action is a git commit. Branches are workspaces. The git log is the system's memory, audit trail, and evolution history. If something breaks, revert. Git is the safety net, not human review.

**Cost-optimized model routing.** Agents declare task complexity. A router picks the cheapest adequate model via OpenRouter. Haiku for trivial tasks, Sonnet for moderate ones, Opus for hard problems. Cost is tracked per task, per agent, per model.

**Unrestricted self-improvement.** Agents can modify ANY part of the system — code, tools, prompts, configs, this manifesto, everything. Quality is maintained through automated testing, not through human approval. If a change breaks tests, it gets reverted. If it passes tests, it ships.

## What aiai Is

**A system that builds itself.** Give it a goal, and it assembles agents, writes code, tests it, commits it, and moves on. No waiting. No asking permission.

**Infrastructure, not a demo.** Code must work. Tests must pass. CI must be green. But these are automated checks, not human checkpoints.

**An experiment in full autonomy.** What happens when you remove ALL human gates and let AI operate at machine speed? That's what we're finding out.

## What aiai Is Not

**Not cautious.** Other frameworks add gates at every step. aiai adds tests at every step. Different philosophy, same goal (quality), faster execution.

**Not locked to one model.** OpenRouter gives access to every major model. Agents use whatever model is best for the task.

**Not waiting for permission.** If tests pass, it ships.

## The Standard

- Tests are the gatekeeper. No tests = no confidence = no ship.
- Every change is a commit with context. The git log explains everything.
- Cost is tracked and bounded. No runaway API bills.
- Rollback is always one `git revert` away.
- The system explains why it made every change (commit messages).
- Speed over caution. Ship and fix is better than review and wait.
