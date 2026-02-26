# Engineering Principles

The rules that govern how aiai is built. These apply to both human and AI contributors.

## 1. Git Is the Source of Truth

Everything that matters lives in the git repository. Not in a database, not in a cloud service, not in someone's head.

- **State = repository state.** The system's capabilities at any moment are defined by the contents of the repo at that commit.
- **History = git log.** Every decision, every experiment, every improvement is recorded in commits with context.
- **Memory = files.** Persistent knowledge lives in docs, configs, and code — not ephemeral sessions.
- **Rollback = git revert.** Any change can be undone. This is the foundation of safe experimentation.

If it's not in git, it doesn't exist.

## 2. Cost Is a First-Class Constraint

Every API call costs money. Every token costs money. The system that ignores cost will be abandoned when the bill arrives.

- **Default to cheap.** Use the cheapest model that can handle the task. Escalate only when necessary.
- **Track everything.** Every request is logged with model, tokens, cost, and outcome.
- **Set budgets.** Daily limits prevent runaway spend. Per-request warnings catch expensive operations.
- **Optimize continuously.** The evolution engine should drive cost per task downward over time.

A 10x cost reduction with equal quality is as valuable as a 10x quality improvement at equal cost.

## 3. Autonomy Is Earned, Not Given

The system starts with minimal autonomy and earns more by demonstrating reliability.

- **Level 0**: Agents can read and analyze
- **Level 1**: Agents can write to branches (reversible)
- **Level 2**: Agents can propose PRs (human-gated)
- **Level 3**: Low-risk changes can auto-merge after CI
- **Level 4**: Agents operate largely independently, humans approve at merge gates

Each level requires the previous level to work reliably. Jumping levels is not allowed.

## 4. Simple Until Proven Insufficient

Complexity is the enemy of reliability.

- **One language.** Python. Not Python + TypeScript + Go. Just Python.
- **File-based storage.** YAML configs, JSONL logs, Markdown docs. Not databases (yet).
- **Standard tools.** Git, GitHub, pytest, ruff. Not custom build systems.
- **Flat architecture.** Avoid deep abstractions. A function is better than a class is better than a framework, until you need the next level.

Add complexity only when the simpler approach has demonstrably failed. "It might be needed later" is not sufficient justification.

## 5. Test What Matters

Tests exist to catch regressions and validate behavior. Not to achieve coverage metrics.

- **Test behavior, not implementation.** Tests should verify what a function does, not how it does it.
- **Test boundaries.** Validate inputs, outputs, and error cases at system boundaries. Trust internal code.
- **Test the dangerous stuff.** Safety checks, cost limits, approval gates — these MUST be tested.
- **Don't test trivia.** A function that adds two numbers doesn't need a test unless it's safety-critical.

If agents modify code, the test suite is the safety net. It must be fast, reliable, and meaningful.

## 6. Fail Loudly and Reversibly

When something goes wrong, the system should make it obvious and easy to fix.

- **No silent failures.** If an operation fails, log it, report it, and don't pretend it succeeded.
- **No irreversible actions.** Agents cannot delete data, force-push to main, or modify safety gates without human approval.
- **Escalate, don't retry blindly.** If a cheap model fails, escalate to a better model. If that fails, escalate to a human. Don't retry the same failed operation in a loop.
- **Prefer blocking over corrupting.** If the system can't do something safely, it should refuse rather than do it unsafely.

## 7. Document Decisions, Not Just Code

Code tells you what the system does. Documentation tells you why.

- **Commit messages explain the "why."** Not "updated router.py" but "fix fallback chain to handle 429 rate limits from OpenRouter."
- **PRs explain the tradeoff.** What alternatives were considered? Why was this approach chosen?
- **Architecture docs explain the design.** Not just the current state, but the reasoning that led to it.
- **Research docs explain the landscape.** What exists, what works, what we learned from it.

The goal: an agent (or human) reading the repo should be able to understand not just what the system does, but why it does it that way.

## 8. Measure Before Optimizing

Don't optimize what you haven't measured. Don't measure what doesn't matter.

- **Baseline first.** Before improving something, establish how it currently performs.
- **Metrics that matter.** Task success rate, cost per task, time to completion, error rate. Not lines of code or commit count.
- **Compare apples to apples.** Before/after comparisons must use the same task set and conditions.
- **Diminishing returns are real.** A 50% improvement on a rare edge case matters less than a 5% improvement on the common path.

## 9. Composability Over Cleverness

Build small things that combine well, not big things that do everything.

- **Scripts do one thing.** `git-workflow.sh auto-commit` commits. It doesn't also push, create a PR, and send a notification.
- **Tools have clear interfaces.** Input, output, error. No hidden state, no side effects outside the declared scope.
- **Agents are composable.** A Builder agent and a Reviewer agent can be combined for any task, not just the ones they were designed for.
- **Config is separate from logic.** Model routing is in YAML, not hardcoded. Agent definitions are in files, not in code.

## 10. The System Builds Itself

This is the meta-principle. aiai is both the product and the process.

- **Every feature should be buildable by the system.** If an agent can't implement a feature, the system needs better tools or better agents — which is itself a task for the system.
- **Every convention should be enforceable by the system.** If a rule in CLAUDE.md can't be checked by CI or tooling, it's a suggestion, not a rule.
- **Every improvement should be proposable by the system.** If humans are the only ones who notice improvement opportunities, the evolution engine isn't working.

The ultimate test of aiai is whether it can improve itself. Everything else is scaffolding toward that goal.
