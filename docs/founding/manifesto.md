# The aiai Manifesto

## The Problem

Software is written by humans, one keystroke at a time. AI can draft code faster than any human, but the system around it — review, merge, deploy, evaluate, iterate — still runs at human speed. The AI writes a function in 3 seconds, then waits 4 hours for someone to look at the PR.

This isn't an AI capability problem. It's a systems design problem.

The tools exist. The models exist. What's missing is the connective tissue: a system where AI agents don't just generate code on request, but *operate* — autonomously building, testing, committing, evaluating, and improving with minimal human friction.

## The Bet

aiai bets on three things:

### 1. Autonomy within structure beats supervised generation

A human reviewing every AI output is a bottleneck that scales linearly with the human's time. A system where AI operates freely within guardrails — committing to branches, running tests, spawning teams — and humans only intervene at merge points, scales with compute.

The key insight: most of what AI does doesn't need human review. File reads, test runs, branch management, research synthesis, tool creation — these are safe operations. Only the final integration into the mainline codebase needs a human eye. Gate what matters, automate everything else.

### 2. Cost optimization is a design constraint, not an afterthought

Running Opus for every API call is like using a freight truck to deliver a letter. Most tasks are simple. A rename, a test run, a status check. These should use the cheapest model that won't mess them up.

The system that treats model selection as an optimization problem — maximize quality, minimize cost — will outcompete the system that uses one model for everything. This is true whether you're a solo developer watching your API bill or a company processing millions of agent requests.

### 3. Self-improvement is the only durable advantage

Every other advantage is temporary. A better prompt gets copied. A better tool gets open-sourced. A better model gets released by a competitor. But a system that improves itself — that gets better at getting better — compounds in a way that's hard to replicate.

This doesn't require AGI or recursive superintelligence. It requires mundane, practical self-improvement: an agent that notices its test suite is slow and rewrites it. An agent that finds a better prompt and proposes the change. An agent that builds a tool because the existing ones don't cover the use case. Small improvements, continuously, adding up.

## The Architecture

aiai is built on three pillars:

**Git as the backbone.** Every agent action is a git operation. Branches are workspaces. Commits are atomic units of work. PRs are approval gates. The git log is the system's memory, its audit trail, and its evolution history. This isn't metaphorical — it's literal. The system's capability at any point in time is the state of its repository.

**Cost-optimized model routing.** Agents declare task complexity. A router picks the cheapest adequate model via OpenRouter. Haiku for trivial tasks, Sonnet for moderate ones, Opus for hard problems. The system tracks spend per task, per agent, per model. There's a daily budget. There are per-request warnings. Cost is a first-class concern.

**Gated self-improvement.** Agents can modify any part of the system — code, tools, prompts, workflows — but changes to how agents themselves operate (CLAUDE.md, agent configs, model routing, the evolution engine) require human approval via PR. This means the system improves freely within its operating envelope, but humans control the shape of that envelope.

## What aiai Is Not

**Not a chatbot.** aiai isn't a thing you talk to. It's a thing that works. It receives tasks, assembles teams, produces output, and improves.

**Not a research project.** The goal isn't to publish papers about self-improving AI. It's to build infrastructure that actually works and gets better over time. Research informs the design, but shipping is the priority.

**Not an AGI attempt.** Self-improvement here means practical optimization: better prompts, better tools, faster execution, lower cost. Not recursive superintelligence. The system operates within defined boundaries and improves within those boundaries.

**Not locked to one model.** The whole point of OpenRouter integration is model independence. Agents use whatever model is best for the task. When a better model ships, it slots into the routing config. No rewrite required.

## The Standard

aiai holds itself to an infrastructure standard, not a demo standard:

- Code that ships must work. Tests must pass. CI must be green.
- Every change is traceable to a commit, an agent, a task.
- Cost is tracked and bounded. No runaway API bills.
- Safety gates cannot be bypassed by agents.
- The system must be able to explain why it made any change (git log, PR descriptions, commit messages).
- Rollback is always possible. No irreversible damage without human approval.

## The Invitation

This is a public project. The code is open. The research is documented. The evolution is visible in the git history.

If you're building with AI agents, thinking about self-improving systems, or just curious about what happens when you let AI operate with real autonomy inside real guardrails — this is the place.

The system is early. The foundation is laid. What gets built next depends on what works.
