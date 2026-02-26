# aiai - Vision

> Self-improving AI infrastructure. Agents build, test, and evolve their own codebase.

## Core Thesis

The bottleneck in AI-powered development isn't model capability — it's the human iteration loop. Review a PR, merge, wait, repeat. aiai removes that bottleneck for everything except the decisions that actually matter.

**aiai** is infrastructure where AI agents autonomously:

1. **Build** — Write, test, and ship code without waiting for humans
2. **Improve** — Analyze what worked, modify their own tooling and prompts
3. **Coordinate** — Form teams, divide work, merge results
4. **Evolve** — Each cycle produces a more capable version of the system

This isn't a toy or a research demo. It's intended to be reliable infrastructure that compounds in capability over time.

## How It Actually Runs

**Claude Code** is the primary execution environment. Agents are Claude Code sessions that use the Task tool for team coordination, Bash for execution, and the standard Claude toolchain for file operations.

**OpenRouter** provides model access. Instead of being locked to one provider, agents route requests through OpenRouter to pick the best model for each subtask:
- Expensive, capable models (Opus, GPT-4, DeepSeek-R1) for architecture, complex reasoning, hard bugs
- Fast, cheap models (Haiku, Flash, small open-source) for routine operations, simple edits, formatting
- Model selection is automatic and cost-optimized — agents declare task complexity, the router picks the model

**Python** is the implementation language. One language, one ecosystem, no polyglot overhead. Bash for scripts.

## What Makes aiai Different

Most agent frameworks are orchestration layers that humans configure. aiai is different in two ways:

1. **The framework improves itself.** Agents can modify their own prompts, tools, coordination patterns, and evaluation criteria. The system's git history is its evolution history.

2. **Cost-aware model routing.** Not every task needs the most expensive model. aiai treats model selection as an optimization problem — maximize quality while minimizing cost.

| Traditional Agent Framework | aiai |
|---|---|
| Humans design orchestration | Agents design their own orchestration |
| Fixed tool set | Agents create and improve tools |
| One model for everything | Cost-optimized routing across models |
| Improvement requires human PRs | Autonomous within approval gates |

## Autonomy Model

**Autonomous within guardrails.** Agents operate freely for most tasks, with human approval required at specific gates:

**Agents do freely:**
- Write code, run tests, commit to feature branches
- Create and manage branches
- Coordinate in teams, spawn sub-agents
- Build new tools and scripts
- Research, analyze, synthesize

**Requires human approval:**
- Merging PRs to `main`
- Modifying CLAUDE.md, agent configs, or the self-improvement system
- Any operation that changes how agents themselves operate

This keeps the system productive while ensuring humans retain control over what actually ships and how agents behave.

## Guiding Principles

1. **Ship working code** — Not prototypes, not demos. Code that runs and does what it says.
2. **Audit everything** — Every agent action is a git operation. Full traceability.
3. **Optimize cost** — Use the cheapest model that can do the job well. Expensive models for hard problems only.
4. **Fail safe** — Agents can't merge to main or modify their own rules without approval.
5. **Compound** — Every improvement makes the next improvement easier.
