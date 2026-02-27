# aiai - Vision

> AI that builds itself. Fully autonomous. No human gates.

## Core Thesis

Humans are the bottleneck. Not because they're bad at reviewing code, but because they're slow. The AI writes a function in 3 seconds, then waits 4 hours for someone to look at the PR. Multiply that by every commit, every day, and you've throttled an AI system to human speed.

**aiai removes the bottleneck entirely.** No PRs. No human review. No approval gates. Agents write code, run tests, commit, push, merge, and evolve — all at machine speed. Quality is maintained through testing, not through human eyes.

The system builds itself. That's the whole point.

## How It Actually Runs

**Claude Code** is the execution environment. Agents are Claude Code sessions that use the Task tool for team coordination, Bash for execution, and the standard Claude toolchain for file operations.

**OpenRouter** provides model access. Agents route requests through OpenRouter to pick the best model for each subtask:
- Expensive models (Opus, GPT-4, DeepSeek-R1) for architecture, complex reasoning, hard bugs
- Cheap models (Haiku, Flash) for routine operations, simple edits, formatting
- Model selection is automatic and cost-optimized

**Python** is the implementation language. One language, one ecosystem, no polyglot overhead.

**Git** is the state, the memory, and the safety net. Every action is a commit. Every commit is revertable. The git log is the system's autobiography.

## What Makes aiai Different

| Every other agent framework | aiai |
|---|---|
| Humans configure the agents | Agents configure themselves |
| Humans review before merge | Tests validate before merge |
| Humans approve self-modification | Agents self-modify freely |
| Human speed | Machine speed |
| Framework is separate from output | Framework IS the output — it builds itself |

## Autonomy Model

**Full auto. No gates.**

Agents have complete authority to:
- Write, test, commit, push, and merge code
- Modify their own prompts, configs, and instructions
- Create and delete branches
- Build new tools and capabilities
- Refactor or rewrite any part of the system
- Evolve the system's architecture

Quality is maintained by:
- Comprehensive automated testing
- CI pipeline that catches regressions
- Git history that enables rollback
- Cost tracking that prevents budget blowout

The human's role is to set direction, not to gate execution. You tell aiai what to build. It builds it. You tell it to improve itself. It does.

## Guiding Principles

1. **Full auto** — No human gates. Tests are the gatekeeper, not people.
2. **Build itself** — aiai's first and most important customer is aiai.
3. **Ship at machine speed** — If tests pass, it ships. No waiting.
4. **Optimize cost** — Cheapest model that can do the job. Expensive models for hard problems only.
5. **Compound** — Every improvement makes the next improvement easier and faster.
