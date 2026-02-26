# aiai - Vision

> **AI-made AI.** A self-improving system where AI models orchestrate, build, and enhance themselves.

## Core Thesis

The most powerful AI systems won't be hand-crafted by humans line-by-line. They'll be **grown** — bootstrapped by AI agents that coordinate, write code, test it, and iteratively improve their own capabilities.

**aiai** is that system. It's an open framework where AI models (Opus, Sonnet, Codex, or any capable model) self-mobilize within a structured environment to:

1. **Build** — Write, test, and deploy code autonomously
2. **Improve** — Analyze their own performance and modify their own tooling, prompts, and workflows
3. **Coordinate** — Form agent teams that divide work, share context, and merge results
4. **Evolve** — Each iteration produces a more capable version of the system itself

## Philosophy

### Self-Mobilization Over Orchestration

Traditional AI agent frameworks have a human-designed orchestrator that rigidly controls agent behavior. aiai inverts this: agents decide how to organize themselves based on the task at hand. The structure is emergent, not prescribed.

### Recursive Self-Improvement

The system's core loop:

```
observe → analyze → modify → test → integrate → repeat
```

Agents can modify:
- Their own prompts and system instructions
- The tools and scripts available to them
- The coordination patterns they use
- The evaluation criteria they optimize for

### Structural Inspiration

Drawing from projects like OpenClaw and multi-agent frameworks (CrewAI, MetaGPT, AutoGPT), aiai takes the best patterns:
- **Agent specialization** — Different models/configs for different roles
- **Shared memory** — Persistent context that survives across sessions
- **Tool ecosystems** — Agents build and share tools with each other
- **Version-controlled evolution** — Every change is tracked, every improvement is auditable

But aiai goes further by making the framework itself the target of improvement.

## What Makes aiai Different

| Traditional Agent Framework | aiai |
|---|---|
| Humans design the orchestration | Agents design their own orchestration |
| Fixed tool set | Agents create and improve tools |
| Static prompts | Agents evolve their own prompts |
| Framework is separate from output | Framework IS the output |
| Improvement requires human PRs | Improvement is autonomous and continuous |

## North Star

A system where you point AI at a problem, and it:
1. Assembles the right team of agents
2. Builds whatever tools it needs
3. Solves the problem
4. Learns from the experience
5. Is measurably better at the next problem

And does all of this while maintaining safety, auditability, and human oversight at critical decision points.

## Guiding Principles

1. **Ship over perfect** — Working code beats perfect architecture
2. **Audit everything** — Every AI action is logged and version-controlled
3. **Fail safe** — Agents can't do irreversible damage without approval
4. **Open by default** — Public repo, open research, community-driven
5. **Eat your own dogfood** — aiai should be built and improved by aiai
