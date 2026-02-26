# aiai - Key Concepts

## Self-Improvement Loop

The fundamental mechanism of aiai. The system continuously evaluates and improves itself.

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ Execute  │────▶│ Evaluate  │────▶│ Improve  │
│ Task     │     │ Outcome   │     │ System   │
└─────────┘     └──────────┘     └─────────┘
     ▲                                  │
     └──────────────────────────────────┘
```

**What gets improved:**
- Agent prompts and system instructions
- Tool implementations and interfaces
- Coordination patterns between agents
- Task decomposition strategies
- Quality evaluation criteria

**What stays fixed (safety rails):**
- Human approval requirements for irreversible actions
- Audit logging of all changes
- Ability to rollback any modification
- Core safety constraints

## Agent Self-Mobilization

Unlike traditional orchestration where a central controller assigns all work, aiai agents **self-mobilize**:

1. **Task appears** — A goal is defined (by human or by another agent)
2. **Agents assess** — Available agents evaluate if they can contribute
3. **Team forms** — Agents with relevant capabilities organize into a team
4. **Roles emerge** — Agents claim tasks based on their strengths
5. **Work executes** — Parallel execution with coordination
6. **Team dissolves** — When the task is done, agents become available again

This is closer to how effective human teams work — people step up for what they're good at rather than being rigidly assigned.

## Capability Bootstrapping

The process of building increasingly powerful capabilities from simple foundations:

**Level 0**: Basic file I/O, shell commands, git operations
**Level 1**: Automated git workflows, structured documentation
**Level 2**: Agent coordination, team formation, task routing
**Level 3**: Self-evaluation, performance tracking, prompt optimization
**Level 4**: Autonomous tool creation, pattern discovery, architecture evolution

Each level builds on the previous. The system starts at Level 0 and works its way up.

## Evolutionary Pressure

Not all improvements are equal. aiai applies selection pressure:

- **Mutations** — Agents propose changes to the system
- **Testing** — Changes are validated against test suites
- **Fitness** — Metrics determine if a change is beneficial
  - Task completion speed
  - Code quality scores
  - Test pass rates
  - Resource efficiency
- **Selection** — Beneficial changes are kept, harmful ones are reverted
- **Propagation** — Successful patterns spread across the system

## Memory and Context

Agents need persistent memory to improve over time:

### Session Memory
- Current task context
- Conversation history
- Working state

### Project Memory
- CLAUDE.md conventions
- Past decisions and their rationale
- Known patterns and anti-patterns

### Evolutionary Memory
- Performance baselines
- History of modifications and their outcomes
- Successful prompts and configurations

## Agentic Version Control

Git isn't just for storing code — it's the system's journal:

- **Every agent action** produces a commit or is part of one
- **Branches** represent parallel experiments
- **Merges** represent successful integrations
- **Reverts** represent failed experiments
- **Tags** represent capability milestones

The git history IS the evolution history.

## Safety Model

### Containment Rings

```
Ring 0 (Unrestricted): Read files, search code, analyze
Ring 1 (Logged):       Write files, run tests, local operations
Ring 2 (Approved):     Git push, external API calls, deployments
Ring 3 (Human-only):   Delete repos, modify safety constraints, production access
```

### Invariants
1. No agent can modify the safety model without human approval
2. All changes are reversible (git revert)
3. Destructive operations require explicit human confirmation
4. The audit log is append-only and tamper-resistant

## Open Questions

These are active areas of exploration:

- **Evaluation problem**: How does a system accurately evaluate its own improvements?
- **Convergence**: Will the system converge on a local optimum or continue improving?
- **Alignment**: How do we ensure self-improvement stays aligned with human intent?
- **Complexity management**: As the system grows, how do we prevent entropy?
- **Multi-model coordination**: How do different model strengths best complement each other?
