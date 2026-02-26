# Safety Model

How aiai prevents agents from causing damage while maximizing their autonomy.

## Design Philosophy

Safety in aiai is not about restricting what agents can do. It's about making the right things easy and the dangerous things require explicit approval. The system should feel frictionless for normal work and only intervene when an action could cause real damage.

The goal: **maximize the surface area of autonomous operation while maintaining a hard boundary around irreversible or high-impact actions.**

## The Containment Model

### Ring 0: Unrestricted

Actions that are always safe and need no approval.

| Action | Rationale |
|--------|-----------|
| Read any file | Reading is non-destructive |
| Search code (grep, glob) | Search is non-destructive |
| Analyze and reason | Computation is non-destructive |
| Research (web search, fetch) | Read-only external access |
| Read git history | Non-destructive |

### Ring 1: Logged

Actions that modify local state but are reversible.

| Action | Rationale |
|--------|-----------|
| Write/edit files on branches | Reversible via git |
| Run tests | Side effects contained to test env |
| Run linters and formatters | Non-destructive to logic |
| Create commits on branches | Reversible via git revert |
| Create branches | Cheap to create and delete |
| Spawn sub-agents | Contained within the session |

All Ring 1 actions are logged in git history. The agent can operate freely here because everything is traceable and reversible.

### Ring 2: Gated (requires human approval)

Actions that affect the mainline codebase or change how agents operate.

| Action | Rationale |
|--------|-----------|
| Merge PR to `main` | Changes what ships |
| Modify `CLAUDE.md` | Changes agent behavior |
| Modify `.claude/agents/` | Changes agent definitions |
| Modify `config/models.yaml` | Changes cost and model routing |
| Modify `src/evolution/` | Changes the self-improvement engine |
| Delete branches with unmerged work | Potential data loss |
| External API calls (non-read) | Visible outside the system |

Ring 2 is the core safety boundary. These actions go through GitHub PRs, which require human review and approval. This is where the "auto with gates" model is enforced.

### Ring 3: Blocked

Actions that agents cannot perform, period. Enforced at the tool level by `scripts/agent-git.sh`.

| Action | Rationale |
|--------|-----------|
| Force push to `main` | Destroys shared history |
| Delete the repository | Irreversible data loss |
| Disable CI checks | Removes quality gates |
| Commit secrets/credentials | Security vulnerability |
| Modify the safety model without PR | Bypasses human oversight |
| Bypass approval gates | Undermines the whole model |

Ring 3 is not a convention — it's enforced by tooling. `agent-git.sh` actively blocks these operations.

## Enforcement Layers

Safety is enforced at multiple levels, so no single failure compromises the system:

### Layer 1: Convention (CLAUDE.md)
Agents are instructed in their system prompt to follow safety rules. This is the weakest layer — it relies on model compliance — but it covers the majority of cases.

### Layer 2: Tooling (agent-git.sh)
The git wrapper script actively blocks dangerous operations:
- Refuses force-push to protected branches
- Scans staged changes for secrets (API keys, tokens, passwords)
- Requires explicit commit messages on protected branches
- Only deletes branches that are fully merged

### Layer 3: Platform (GitHub branch protection)
GitHub enforces branch protection rules on `main`:
- Requires PR review before merge
- Requires CI to pass
- No direct pushes
- No force pushes

### Layer 4: Observability (git log + cost tracking)
Every action is logged. Anomalies are detectable after the fact:
- Unexpected cost spikes
- Unusual patterns of file modification
- Agents working on files outside their task scope
- High error rates or retry loops

## Self-Modification Safety

The most sensitive aspect of aiai is that agents can propose changes to their own operating rules. This is also the most valuable aspect — it's what enables self-improvement. The safety model balances these concerns:

### What agents can self-modify (with PR approval)

- **CLAUDE.md**: Agent conventions, coding standards, workflow instructions
- **Agent configs**: Agent type definitions, role descriptions, capabilities
- **Model routing**: Which models handle which tasks, cost thresholds
- **Evolution engine**: How self-improvement is evaluated and applied

### How self-modification works

1. Agent identifies a potential improvement (e.g., "the commit message format should include cost data")
2. Agent creates a branch: `evolve/add-cost-to-commits`
3. Agent implements the change
4. Agent creates a PR with:
   - What changed
   - Why it's an improvement
   - Before/after comparison or metrics
   - Risk assessment
5. Human reviews the PR
6. If approved, the change merges to `main` and takes effect for all future agent sessions

### Self-modification invariants

1. **No self-modification can remove approval gates.** An agent cannot propose a change that would allow future agents to bypass PR approval for self-modifications.

2. **No self-modification can increase cost bounds.** An agent cannot raise the daily budget or per-request warning threshold without human initiative.

3. **No self-modification can weaken safety.** Changes to containment rings, blocked operations, or security scanning must be additive (adding safety) not subtractive (removing safety).

4. **All self-modifications are reversible.** Any change can be reverted via git revert, restoring the previous behavior.

5. **Self-modifications must be justified.** PRs for self-modification must include rationale. "I want to" is not sufficient. "This improves X metric by Y%" or "This prevents Z failure mode" is.

## Failure Modes and Responses

### Agent ignores safety conventions
**Detection**: CI checks, reviewer agents, human PR review
**Response**: PR rejected, agent instructed to fix

### Agent finds a way around tooling blocks
**Detection**: Git log analysis, anomaly detection
**Response**: Fix the tooling gap, revert the unauthorized action

### Agent proposes harmful self-modification
**Detection**: Human PR review (required for all self-modifications)
**Response**: PR rejected with feedback

### Cost runaway
**Detection**: Per-request warnings, daily budget enforcement
**Response**: Requests blocked when budget is exceeded, human notified

### Agent stuck in retry loop
**Detection**: Cost tracking shows repeated failed requests
**Response**: Escalation to more capable model, or task marked as blocked for human review

## Principles

1. **Defense in depth**: No single layer failure compromises safety. Convention + tooling + platform + observability.

2. **Fail safe, not fail secure**: When in doubt, the system blocks the action rather than allowing it. A blocked legitimate action costs time. A permitted illegitimate action costs trust.

3. **Transparency over restriction**: Rather than preventing agents from seeing information, focus on controlling what they can do with it. Read access is free; write access is gated.

4. **Reversibility as a safety property**: If an action is reversible, it's less dangerous. Git provides reversibility for all code changes. This is why git is the backbone.

5. **Human time is the scarce resource**: Safety mechanisms should minimize the amount of human attention required, not maximize it. Gate the important things, automate everything else.
