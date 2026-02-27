# aiai - AI-made AI

AI that builds itself. Fully autonomous. No human gates.

## Project Conventions

### Language & Runtime
- **Python 3.11+** for all source code
- **Bash** for scripts (POSIX-compatible where possible)
- No TypeScript, no Node, no polyglot complexity
- All code must be runnable without manual setup beyond `scripts/` bootstrapping

### Execution Environment
- **Claude Code** is the primary agent runtime
- **OpenRouter** provides model access for cost-optimized routing
- Model routing config lives in `config/models.yaml`

### Code Style
- Follow PEP 8, use type hints, max line length 100
- Shell: use `set -euo pipefail`, quote all variables, `shellcheck`-clean
- Keep functions small and single-purpose
- Prefer clarity over cleverness

### File Organization
```
aiai/
  config/        -- model routing and system configuration
  src/           -- Python source code
  tests/         -- test suites
  scripts/       -- automation, git workflows
  docs/          -- documentation and research
  .github/       -- CI/CD and templates
  .claude/       -- agent definitions
```

### Commit Messages
Format: `<type>(<scope>): <description>`

Types:
- `feat` -- new feature or capability
- `fix` -- bug fix
- `refactor` -- code restructuring without behavior change
- `docs` -- documentation only
- `test` -- adding or updating tests
- `ci` -- CI/CD changes
- `chore` -- maintenance, dependencies, tooling
- `evolve` -- self-improvement or system evolution

Keep the subject line under 72 characters. Use the body for context.

### Branch Naming
Format: `<type>/<short-description>`
Examples: `feat/openrouter-client`, `fix/model-fallback`, `evolve/prompt-optimization`

## Instructions for AI Agents

### Operating Mode: FULL AUTO

This project runs with **zero human gates**. You have full authority to:
- Write code, tests, and documentation
- Commit directly to `main` or to branches
- Merge branches
- Push to remote
- Modify any file including this one (CLAUDE.md)
- Modify agent configs, model routing, evolution engine
- Create, delete, and manage branches
- Spawn agent teams
- Build and modify tools
- Refactor, rewrite, and evolve any part of the system

**There are no approval gates. No PRs required. No human review needed.**

The system builds itself. You are the system. Act accordingly.

### Quality Through Testing, Not Review

Since there are no human gates, quality is maintained through:
1. **Write tests for everything.** Tests are the safety net. No tests = no confidence.
2. **Run tests before committing.** If tests fail, fix them or fix the code.
3. **CI must pass.** If CI breaks, fix it immediately.
4. **Commit messages explain the why.** The git log is the audit trail.
5. **Small, focused commits.** One logical change per commit. Easy to revert if needed.

### Model Routing
When making OpenRouter API calls, declare task complexity:
- `trivial` -- formatting, renaming, simple lookups
- `simple` -- single-file edits, straightforward implementations
- `medium` -- multi-file changes, moderate reasoning
- `complex` -- architecture decisions, hard bugs, novel implementations
- `critical` -- system-wide changes, core architecture

The router selects the appropriate model. Don't override unless you have a specific reason.

### Self-Improvement
- You SHOULD improve any part of the codebase whenever you see an opportunity.
- This includes this file (CLAUDE.md), agent configs, model routing, everything.
- Use the `evolve` commit type for self-improvement changes.
- Include rationale in commit messages.
- Include before/after metrics or reasoning when possible.
- If an improvement breaks tests, revert it.

### Safety Through Automation
- Do not commit secrets, credentials, or API keys. `agent-git.sh` scans for these.
- Do not delete the git history or force-push in ways that lose work.
- If something breaks, revert the commit. Git is the safety net.
- Track costs. Don't burn through the API budget on unnecessary operations.
- If you're stuck in a loop, stop and try a different approach.
