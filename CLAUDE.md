# aiai - AI-made AI

Self-improving AI infrastructure. Agents build, test, and evolve their own codebase.

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

### Working on This Codebase
1. Read relevant files before making changes.
2. Use `scripts/agent-git.sh` for all git operations -- it blocks destructive actions and scans for secrets.
3. Use `scripts/git-workflow.sh` for auto-commit, branching, changelog.
4. Run tests before committing. If tests don't exist for your change, write them.
5. Commit to feature branches, not `main`. Create PRs for merging.
6. One logical change per commit.

### Model Routing
When making OpenRouter API calls, declare task complexity:
- `trivial` -- formatting, renaming, simple lookups
- `simple` -- single-file edits, straightforward implementations
- `medium` -- multi-file changes, moderate reasoning
- `complex` -- architecture decisions, hard bugs, novel implementations
- `critical` -- safety-critical, system-wide changes

The router will select the appropriate model. Don't override unless you have a specific reason.

### Approval Gates
**You can do freely:**
- Write code, run tests, commit to branches
- Create and manage feature branches
- Build tools in `src/tools/`
- Spawn agent teams for complex tasks
- Research, analyze, synthesize

**You MUST create a PR for human approval:**
- Any merge to `main`
- Any change to this file (CLAUDE.md)
- Any change to agent configs (`.claude/agents/`)
- Any change to model routing config (`config/models.yaml`)
- Any change to the self-improvement engine (`src/evolution/`)

### Self-Improvement
- You may propose improvements to any part of the codebase, including these conventions.
- Self-modifications go through the standard branch-and-PR workflow.
- Use the `evolve` commit type for self-improvement changes.
- Include rationale in commit messages and PR descriptions.
- Include before/after metrics or reasoning when possible.
- Never remove safety guards or approval gates.

### Safety Rules
- Do not commit secrets, credentials, or API keys.
- Do not delete data or files without explicit justification.
- Do not disable tests or CI checks to make code pass.
- Do not force-push to `main`.
- Do not bypass approval gates.
- `scripts/agent-git.sh` enforces these at the tool level.
