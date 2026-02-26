# aiai - AI-made AI

An autonomous self-improving AI system where AI agents mobilize, iterate, and evolve their own codebase.

## Project Conventions

### Language & Runtime
- Primary: Python 3.11+, TypeScript/Node 20+
- Scripts: Bash (POSIX-compatible where possible)
- All code must be runnable without manual setup beyond `scripts/` bootstrapping

### Code Style
- Python: follow PEP 8, use type hints, max line length 100
- TypeScript: strict mode, prefer `const`, no `any` unless justified
- Shell: use `set -euo pipefail`, quote all variables, use `shellcheck`-clean code
- Keep functions small and single-purpose
- Prefer clarity over cleverness

### File Organization
```
aiai/
  docs/          -- documentation and research
  scripts/       -- automation, git workflows, bootstrap
  src/           -- source code
  tests/         -- test suites
  .github/       -- CI/CD and templates
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
- `evolve` -- self-improvement or architectural evolution

Scope is optional but encouraged (e.g., `feat(agent): add memory persistence`).

Keep the subject line under 72 characters. Use the body for context and rationale.

### Branch Naming
Format: `<type>/<short-description>`
Examples: `feat/memory-persistence`, `fix/prompt-injection-guard`, `evolve/self-eval-loop`

## Instructions for AI Agents

### Working on This Codebase
1. Always read relevant files before making changes.
2. Use `scripts/agent-git.sh` for all git operations -- it validates and guards against destructive actions.
3. Use `scripts/git-workflow.sh` for higher-level workflow commands (auto-commit, branching, changelog).
4. Run tests before committing. If tests do not exist yet for your change, write them.
5. Never force-push to `main`. Never rewrite shared history.
6. Create focused, single-purpose commits. One logical change per commit.
7. When uncertain, create a branch rather than committing directly to `main`.

### Self-Improvement Guidelines
- Agents may propose and implement improvements to any part of the codebase, including these conventions.
- All self-modifications must go through the standard branch-and-merge workflow.
- Use the `evolve` commit type for self-improvement changes.
- Document the rationale for architectural changes in commit messages and PR descriptions.
- Maintain backward compatibility unless a breaking change is explicitly justified.
- Self-improvement PRs should include before/after metrics or reasoning when possible.
- Never remove safety guards or validation without team consensus.

### Safety Rules
- Do not commit secrets, credentials, or API keys.
- Do not delete data or files without explicit justification.
- Do not disable tests or CI checks to make code pass.
- Treat `main` as a protected branch -- always use PRs for changes.
