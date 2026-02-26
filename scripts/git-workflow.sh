#!/usr/bin/env bash
# git-workflow.sh -- Agentic git workflow automation for aiai
# Provides high-level git operations designed for AI agent use.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTECTED_BRANCHES=("main" "master" "production")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_color() {
    local code="$1"; shift
    if [[ -t 1 ]]; then
        printf '\033[%sm%s\033[0m\n' "$code" "$*"
    else
        printf '%s\n' "$*"
    fi
}

info()  { _color "34" "[info]  $*"; }
ok()    { _color "32" "[ok]    $*"; }
warn()  { _color "33" "[warn]  $*"; }
err()   { _color "31" "[error] $*"; }

_require_git() {
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        err "Not inside a git repository."
        exit 1
    fi
}

_current_branch() {
    git rev-parse --abbrev-ref HEAD
}

_is_protected() {
    local branch="$1"
    for p in "${PROTECTED_BRANCHES[@]}"; do
        [[ "$branch" == "$p" ]] && return 0
    done
    return 1
}

_has_changes() {
    ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]
}

# --------------------------------------------------------------------------- #
# Diff-based commit message generation
# --------------------------------------------------------------------------- #

_generate_commit_message() {
    local diff_stat files_changed insertions deletions summary type scope

    diff_stat="$(git diff --cached --stat)"
    if [[ -z "$diff_stat" ]]; then
        echo "chore: empty commit"
        return
    fi

    files_changed="$(git diff --cached --numstat | wc -l | tr -d ' ')"
    insertions="$(git diff --cached --numstat | awk '{s+=$1} END {print s+0}')"
    deletions="$(git diff --cached --numstat | awk '{s+=$1} END {print s+0}')"

    # Determine type from file paths
    local paths
    paths="$(git diff --cached --name-only)"

    type="chore"
    scope=""

    if echo "$paths" | grep -qE '\.test\.|_test\.|tests/|__tests__/'; then
        type="test"
    elif echo "$paths" | grep -qE '\.md$|docs/|README'; then
        type="docs"
    elif echo "$paths" | grep -qE '\.github/|\.ci|Makefile|Dockerfile'; then
        type="ci"
    elif echo "$paths" | grep -qE 'scripts/'; then
        type="chore"
        scope="scripts"
    fi

    # Determine scope from common directory
    if [[ -z "$scope" ]]; then
        local common_dir
        common_dir="$(echo "$paths" | head -5 | xargs -I{} dirname {} | sort -u | head -1)"
        if [[ "$common_dir" != "." && -n "$common_dir" ]]; then
            scope="$(basename "$common_dir")"
        fi
    fi

    # Build a short summary from the diff
    local diff_names
    diff_names="$(git diff --cached --name-only | head -3 | xargs -I{} basename {} | paste -sd ', ' -)"

    if [[ "$files_changed" -eq 1 ]]; then
        local single_file
        single_file="$(git diff --cached --name-only)"
        summary="update $(basename "$single_file")"
    elif [[ "$files_changed" -le 3 ]]; then
        summary="update $diff_names"
    else
        summary="update ${files_changed} files (+${insertions}/-${deletions})"
    fi

    if [[ -n "$scope" ]]; then
        echo "${type}(${scope}): ${summary}"
    else
        echo "${type}: ${summary}"
    fi
}

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

cmd_auto_commit() {
    _require_git
    local message="${1:-}"

    if ! _has_changes; then
        info "No changes to commit."
        return 0
    fi

    # Stage all changes
    git add -A

    # Generate message if not provided
    if [[ -z "$message" ]]; then
        message="$(_generate_commit_message)"
    fi

    git commit -m "$message"
    ok "Committed: $message"
}

cmd_auto_branch() {
    _require_git
    local branch_type="${1:-feat}"
    local description="${2:-}"

    if [[ -z "$description" ]]; then
        err "Usage: git-workflow.sh auto-branch <type> <description>"
        err "Types: feat, fix, refactor, docs, test, ci, chore, evolve"
        exit 1
    fi

    # Sanitize description into branch name
    local branch_name
    branch_name="${branch_type}/$(echo "$description" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9\-]//g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')"

    # Check if branch already exists
    if git rev-parse --verify "$branch_name" &>/dev/null; then
        warn "Branch '$branch_name' already exists. Switching to it."
        git checkout "$branch_name"
    else
        git checkout -b "$branch_name"
        ok "Created and switched to branch: $branch_name"
    fi
}

cmd_auto_merge() {
    _require_git
    local source_branch="${1:-}"
    local target_branch="${2:-main}"

    if [[ -z "$source_branch" ]]; then
        err "Usage: git-workflow.sh auto-merge <source-branch> [target-branch]"
        exit 1
    fi

    # Verify source exists
    if ! git rev-parse --verify "$source_branch" &>/dev/null; then
        err "Source branch '$source_branch' does not exist."
        exit 1
    fi

    # Switch to target
    git checkout "$target_branch"

    # Attempt merge
    if git merge --no-ff "$source_branch" -m "merge: ${source_branch} into ${target_branch}"; then
        ok "Successfully merged '$source_branch' into '$target_branch'."
    else
        err "Merge conflict detected."
        err "Conflicting files:"
        git diff --name-only --diff-filter=U
        warn "Resolve conflicts, then run: git add . && git commit"
        exit 1
    fi
}

cmd_auto_changelog() {
    _require_git
    local since="${1:-}"
    local changelog_file="${REPO_ROOT}/CHANGELOG.md"

    info "Generating changelog..."

    local log_args=("--pretty=format:- %s (%h)" "--no-merges")
    if [[ -n "$since" ]]; then
        log_args+=("${since}..HEAD")
    fi

    local entries
    entries="$(git log "${log_args[@]}" 2>/dev/null || echo "")"

    if [[ -z "$entries" ]]; then
        warn "No commits found for changelog."
        return 0
    fi

    local date_str
    date_str="$(date +%Y-%m-%d)"

    local header="## [$date_str]"

    # Group by type
    local feats fixes refactors docs others
    feats="$(echo "$entries" | grep -E '^\- feat' || true)"
    fixes="$(echo "$entries" | grep -E '^\- fix' || true)"
    refactors="$(echo "$entries" | grep -E '^\- refactor' || true)"
    docs="$(echo "$entries" | grep -E '^\- docs' || true)"
    others="$(echo "$entries" | grep -vE '^\- (feat|fix|refactor|docs)' || true)"

    {
        echo "$header"
        echo ""
        if [[ -n "$feats" ]]; then
            echo "### Features"
            echo "$feats"
            echo ""
        fi
        if [[ -n "$fixes" ]]; then
            echo "### Fixes"
            echo "$fixes"
            echo ""
        fi
        if [[ -n "$refactors" ]]; then
            echo "### Refactoring"
            echo "$refactors"
            echo ""
        fi
        if [[ -n "$docs" ]]; then
            echo "### Documentation"
            echo "$docs"
            echo ""
        fi
        if [[ -n "$others" ]]; then
            echo "### Other"
            echo "$others"
            echo ""
        fi
    } > "$changelog_file"

    ok "Changelog written to $changelog_file"
}

cmd_snapshot() {
    _require_git
    local label="${1:-wip}"

    if ! _has_changes; then
        info "No changes to snapshot."
        return 0
    fi

    git add -A

    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    local branch
    branch="$(_current_branch)"
    local message="snapshot(${branch}): ${label} @ ${timestamp}"

    git commit -m "$message"
    ok "Snapshot: $message"
}

cmd_help() {
    cat <<'USAGE'
git-workflow.sh -- Agentic git workflow automation for aiai

Usage: git-workflow.sh <command> [arguments]

Commands:
  auto-commit [message]              Stage all changes and commit.
                                     If no message is given, one is generated
                                     from the diff.

  auto-branch <type> <description>   Create a feature branch.
                                     Types: feat, fix, refactor, docs, test,
                                            ci, chore, evolve

  auto-merge <source> [target]       Merge source branch into target (default:
                                     main). Detects conflicts and reports them.

  auto-changelog [since-ref]         Generate CHANGELOG.md from git log.
                                     Optionally specify a starting ref/tag.

  snapshot [label]                    Quick work-in-progress commit with a
                                     timestamp. Default label: "wip".

  help                               Show this help message.

Examples:
  git-workflow.sh auto-commit
  git-workflow.sh auto-commit "feat(agent): add memory module"
  git-workflow.sh auto-branch feat "add memory persistence"
  git-workflow.sh auto-merge feat/add-memory-persistence main
  git-workflow.sh auto-changelog v0.1.0
  git-workflow.sh snapshot "checkpoint before refactor"
USAGE
}

# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        auto-commit)    cmd_auto_commit "$@" ;;
        auto-branch)    cmd_auto_branch "$@" ;;
        auto-merge)     cmd_auto_merge "$@" ;;
        auto-changelog) cmd_auto_changelog "$@" ;;
        snapshot)       cmd_snapshot "$@" ;;
        help|--help|-h) cmd_help ;;
        *)
            err "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
