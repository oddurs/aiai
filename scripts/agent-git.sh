#!/usr/bin/env bash
# agent-git.sh -- Safe git wrapper for AI agents
# Validates operations, prevents destructive actions, provides structured output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTECTED_BRANCHES=("main" "master" "production")
OUTPUT_FORMAT="${AGENT_GIT_FORMAT:-text}"  # "text" or "json"

# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

_json() {
    local status="$1" message="$2" data="${3:-{}}"
    printf '{"status":"%s","message":"%s","data":%s}\n' "$status" "$message" "$data"
}

_out() {
    local status="$1"; shift
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        _json "$status" "$*"
    else
        case "$status" in
            ok)    printf '[ok]    %s\n' "$*" ;;
            error) printf '[error] %s\n' "$*" >&2 ;;
            warn)  printf '[warn]  %s\n' "$*" ;;
            info)  printf '[info]  %s\n' "$*" ;;
        esac
    fi
}

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_require_git() {
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        _out error "Not inside a git repository."
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

_block_destructive() {
    local action="$1"
    _out error "Blocked destructive action: $action"
    _out error "AI agents must not perform destructive operations on protected branches."
    exit 1
}

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

cmd_status() {
    _require_git
    local branch modified untracked staged
    branch="$(_current_branch)"
    modified="$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')"
    untracked="$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')"
    staged="$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')"

    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        local is_clean="false"
        if [[ "$modified" -eq 0 && "$untracked" -eq 0 && "$staged" -eq 0 ]]; then
            is_clean="true"
        fi
        _json "ok" "status" "$(printf '{"branch":"%s","modified":%d,"untracked":%d,"staged":%d,"clean":%s}' \
            "$branch" "$modified" "$untracked" "$staged" "$is_clean")"
    else
        echo "Branch:    $branch"
        echo "Modified:  $modified"
        echo "Untracked: $untracked"
        echo "Staged:    $staged"
        if [[ "$modified" -eq 0 && "$untracked" -eq 0 && "$staged" -eq 0 ]]; then
            echo "Clean:     yes"
        else
            echo "Clean:     no"
        fi
    fi
}

cmd_safe_commit() {
    _require_git
    local message="${1:-}"

    # Validate: do not commit directly to protected branches without a message
    local branch
    branch="$(_current_branch)"
    if _is_protected "$branch" && [[ -z "$message" ]]; then
        _out warn "On protected branch '$branch'. Provide an explicit commit message."
        exit 1
    fi

    # Check for secrets in staged files
    git add -A
    local staged_content
    staged_content="$(git diff --cached --unified=0 2>/dev/null || true)"

    if echo "$staged_content" | grep -qiE '(api[_-]?key|secret|password|token|credential)\s*[:=]'; then
        _out error "Potential secret detected in staged changes. Aborting commit."
        _out error "Review staged files and remove secrets before committing."
        git reset HEAD -- . &>/dev/null
        exit 1
    fi

    # Generate message if not provided
    if [[ -z "$message" ]]; then
        message="$("$SCRIPT_DIR/git-workflow.sh" auto-commit --dry-run 2>/dev/null || echo "chore: auto-commit")"
        # Fallback: generate from diff stats
        local files_changed
        files_changed="$(git diff --cached --name-only | wc -l | tr -d ' ')"
        message="chore: update ${files_changed} file(s)"
    fi

    git commit -m "$message"
    _out ok "Committed on '$branch': $message"
}

cmd_safe_push() {
    _require_git
    local branch
    branch="$(_current_branch)"
    local force="${1:-}"

    # Block force push to protected branches
    if [[ "$force" == "--force" || "$force" == "-f" ]]; then
        if _is_protected "$branch"; then
            _block_destructive "force push to '$branch'"
        fi
        _out warn "Force pushing to '$branch'..."
        git push --force-with-lease origin "$branch"
    else
        git push origin "$branch"
    fi

    _out ok "Pushed '$branch' to origin."
}

cmd_create_branch() {
    _require_git
    local name="${1:-}"

    if [[ -z "$name" ]]; then
        _out error "Usage: agent-git.sh branch <branch-name>"
        exit 1
    fi

    # Sanitize
    name="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9\/-]//g')"

    if git rev-parse --verify "$name" &>/dev/null; then
        _out info "Branch '$name' exists. Switching to it."
        git checkout "$name"
    else
        git checkout -b "$name"
        _out ok "Created branch: $name"
    fi
}

cmd_safe_delete_branch() {
    _require_git
    local branch="${1:-}"

    if [[ -z "$branch" ]]; then
        _out error "Usage: agent-git.sh delete-branch <branch-name>"
        exit 1
    fi

    if _is_protected "$branch"; then
        _block_destructive "delete protected branch '$branch'"
    fi

    local current
    current="$(_current_branch)"
    if [[ "$current" == "$branch" ]]; then
        _out error "Cannot delete current branch. Switch to another branch first."
        exit 1
    fi

    # Only delete if fully merged
    if git branch --merged | grep -qE "^\s*${branch}$"; then
        git branch -d "$branch"
        _out ok "Deleted merged branch: $branch"
    else
        _out error "Branch '$branch' is not fully merged. Refusing to delete."
        _out info "Use the standard git CLI if you are certain."
        exit 1
    fi
}

cmd_diff_summary() {
    _require_git
    local files_changed insertions deletions

    files_changed="$(git diff --stat HEAD 2>/dev/null | tail -1 || echo "no changes")"
    local names
    names="$(git diff --name-only HEAD 2>/dev/null | head -10 || true)"

    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        local files_list
        files_list="$(git diff --name-only HEAD 2>/dev/null | head -10 | jq -R . | jq -s . 2>/dev/null || echo "[]")"
        _json "ok" "diff summary" "$(printf '{"summary":"%s","files":%s}' \
            "$(echo "$files_changed" | tr '"' "'")" "$files_list")"
    else
        echo "Diff summary:"
        echo "$files_changed"
        if [[ -n "$names" ]]; then
            echo ""
            echo "Changed files:"
            echo "$names"
        fi
    fi
}

cmd_log() {
    _require_git
    local count="${1:-10}"
    local format

    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        format='{"hash":"%H","short":"%h","subject":"%s","author":"%an","date":"%ai"}'
        printf '[\n'
        git log -"$count" --pretty=format:"$format," 2>/dev/null | sed '$ s/,$//'
        printf '\n]\n'
    else
        git log -"$count" --oneline --decorate 2>/dev/null
    fi
}

cmd_help() {
    cat <<'USAGE'
agent-git.sh -- Safe git wrapper for AI agents

Usage: agent-git.sh <command> [arguments]

Environment:
  AGENT_GIT_FORMAT=json    Output in JSON (default: text)

Commands:
  status                   Show branch, modified/untracked/staged counts.
  commit [message]         Stage all, check for secrets, commit.
                           Requires explicit message on protected branches.
  push [--force]           Push current branch. Blocks force-push to
                           protected branches.
  branch <name>            Create or switch to a branch (sanitized name).
  delete-branch <name>     Delete a branch (only if merged, not protected).
  diff                     Show diff summary and changed files.
  log [count]              Show recent commits (default: 10).
  help                     Show this help message.

Safety:
  - Force push to main/master/production is blocked.
  - Deleting protected branches is blocked.
  - Commits are scanned for potential secrets before creation.
  - Direct commits to protected branches require explicit messages.

Examples:
  agent-git.sh status
  AGENT_GIT_FORMAT=json agent-git.sh status
  agent-git.sh commit "feat(core): add eval loop"
  agent-git.sh branch feat/memory-module
  agent-git.sh push
  agent-git.sh log 5
USAGE
}

# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        status)        cmd_status "$@" ;;
        commit)        cmd_safe_commit "$@" ;;
        push)          cmd_safe_push "$@" ;;
        branch)        cmd_create_branch "$@" ;;
        delete-branch) cmd_safe_delete_branch "$@" ;;
        diff)          cmd_diff_summary "$@" ;;
        log)           cmd_log "$@" ;;
        help|--help|-h) cmd_help ;;
        *)
            _out error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
