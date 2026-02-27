"""Invariant checker: enforces safety constitution rules."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Invariant:
    """A named safety invariant with a check function.

    The check function returns None if the invariant holds,
    or an error message string if it is violated.
    """

    name: str
    description: str
    check_fn: Callable[[], str | None]


@dataclass(frozen=True)
class InvariantViolation:
    """Record of a failed invariant check."""

    invariant_name: str
    message: str


class InvariantChecker:
    """Runs a list of invariants and reports violations.

    Comes with built-in invariants for common safety checks.
    Additional invariants can be registered at runtime.
    """

    def __init__(self, repo_root: str | None = None) -> None:
        self._invariants: list[Invariant] = []
        self._repo_root = repo_root

    def register(self, invariant: Invariant) -> None:
        """Register a new invariant to check."""
        self._invariants.append(invariant)

    def register_builtins(self) -> None:
        """Register the built-in safety invariants."""
        self.register(Invariant(
            name="no_secrets_in_staged",
            description="No secrets or API keys in staged git changes",
            check_fn=self._check_no_secrets_staged,
        ))
        self.register(Invariant(
            name="no_force_push_protected",
            description="No force-push to protected branches",
            check_fn=self._check_no_force_push_protected,
        ))
        self.register(Invariant(
            name="test_count_non_decreasing",
            description="Passing test count never decreases",
            check_fn=self._check_test_count_non_decreasing,
        ))

    def check_all(self) -> list[InvariantViolation]:
        """Run all registered invariants and return any violations."""
        violations: list[InvariantViolation] = []
        for inv in self._invariants:
            try:
                result = inv.check_fn()
                if result is not None:
                    violations.append(InvariantViolation(
                        invariant_name=inv.name,
                        message=result,
                    ))
            except Exception as exc:
                violations.append(InvariantViolation(
                    invariant_name=inv.name,
                    message=f"Check raised exception: {exc}",
                ))
        return violations

    @property
    def invariants(self) -> list[Invariant]:
        """Return the list of registered invariants."""
        return list(self._invariants)

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the repo root."""
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self._repo_root,
        )

    def _check_no_secrets_staged(self) -> str | None:
        """Scan staged changes for potential secrets."""
        result = self._run_git("diff", "--cached", "--unified=0")
        if result.returncode != 0:
            return None  # Not in a git repo or nothing staged

        suspicious_patterns = [
            "PRIVATE KEY",
            "sk-",
            "sk_live_",
            "sk_test_",
            "AKIA",  # AWS access key prefix
            "password=",
            "secret=",
            "api_key=",
            "apikey=",
            "token=",
        ]

        added_lines = [
            line[1:]
            for line in result.stdout.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]

        for line in added_lines:
            upper = line.upper()
            for pattern in suspicious_patterns:
                if pattern.upper() in upper:
                    # Exclude lines that are clearly comments or documentation
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//"):
                        continue
                    return (
                        f"Potential secret found in staged changes: "
                        f"pattern '{pattern}' in line: {line[:80]}"
                    )
        return None

    def _check_no_force_push_protected(self) -> str | None:
        """Check that protected branches have not been force-pushed.

        This checks if the current branch is a protected branch and the
        reflog shows a force-push event. In practice, this is best used
        as a pre-push hook, but here we do a basic reflog check.
        """
        result = self._run_git("branch", "--show-current")
        if result.returncode != 0:
            return None

        branch = result.stdout.strip()
        protected = {"main", "master", "production"}
        if branch not in protected:
            return None

        # Check recent reflog for forced updates
        reflog = self._run_git("reflog", "show", "--format=%gs", "-n", "5")
        if reflog.returncode != 0:
            return None

        for entry in reflog.stdout.splitlines():
            if "forced-update" in entry.lower():
                return f"Force-push detected on protected branch '{branch}'"

        return None

    def _check_test_count_non_decreasing(self) -> str | None:
        """Verify the number of test files hasn't decreased.

        Uses a simple heuristic: count files matching test_*.py.
        A more robust implementation would track actual passing test count
        in a metrics store.
        """
        result = self._run_git(
            "ls-files", "--", "tests/**/test_*.py", "tests/test_*.py"
        )
        if result.returncode != 0:
            return None

        current_count = len([
            line for line in result.stdout.splitlines() if line.strip()
        ])

        # Compare with the previous commit
        prev = self._run_git(
            "show", "HEAD:.", "--name-only",
        )
        if prev.returncode != 0:
            return None  # No previous commit to compare

        prev_test_files = [
            line for line in prev.stdout.splitlines()
            if line.strip().startswith("tests/") and line.strip().endswith(".py")
            and "test_" in line
        ]
        prev_count = len(prev_test_files)

        if prev_count > 0 and current_count < prev_count:
            return (
                f"Test file count decreased: {prev_count} -> {current_count}. "
                f"Tests should never be removed without replacement."
            )
        return None
