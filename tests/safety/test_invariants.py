"""Tests for aiai_safety.invariants."""

from __future__ import annotations

from aiai_safety.invariants import Invariant, InvariantChecker, InvariantViolation


class TestInvariantDataclass:
    def test_fields(self) -> None:
        inv = Invariant(
            name="test",
            description="A test invariant",
            check_fn=lambda: None,
        )
        assert inv.name == "test"
        assert inv.description == "A test invariant"

    def test_check_fn_returns_none_means_pass(self) -> None:
        inv = Invariant(name="ok", description="always passes", check_fn=lambda: None)
        assert inv.check_fn() is None

    def test_check_fn_returns_string_means_fail(self) -> None:
        inv = Invariant(name="fail", description="always fails", check_fn=lambda: "bad")
        assert inv.check_fn() == "bad"


class TestInvariantViolation:
    def test_fields(self) -> None:
        v = InvariantViolation(invariant_name="test", message="something broke")
        assert v.invariant_name == "test"
        assert v.message == "something broke"


class TestInvariantChecker:
    def test_empty_checker_returns_no_violations(self) -> None:
        checker = InvariantChecker()
        assert checker.check_all() == []

    def test_passing_invariant(self) -> None:
        checker = InvariantChecker()
        checker.register(Invariant(
            name="always-pass",
            description="Always passes",
            check_fn=lambda: None,
        ))
        assert checker.check_all() == []

    def test_failing_invariant(self) -> None:
        checker = InvariantChecker()
        checker.register(Invariant(
            name="always-fail",
            description="Always fails",
            check_fn=lambda: "invariant violated",
        ))
        violations = checker.check_all()
        assert len(violations) == 1
        assert violations[0].invariant_name == "always-fail"
        assert violations[0].message == "invariant violated"

    def test_mixed_invariants(self) -> None:
        checker = InvariantChecker()
        checker.register(Invariant(
            name="pass-1",
            description="Passes",
            check_fn=lambda: None,
        ))
        checker.register(Invariant(
            name="fail-1",
            description="Fails",
            check_fn=lambda: "error 1",
        ))
        checker.register(Invariant(
            name="pass-2",
            description="Also passes",
            check_fn=lambda: None,
        ))
        checker.register(Invariant(
            name="fail-2",
            description="Also fails",
            check_fn=lambda: "error 2",
        ))
        violations = checker.check_all()
        assert len(violations) == 2
        names = [v.invariant_name for v in violations]
        assert "fail-1" in names
        assert "fail-2" in names

    def test_exception_in_check_fn_becomes_violation(self) -> None:
        def bad_check() -> str | None:
            raise RuntimeError("boom")

        checker = InvariantChecker()
        checker.register(Invariant(
            name="exploder",
            description="Raises exception",
            check_fn=bad_check,
        ))
        violations = checker.check_all()
        assert len(violations) == 1
        assert violations[0].invariant_name == "exploder"
        assert "boom" in violations[0].message

    def test_register_multiple_and_list(self) -> None:
        checker = InvariantChecker()
        inv1 = Invariant(name="a", description="A", check_fn=lambda: None)
        inv2 = Invariant(name="b", description="B", check_fn=lambda: None)
        checker.register(inv1)
        checker.register(inv2)
        assert len(checker.invariants) == 2
        assert checker.invariants[0].name == "a"
        assert checker.invariants[1].name == "b"


class TestBuiltinInvariants:
    def test_register_builtins_adds_invariants(self) -> None:
        checker = InvariantChecker()
        checker.register_builtins()
        names = [inv.name for inv in checker.invariants]
        assert "no_secrets_in_staged" in names
        assert "no_force_push_protected" in names
        assert "test_count_non_decreasing" in names

    def test_builtins_run_without_error_outside_git(self) -> None:
        """Built-in invariants should handle not being in a git repo gracefully."""
        import tempfile

        checker = InvariantChecker(repo_root=tempfile.gettempdir())
        checker.register_builtins()
        # Should not raise; may return violations or empty list
        violations = checker.check_all()
        # At minimum, no unhandled exceptions
        assert isinstance(violations, list)
