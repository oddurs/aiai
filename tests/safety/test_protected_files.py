"""Tests for aiai_safety.protected_files."""

from __future__ import annotations

import os

from aiai_core.config import SafetyConfig
from aiai_safety.protected_files import ProtectedFileChecker


def _make_config(protected: list[str] | None = None) -> SafetyConfig:
    if protected is None:
        protected = ["CLAUDE.md", "config/safety.yaml", "config/*.yaml"]
    return SafetyConfig(protected_files=protected)


class TestProtectedFileChecker:
    def test_exact_match_is_protected(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("CLAUDE.md") is False

    def test_unprotected_file_is_allowed(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("src/main.py") is True

    def test_glob_pattern_match(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("config/models.yaml") is False
        assert checker.check_modification("config/evolution.yaml") is False

    def test_non_matching_glob(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("config/readme.txt") is True

    def test_leading_dot_slash_normalized(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("./CLAUDE.md") is False
        assert checker.check_modification("./config/safety.yaml") is False

    def test_leading_slash_normalized(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("/CLAUDE.md") is False

    def test_is_protected_convenience(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.is_protected("CLAUDE.md") is True
        assert checker.is_protected("src/main.py") is False

    def test_get_protected_files(self) -> None:
        patterns = ["CLAUDE.md", "config/*.yaml"]
        checker = ProtectedFileChecker(_make_config(patterns))
        assert checker.get_protected_files() == patterns

    def test_empty_protected_list_allows_all(self) -> None:
        checker = ProtectedFileChecker(_make_config([]))
        assert checker.check_modification("CLAUDE.md") is True
        assert checker.check_modification("anything.txt") is True


class TestOverrideProtection:
    def test_override_env_var_bypasses_check(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        assert checker.check_modification("CLAUDE.md") is False
        os.environ["AIAI_OVERRIDE_PROTECTION"] = "1"
        try:
            assert checker.check_modification("CLAUDE.md") is True
        finally:
            del os.environ["AIAI_OVERRIDE_PROTECTION"]

    def test_override_only_with_exact_value(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        os.environ["AIAI_OVERRIDE_PROTECTION"] = "true"
        try:
            # Only "1" should bypass, not "true"
            assert checker.check_modification("CLAUDE.md") is False
        finally:
            del os.environ["AIAI_OVERRIDE_PROTECTION"]

    def test_override_not_set(self) -> None:
        checker = ProtectedFileChecker(_make_config())
        # Ensure the env var is not set
        os.environ.pop("AIAI_OVERRIDE_PROTECTION", None)
        assert checker.check_modification("CLAUDE.md") is False


class TestProtectedFilesFromRealConfig:
    def test_load_from_real_safety_config(self) -> None:
        """Load the actual safety config and verify protection."""
        from aiai_core.config import load_safety_config

        config = load_safety_config()
        checker = ProtectedFileChecker(config)
        assert checker.is_protected("CLAUDE.md") is True
        assert checker.is_protected("config/safety.yaml") is True
        assert checker.is_protected("src/some_random_file.py") is False
