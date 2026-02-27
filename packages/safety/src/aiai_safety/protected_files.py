"""Protected file checker: prevents unauthorized modification of critical files."""

from __future__ import annotations

import fnmatch
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiai_core.config import SafetyConfig


class ProtectedFileChecker:
    """Checks whether file modifications are allowed based on SafetyConfig.

    Protected file patterns are loaded from SafetyConfig.protected_files.
    Supports glob patterns via fnmatch.
    Set AIAI_OVERRIDE_PROTECTION=1 to bypass all checks (emergency use).
    """

    def __init__(self, config: SafetyConfig) -> None:
        self._patterns = list(config.protected_files)

    def check_modification(self, file_path: str) -> bool:
        """Check if a file modification is allowed.

        Returns True if modification is allowed, False if the file is protected.
        """
        if os.environ.get("AIAI_OVERRIDE_PROTECTION") == "1":
            return True

        # Normalize the path (remove leading ./ or /)
        normalized = file_path.lstrip("./")

        for pattern in self._patterns:
            pattern_normalized = pattern.lstrip("./")
            if fnmatch.fnmatch(normalized, pattern_normalized):
                return False
            # Also check if the file_path ends with the pattern (for relative paths)
            if normalized == pattern_normalized:
                return False
            # Check basename match for non-glob patterns
            if (
                "*" not in pattern
                and "?" not in pattern
                and normalized.endswith("/" + pattern_normalized)
            ):
                return False

        return True

    def get_protected_files(self) -> list[str]:
        """Return the list of protected file patterns."""
        return list(self._patterns)

    def is_protected(self, file_path: str) -> bool:
        """Check if a file is protected (convenience inverse of check_modification)."""
        return not self.check_modification(file_path)
