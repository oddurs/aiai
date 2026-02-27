"""Loop detector: identifies stuck agents repeating the same actions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopDetection:
    """Information about a detected loop."""

    pattern: str  # "repeated_action", "oscillation", "no_progress"
    detail: str
    window_size: int


class LoopDetector:
    """Detects stuck agents by tracking recent actions and output hashes.

    Detects three patterns:
    1. Repeated identical actions (N consecutive identical actions)
    2. Oscillation (A-B-A-B alternating pattern)
    3. No progress (same output hash N times within the window)
    """

    def __init__(
        self,
        window_size: int = 20,
        repeat_threshold: int = 3,
        output_repeat_threshold: int = 5,
    ) -> None:
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold
        self._output_repeat_threshold = output_repeat_threshold
        self._history: deque[tuple[str, str]] = deque(maxlen=window_size)

    def record(self, action: str, output_hash: str) -> None:
        """Record an action and its output hash."""
        self._history.append((action, output_hash))

    def check(self) -> LoopDetection | None:
        """Check for loop patterns. Returns LoopDetection if found, else None."""
        if len(self._history) < 2:
            return None

        result = self._check_repeated_action()
        if result:
            return result

        result = self._check_oscillation()
        if result:
            return result

        result = self._check_no_progress()
        if result:
            return result

        return None

    def _check_repeated_action(self) -> LoopDetection | None:
        """Detect N identical consecutive actions."""
        if len(self._history) < self._repeat_threshold:
            return None

        items = list(self._history)
        last_action = items[-1][0]
        count = 0
        for action, _ in reversed(items):
            if action == last_action:
                count += 1
            else:
                break

        if count >= self._repeat_threshold:
            return LoopDetection(
                pattern="repeated_action",
                detail=f"Action '{last_action}' repeated {count} times consecutively",
                window_size=len(self._history),
            )
        return None

    def _check_oscillation(self) -> LoopDetection | None:
        """Detect A-B-A-B alternating pattern (at least 4 entries = 2 full cycles)."""
        if len(self._history) < 4:
            return None

        items = list(self._history)
        # Check last 4+ entries for alternation
        a_action = items[-2][0]
        b_action = items[-1][0]

        if a_action == b_action:
            return None

        # Count how far back the alternation extends
        cycle_count = 0
        idx = len(items) - 1
        while idx >= 1:
            if items[idx][0] == b_action and items[idx - 1][0] == a_action:
                cycle_count += 1
                idx -= 2
            else:
                break

        if cycle_count >= 2:
            return LoopDetection(
                pattern="oscillation",
                detail=(
                    f"Oscillating between '{a_action}' and '{b_action}' "
                    f"for {cycle_count} cycles"
                ),
                window_size=len(self._history),
            )
        return None

    def _check_no_progress(self) -> LoopDetection | None:
        """Detect same output hash repeated N times within the window."""
        if len(self._history) < self._output_repeat_threshold:
            return None

        output_counts: dict[str, int] = {}
        for _, output_hash in self._history:
            output_counts[output_hash] = output_counts.get(output_hash, 0) + 1

        for output_hash, count in output_counts.items():
            if count >= self._output_repeat_threshold:
                return LoopDetection(
                    pattern="no_progress",
                    detail=(
                        f"Output hash '{output_hash}' appeared {count} times "
                        f"in window of {len(self._history)}"
                    ),
                    window_size=len(self._history),
                )
        return None

    def clear(self) -> None:
        """Clear the action history."""
        self._history.clear()
