"""Tests for aiai_safety.loop_detector."""

from __future__ import annotations

from aiai_safety.loop_detector import LoopDetection, LoopDetector


class TestRepeatedActionDetection:
    def test_no_detection_with_few_entries(self) -> None:
        ld = LoopDetector(repeat_threshold=3)
        ld.record("action-a", "hash-1")
        ld.record("action-a", "hash-2")
        assert ld.check() is None

    def test_detects_repeated_action(self) -> None:
        ld = LoopDetector(repeat_threshold=3)
        ld.record("action-a", "hash-1")
        ld.record("action-a", "hash-2")
        ld.record("action-a", "hash-3")
        result = ld.check()
        assert result is not None
        assert result.pattern == "repeated_action"
        assert "action-a" in result.detail
        assert "3" in result.detail

    def test_no_detection_when_mixed(self) -> None:
        ld = LoopDetector(repeat_threshold=3)
        ld.record("action-a", "hash-1")
        ld.record("action-b", "hash-2")
        ld.record("action-a", "hash-3")
        assert ld.check() is None

    def test_custom_threshold(self) -> None:
        ld = LoopDetector(repeat_threshold=5)
        for i in range(4):
            ld.record("action-a", f"hash-{i}")
        assert ld.check() is None
        ld.record("action-a", "hash-5")
        result = ld.check()
        assert result is not None
        assert result.pattern == "repeated_action"


class TestOscillationDetection:
    def test_no_detection_with_short_history(self) -> None:
        ld = LoopDetector()
        ld.record("a", "h1")
        ld.record("b", "h2")
        ld.record("a", "h3")
        assert ld.check() is None

    def test_detects_oscillation(self) -> None:
        ld = LoopDetector(repeat_threshold=100)  # High to avoid repeated trigger
        ld.record("a", "h1")
        ld.record("b", "h2")
        ld.record("a", "h3")
        ld.record("b", "h4")
        result = ld.check()
        assert result is not None
        assert result.pattern == "oscillation"
        assert "a" in result.detail
        assert "b" in result.detail

    def test_longer_oscillation(self) -> None:
        ld = LoopDetector(repeat_threshold=100)
        for _ in range(5):
            ld.record("x", "h1")
            ld.record("y", "h2")
        result = ld.check()
        assert result is not None
        assert result.pattern == "oscillation"

    def test_no_oscillation_with_same_action(self) -> None:
        ld = LoopDetector(repeat_threshold=100)
        ld.record("a", "h1")
        ld.record("a", "h2")
        ld.record("a", "h3")
        ld.record("a", "h4")
        # This is repeated, not oscillation
        result = ld.check()
        assert result is None or result.pattern != "oscillation"


class TestNoProgressDetection:
    def test_no_detection_below_threshold(self) -> None:
        ld = LoopDetector(
            repeat_threshold=100,
            output_repeat_threshold=5,
        )
        for i in range(4):
            ld.record(f"action-{i}", "same-hash")
        assert ld.check() is None

    def test_detects_repeated_output(self) -> None:
        ld = LoopDetector(
            repeat_threshold=100,
            output_repeat_threshold=5,
        )
        for i in range(5):
            ld.record(f"action-{i}", "same-hash")
        result = ld.check()
        assert result is not None
        assert result.pattern == "no_progress"
        assert "same-hash" in result.detail

    def test_mixed_outputs_no_detection(self) -> None:
        ld = LoopDetector(
            repeat_threshold=100,
            output_repeat_threshold=5,
        )
        for i in range(10):
            ld.record(f"action-{i}", f"hash-{i}")
        assert ld.check() is None


class TestWindowBehavior:
    def test_window_limits_history(self) -> None:
        ld = LoopDetector(window_size=5, repeat_threshold=4)
        # Fill with "a" entries
        for i in range(3):
            ld.record("a", f"h{i}")
        # Break the sequence
        ld.record("b", "hb")
        # Add more "a" entries
        for i in range(3):
            ld.record("a", f"h{i+10}")
        # Window is 5, so the old "a" entries have been pushed out
        # Only the last 3 "a" entries are in the window (after "b")
        result = ld.check()
        assert result is None or result.pattern != "repeated_action"

    def test_clear_resets_state(self) -> None:
        ld = LoopDetector(repeat_threshold=3)
        ld.record("a", "h1")
        ld.record("a", "h2")
        ld.record("a", "h3")
        assert ld.check() is not None
        ld.clear()
        assert ld.check() is None


class TestLoopDetectionDataclass:
    def test_fields(self) -> None:
        detection = LoopDetection(
            pattern="repeated_action",
            detail="test detail",
            window_size=10,
        )
        assert detection.pattern == "repeated_action"
        assert detection.detail == "test detail"
        assert detection.window_size == 10

    def test_frozen(self) -> None:
        detection = LoopDetection(
            pattern="test",
            detail="detail",
            window_size=5,
        )
        try:
            detection.pattern = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass
