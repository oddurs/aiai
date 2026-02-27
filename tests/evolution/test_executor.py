"""Tests for HypothesisExecutor."""

import pytest
from aiai_core.config import EvolutionConfig
from aiai_evolution.executor import ExecutionResult, HypothesisExecutor
from aiai_evolution.hypothesis import Hypothesis


def _make_hypothesis(hypothesis_id: str = "test-1") -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        title="Test hypothesis",
        description="A test hypothesis",
        pattern_source="cost_spike",
        expected_improvement=0.15,
        budget_usd=5.0,
    )


class FakeGitRunner:
    """Records git commands for verification."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.commands: list[list[str]] = []
        self._fail_on = fail_on

    def __call__(self, cmd: list[str]) -> None:
        self.commands.append(cmd)
        if self._fail_on and self._fail_on in " ".join(cmd):
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}")


class TestExecutionResult:
    def test_creation(self) -> None:
        result = ExecutionResult(
            hypothesis_id="h1",
            success=True,
            improvement_pct=0.15,
            cost_usd=1.5,
        )
        assert result.hypothesis_id == "h1"
        assert result.success is True
        assert result.improvement_pct == 0.15

    def test_failure_with_error(self) -> None:
        result = ExecutionResult(
            hypothesis_id="h1",
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestHypothesisExecutor:
    @pytest.mark.asyncio
    async def test_creates_checkpoint(self) -> None:
        git = FakeGitRunner()
        config = EvolutionConfig(checkpoint_tag_prefix="pre-evo")
        executor = HypothesisExecutor(config=config, git_runner=git)
        hypothesis = _make_hypothesis("h1")
        await executor.execute(hypothesis)
        # Should have created a tag
        assert any("tag" in cmd for cmd in git.commands)
        tag_cmd = [cmd for cmd in git.commands if "tag" in cmd][0]
        assert "pre-evo/h1" in tag_cmd

    @pytest.mark.asyncio
    async def test_reverts_when_improvement_below_threshold(self) -> None:
        git = FakeGitRunner()
        config = EvolutionConfig(
            min_improvement_threshold=0.05,
            auto_revert_on_regression=True,
            checkpoint_tag_prefix="pre-evo",
        )
        executor = HypothesisExecutor(config=config, git_runner=git)
        hypothesis = _make_hypothesis("h2")
        result = await executor.execute(hypothesis)
        # Placeholder returns 0% improvement, so should revert
        assert result.success is False
        assert result.error == "Improvement below threshold"
        assert hypothesis.status == "reverted"
        # Should have reset --hard
        reset_cmds = [cmd for cmd in git.commands if "reset" in cmd]
        assert len(reset_cmds) == 1

    @pytest.mark.asyncio
    async def test_no_revert_when_disabled(self) -> None:
        git = FakeGitRunner()
        config = EvolutionConfig(
            min_improvement_threshold=0.05,
            auto_revert_on_regression=False,
            checkpoint_tag_prefix="pre-evo",
        )
        executor = HypothesisExecutor(config=config, git_runner=git)
        hypothesis = _make_hypothesis("h3")
        result = await executor.execute(hypothesis)
        assert result.success is False
        # No reset commands when auto_revert is disabled
        reset_cmds = [cmd for cmd in git.commands if "reset" in cmd]
        assert len(reset_cmds) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_failure(self) -> None:
        git = FakeGitRunner(fail_on="tag")
        executor = HypothesisExecutor(git_runner=git)
        hypothesis = _make_hypothesis("h4")
        result = await executor.execute(hypothesis)
        assert result.success is False
        assert "Checkpoint failed" in (result.error or "")
        assert hypothesis.status == "failed"

    @pytest.mark.asyncio
    async def test_revert_failure_does_not_crash(self) -> None:
        """Even if revert fails, the executor should not raise."""
        git = FakeGitRunner(fail_on="reset")
        config = EvolutionConfig(
            auto_revert_on_regression=True,
            min_improvement_threshold=0.05,
        )
        executor = HypothesisExecutor(config=config, git_runner=git)
        hypothesis = _make_hypothesis("h5")
        result = await executor.execute(hypothesis)
        assert result.success is False
        assert hypothesis.status == "reverted"

    @pytest.mark.asyncio
    async def test_result_contains_metrics(self) -> None:
        git = FakeGitRunner()
        executor = HypothesisExecutor(git_runner=git)
        hypothesis = _make_hypothesis("h6")
        result = await executor.execute(hypothesis)
        assert result.hypothesis_id == "h6"
        assert isinstance(result.before_metrics, dict)
        assert isinstance(result.after_metrics, dict)

    @pytest.mark.asyncio
    async def test_default_config(self) -> None:
        git = FakeGitRunner()
        executor = HypothesisExecutor(git_runner=git)
        hypothesis = _make_hypothesis("h7")
        result = await executor.execute(hypothesis)
        # With default config (min_improvement_threshold=0.05), placeholder returns 0
        assert result.success is False
