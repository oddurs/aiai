"""Hypothesis executor: checkpoint, execute, validate, revert."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiai_core.config import EvolutionConfig
from aiai_core.logging import get_logger

if TYPE_CHECKING:
    from aiai_evolution.hypothesis import Hypothesis


@dataclass
class ExecutionResult:
    """Result of executing a hypothesis."""

    hypothesis_id: str
    success: bool
    before_metrics: dict[str, object] = field(default_factory=dict)  # noqa: UP006
    after_metrics: dict[str, object] = field(default_factory=dict)  # noqa: UP006
    improvement_pct: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class HypothesisExecutor:
    """Executes improvement hypotheses with checkpointing and rollback.

    Framework:
    1. Create a git checkpoint tag before execution
    2. Execute the hypothesis (placeholder for now)
    3. Validate results against improvement threshold
    4. Auto-revert on failure or regression
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        git_runner: object | None = None,
    ) -> None:
        if config is None:
            config = EvolutionConfig()
        self._config = config
        self._logger = get_logger()
        # Allow injecting a git command runner for testing.
        self._git_runner = git_runner or _default_git_runner

    def _create_checkpoint(self, hypothesis: Hypothesis) -> str:
        """Create a git tag as a checkpoint before executing."""
        tag_name = (
            f"{self._config.checkpoint_tag_prefix}/{hypothesis.id}"
        )
        self._git_runner(["git", "tag", tag_name])  # type: ignore[operator]
        self._logger.info("executor.checkpoint", tag=tag_name, hypothesis=hypothesis.id)
        return tag_name

    def _revert_to_checkpoint(self, tag_name: str) -> None:
        """Revert to a checkpoint tag."""
        self._git_runner(["git", "reset", "--hard", tag_name])  # type: ignore[operator]
        self._logger.info("executor.revert", tag=tag_name)

    async def execute(self, hypothesis: Hypothesis) -> ExecutionResult:
        """Execute a hypothesis with checkpoint and validation.

        Args:
            hypothesis: The hypothesis to execute.

        Returns:
            ExecutionResult with success/failure and metrics.
        """
        hypothesis.status = "executing"
        self._logger.info(
            "executor.start",
            hypothesis=hypothesis.id,
            title=hypothesis.title,
        )

        # 1. Checkpoint
        try:
            tag_name = self._create_checkpoint(hypothesis)
        except Exception as exc:
            hypothesis.status = "failed"
            return ExecutionResult(
                hypothesis_id=hypothesis.id,
                success=False,
                error=f"Checkpoint failed: {exc}",
            )

        # 2. Execute (placeholder -- real implementation will call router)
        try:
            before_metrics: dict[str, object] = {"placeholder": True}
            after_metrics: dict[str, object] = {"placeholder": True}
            improvement = 0.0
            cost = 0.0

            # Placeholder: simulate a successful execution
            await asyncio.sleep(0)  # yield control to event loop

            # 3. Validate
            if improvement < self._config.min_improvement_threshold:
                if self._config.auto_revert_on_regression:
                    try:
                        self._revert_to_checkpoint(tag_name)
                    except Exception as revert_exc:
                        self._logger.error(
                            "executor.revert_failed",
                            tag=tag_name,
                            error=str(revert_exc),
                        )
                hypothesis.status = "reverted"
                return ExecutionResult(
                    hypothesis_id=hypothesis.id,
                    success=False,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    improvement_pct=improvement,
                    cost_usd=cost,
                    error="Improvement below threshold",
                )

            # 4. Success
            hypothesis.status = "succeeded"
            self._logger.info(
                "executor.success",
                hypothesis=hypothesis.id,
                improvement=improvement,
                cost=cost,
            )
            return ExecutionResult(
                hypothesis_id=hypothesis.id,
                success=True,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_pct=improvement,
                cost_usd=cost,
            )

        except Exception as exc:
            # Auto-revert on any error
            if self._config.auto_revert_on_regression:
                try:
                    self._revert_to_checkpoint(tag_name)
                except Exception as revert_exc:
                    self._logger.error(
                        "executor.revert_failed",
                        tag=tag_name,
                        error=str(revert_exc),
                    )
            hypothesis.status = "failed"
            return ExecutionResult(
                hypothesis_id=hypothesis.id,
                success=False,
                error=str(exc),
            )


def _default_git_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command. Default implementation uses subprocess."""
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
