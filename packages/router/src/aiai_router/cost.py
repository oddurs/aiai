"""Cost tracking and budget enforcement."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from aiai_core.logging import JSONLLogger
from aiai_core.types import CostRecord, RouteResponse

if TYPE_CHECKING:
    from aiai_core.config import CostConfig


class BudgetExceededError(Exception):
    """Raised when spending exceeds the daily budget."""


class CostTracker:
    """Tracks API costs, logs them, and enforces daily budget limits.

    Maintains an in-memory running total that resets at the start of
    each new UTC day. Logs each cost event as a CostRecord to JSONL.
    """

    def __init__(
        self,
        config: CostConfig,
        logger: JSONLLogger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or JSONLLogger(path=config.log_file)
        self._daily_total: float = 0.0
        self._current_date: date = datetime.now(UTC).date()

    def _check_date_reset(self) -> None:
        """Reset daily total if the UTC date has changed."""
        today = datetime.now(UTC).date()
        if today != self._current_date:
            self._current_date = today
            self._daily_total = 0.0

    def record(self, response: RouteResponse, task_id: str = "") -> CostRecord:
        """Record a cost event from a model response.

        Logs the cost to JSONL, updates the daily total, and checks
        budget limits.

        Args:
            response: The route response containing cost and token info.
            task_id: Optional task identifier for attribution.

        Returns:
            The CostRecord that was logged.

        Raises:
            BudgetExceededError: If the daily budget has been exceeded.
        """
        self._check_date_reset()
        self._daily_total += response.cost_usd

        cost_record = CostRecord(
            model=response.model,
            tier=response.tier.value,
            task_id=task_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            daily_total_usd=self._daily_total,
        )

        self._logger.info("cost.record", **cost_record.to_dict())

        if response.cost_usd >= self._config.warn_threshold_usd:
            self._logger.warn(
                "cost.high_request",
                model=response.model,
                cost_usd=response.cost_usd,
                threshold_usd=self._config.warn_threshold_usd,
            )

        if self._daily_total > self._config.daily_budget_usd:
            raise BudgetExceededError(
                f"Daily budget exceeded: ${self._daily_total:.2f} > "
                f"${self._config.daily_budget_usd:.2f}"
            )

        return cost_record

    def daily_total(self) -> float:
        """Return the current daily spending total in USD."""
        self._check_date_reset()
        return self._daily_total

    def budget_remaining(self) -> float:
        """Return the remaining daily budget in USD."""
        self._check_date_reset()
        return max(0.0, self._config.daily_budget_usd - self._daily_total)
