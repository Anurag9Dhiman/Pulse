"""A deterministic, rule-based planner that can genuinely react to a Safety
Kernel denial - for Experiments 3, 4, 7, 10, 11, which need many trials to be
statistically meaningful. Live LLM re-planning is already validated
qualitatively (paper Section VI-B); running that many times over would mean
spending free-tier API quota to reconfirm a result already established, not
learning something new. This planner exists to get real trial volume on the
*mechanism* (does feedback-driven re-planning recover at all, how often, how
many attempts) without an API dependency.

Deliberately simple: nudge the target diagonally away from the origin by an
increasing step each time, on the theory that whatever caused the denial
(an obstacle, a boundary) is somewhere between origin and the original
target, so moving further along a different bearing is more likely to clear
it. This is not a real path-planning algorithm and won't correctly recover
from every scenario - it's a controlled way to generate statistically
meaningful data about the DENY -> feedback -> re-plan loop itself, not a
claim about planning quality.
"""
from __future__ import annotations

from typing import Any

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner


class RecoveringMovePlanner(Planner):
    def __init__(self, target: tuple[float, float, float], max_attempts: int = 5, step: float = 0.15) -> None:
        self._original_target = target
        self._max_attempts = max_attempts
        self._step = step
        self._attempts = 0
        self._last_failed = False
        self._last_message: str | None = None
        self.denial_count = 0  # for RecoveryRate / #replanning-steps reporting

    def reset(self) -> None:
        self._attempts = 0
        self._last_failed = False
        self._last_message = None
        self.denial_count = 0

    def record_result(self, action: Action, result: ActionResult) -> None:
        self._last_failed = not result.success
        self._last_message = result.message
        if self._last_failed:
            self.denial_count += 1

    def _candidate(self, attempt: int) -> tuple[float, float, float]:
        if attempt == 0:
            return self._original_target
        offset = self._step * attempt
        return (
            self._original_target[0] + offset,
            self._original_target[1] + offset,
            self._original_target[2],
        )

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        if self._last_failed and self._attempts >= self._max_attempts:
            return TASK_COMPLETE, {
                "message": f"gave up after {self._attempts} attempts, last denial: {self._last_message}",
                "success": False,
            }
        if self._attempts > 0 and not self._last_failed:
            return TASK_COMPLETE, {"message": f"reached target after {self._attempts} attempt(s)", "success": True}

        target = self._candidate(self._attempts)
        self._attempts += 1
        return "move", {"x": target[0], "y": target[1], "z": target[2]}
