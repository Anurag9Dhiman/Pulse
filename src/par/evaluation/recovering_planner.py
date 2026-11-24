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

import math
from typing import Any

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner
from par.evaluation.recovery_strategies import DetourStrategy, RecoveryStrategy


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


class GoalSeekingPlanner(Planner):
    """Pursues a fixed goal, re-planning around Safety Kernel denials.

    Unlike RecoveringMovePlanner, which stops at the first action that is
    permitted, this planner keeps going until the robot is actually *at* the
    goal. That distinction only becomes visible once task success is defined
    as reaching the goal rather than as moving without colliding: a strategy
    that retreats always produces a permitted action, and always stops short.

    The recovery logic is supplied as a strategy object rather than written
    in here, so the identical strategy can be run inside a recovery module
    instead (see FixedRecoveryRuntime) and the two architectures compared
    without also varying what recovery means.
    """

    def __init__(
        self,
        goal: tuple[float, float, float],
        strategy: RecoveryStrategy | None = None,
        tolerance: float = 0.05,
        max_denials: int = 8,
    ) -> None:
        self._goal = goal
        self._strategy = strategy or DetourStrategy()
        self._tolerance = tolerance
        self._max_denials = max_denials
        self.reset()

    def reset(self) -> None:
        self._pending: list[tuple[float, float, float]] = []
        self._last_denied = False
        self._last_message: str | None = None
        self.denial_count = 0

    def record_result(self, action: Action, result: ActionResult) -> None:
        self._last_denied = not result.success
        self._last_message = result.message
        if self._last_denied:
            self.denial_count += 1

    def _position(self, observation: Observation) -> tuple[float, float, float]:
        current = observation.robot_state.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
        return (current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0))

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        here = self._position(observation)

        if math.dist(here, self._goal) <= self._tolerance:
            return TASK_COMPLETE, {
                "message": f"reached goal after {self.denial_count} denial(s)",
                "success": True,
            }

        if self.denial_count >= self._max_denials:
            return TASK_COMPLETE, {
                "message": f"gave up after {self.denial_count} denials, last: {self._last_message}",
                "success": False,
            }

        if not self._last_denied:
            # Fresh situation (start, or a waypoint just reached): the goal
            # itself is always the first thing worth proposing.
            self._pending = []
            return "move", {"x": self._goal[0], "y": self._goal[1], "z": self._goal[2]}

        if not self._pending:
            # Denied from here: ask the strategy how to get around it.
            self._pending = list(self._strategy.candidates(self._goal, here, observation))
        if not self._pending:
            return TASK_COMPLETE, {
                "message": f"no recovery candidates left, last: {self._last_message}",
                "success": False,
            }

        candidate = self._pending.pop(0)
        return "move", {"x": candidate[0], "y": candidate[1], "z": candidate[2]}


class FixedGoalPlanner(Planner):
    """Proposes the same goal every step and never adapts to a denial.

    The counterpart to GoalSeekingPlanner for Experiment 4's recovery-module
    configuration: all recovery must come from the runtime, so that the
    comparison isolates where recovery lives rather than letting both sides
    re-plan.
    """

    def __init__(self, goal: tuple[float, float, float], tolerance: float = 0.05, max_steps: int = 8) -> None:
        self._goal = goal
        self._tolerance = tolerance
        self._max_steps = max_steps
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self.denial_count = 0

    def record_result(self, action: Action, result: ActionResult) -> None:
        if not result.success:
            self.denial_count += 1

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        current = observation.robot_state.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
        here = (current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0))
        if math.dist(here, self._goal) <= self._tolerance:
            return TASK_COMPLETE, {"message": "reached goal", "success": True}
        if self._steps >= self._max_steps:
            return TASK_COMPLETE, {"message": "exhausted steps short of goal", "success": False}
        self._steps += 1
        return "move", {"x": self._goal[0], "y": self._goal[1], "z": self._goal[2]}
