from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.observation import Observation

TASK_COMPLETE = "task_complete"


class PlannerError(Exception):
    pass


class Planner(ABC):
    @abstractmethod
    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        """Return the (skill_name, parameters) to execute next, or (TASK_COMPLETE, {...})."""

    def reset(self) -> None:
        """Called when the agent starts a new goal. Stateless planners can ignore this."""

    def record_result(self, action: Action, result: ActionResult) -> None:
        """Called after each executed action. Stateless planners can ignore this."""


class RuleBasedPlanner(Planner):
    """Deterministic keyword-matching stand-in for the LLM planner."""

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        goal_lower = goal.lower()
        for capability in capabilities:
            if capability.name in goal_lower:
                return capability.name, self._extract_params(capability.name, goal)
        names = [c.name for c in capabilities]
        raise PlannerError(f"no skill in {names} matches goal '{goal}'")

    def _extract_params(self, name: str, goal: str) -> dict[str, Any]:
        target = goal.split()[-1].strip(".,!?")
        if name == "pick":
            return {"object": target}
        if name in ("place", "inspect"):
            return {"target": target}
        if name == "move":
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return {}
