from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from par.core.observation import Observation


class PlannerError(Exception):
    pass


class Planner(ABC):
    @abstractmethod
    def propose(
        self, goal: str, observation: Observation, skill_names: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """Return the (skill_name, parameters) to execute next."""


class RuleBasedPlanner(Planner):
    """Deterministic keyword-matching stand-in for the Week 2 LLM planner."""

    def propose(
        self, goal: str, observation: Observation, skill_names: list[str]
    ) -> tuple[str, dict[str, Any]]:
        goal_lower = goal.lower()
        for name in skill_names:
            if name in goal_lower:
                return name, self._extract_params(name, goal)
        raise PlannerError(f"no skill in {skill_names} matches goal '{goal}'")

    def _extract_params(self, name: str, goal: str) -> dict[str, Any]:
        target = goal.split()[-1].strip(".,!?")
        if name == "pick":
            return {"object": target}
        if name in ("place", "inspect"):
            return {"target": target}
        if name == "move":
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return {}
