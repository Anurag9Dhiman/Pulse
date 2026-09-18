from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.robots.base import RobotInterface


class MockRobot(RobotInterface):
    """In-memory robot for Week 1 development: no ROS 2, no hardware."""

    def __init__(self) -> None:
        self._position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._holding: str | None = None
        self._objects: dict[str, dict[str, Any]] = {
            "red_object": {"position": {"x": 0.5, "y": 0.2, "z": 0.0}},
            "blue_container": {"position": {"x": -0.3, "y": 0.4, "z": 0.0}},
        }

    def get_observation(self) -> Observation:
        return Observation(
            observation_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            source="mock",
            robot_state={"position": dict(self._position), "holding": self._holding},
            detections=[{"name": name, **info} for name, info in self._objects.items()],
        )

    def execute(self, action: Action) -> ActionResult:
        handler = getattr(self, f"_do_{action.skill_name}", None)
        if handler is None:
            return self._result(action, False, f"mock robot cannot execute '{action.skill_name}'")
        return handler(action)

    def _result(self, action: Action, success: bool, message: str) -> ActionResult:
        return ActionResult(
            action_id=action.action_id,
            success=success,
            message=message,
            completed_at=datetime.now(timezone.utc),
        )

    def _do_detect(self, action: Action) -> ActionResult:
        return self._result(action, True, f"detected {len(self._objects)} objects")

    def _do_move(self, action: Action) -> ActionResult:
        self._position = {axis: action.parameters.get(axis, 0.0) for axis in ("x", "y", "z")}
        return self._result(action, True, f"moved to {self._position}")

    def _do_pick(self, action: Action) -> ActionResult:
        target = action.parameters.get("object")
        if target not in self._objects:
            return self._result(action, False, f"object '{target}' not found")
        if self._holding is not None:
            return self._result(action, False, f"gripper already holding '{self._holding}'")
        self._holding = target
        return self._result(action, True, f"picked '{target}'")

    def _do_place(self, action: Action) -> ActionResult:
        if self._holding is None:
            return self._result(action, False, "gripper is empty")
        target = action.parameters.get("target")
        held = self._holding
        self._holding = None
        return self._result(action, True, f"placed '{held}' at '{target}'")

    def _do_stop(self, action: Action) -> ActionResult:
        return self._result(action, True, "stopped")

    def _do_inspect(self, action: Action) -> ActionResult:
        target = action.parameters.get("target")
        found = target in self._objects
        return self._result(action, found, f"inspected '{target}': {'found' if found else 'not found'}")
