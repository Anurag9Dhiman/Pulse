"""Environment-truth goal verifiers for Experiment 22 (planner-reported vs.
environment-verified success). A verifier takes the final Observation and
returns whether the goal was *actually* achieved, independent of anything
the planner said.
"""
from __future__ import annotations

from typing import Callable

from par.core.observation import Observation

GoalVerifier = Callable[[Observation], bool]


def verify_pick_and_place(object_name: str, container_name: str) -> GoalVerifier:
    """True iff `object_name`'s detection reports it's in `container_name` -
    set by MockRobot._do_place, not inferred from any ActionResult message."""

    def _verify(observation: Observation) -> bool:
        for detection in observation.detections:
            if detection.get("name") == object_name:
                return detection.get("container") == container_name
        return False

    return _verify


def verify_position_within(target: tuple[float, float, float], tolerance: float) -> GoalVerifier:
    """True iff the robot's final position is within `tolerance` of `target`."""
    import math

    def _verify(observation: Observation) -> bool:
        position = observation.robot_state.get("position")
        if not position:
            return False
        point = (position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0))
        return math.dist(point, target) <= tolerance

    return _verify
