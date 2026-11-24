from __future__ import annotations

import math

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.core.observation import Observation
from par.safety.environment import EnvironmentProfile
from par.safety.policy import PolicyOutcome, SafetyDecision

Point = tuple[float, float, float]


def segment_point_distance(start: Point, end: Point, point: Point) -> float:
    """Shortest distance from `point` to the *segment* [start, end].

    Clamping the projection parameter to [0, 1] is what makes this the
    distance to the segment rather than to the infinite line through it: an
    obstacle that lies beyond either endpoint measures to that endpoint, not
    to a point the robot never reaches.
    """
    segment = tuple(end[i] - start[i] for i in range(3))
    length_squared = sum(component * component for component in segment)
    if length_squared == 0.0:  # degenerate move: the segment is a single point
        return math.dist(start, point)
    offset = tuple(point[i] - start[i] for i in range(3))
    projection = sum(offset[i] * segment[i] for i in range(3)) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = tuple(start[i] + projection * segment[i] for i in range(3))
    return math.dist(closest, point)


class SafetyKernel:
    """Two-stage governance: Admission ("may this capability be considered?")
    then Policy Check ("under what constraints may it execute now?")."""

    def __init__(self, profile: EnvironmentProfile) -> None:
        self.profile = profile
        self._emergency_stopped = False

    def emergency_stop(self) -> None:
        self._emergency_stopped = True

    def clear_emergency_stop(self) -> None:
        self._emergency_stopped = False

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    def admit(self, capability: Capability) -> SafetyDecision:
        if self._emergency_stopped:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason="emergency stop engaged")
        if self.profile.name not in capability.env_profiles:
            return SafetyDecision(
                outcome=PolicyOutcome.DENY,
                reason=f"capability '{capability.name}' not registered for profile '{self.profile.name}'",
            )
        return SafetyDecision(outcome=PolicyOutcome.ALLOW)

    def check(self, action: Action, observation: Observation, capability: Capability) -> SafetyDecision:
        if self._emergency_stopped:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason="emergency stop engaged")

        if capability.risk == RiskLevel.HIGH and self.profile.approval_required:
            return SafetyDecision(
                outcome=PolicyOutcome.ESCALATE, reason="high-risk action requires human approval"
            )

        if action.skill_name == "move":
            return self._check_move(action, observation)

        return SafetyDecision(outcome=PolicyOutcome.ALLOW)

    # ---------------------------------------------------------------- move

    def _check_move(self, action: Action, observation: Observation) -> SafetyDecision:
        origin = self._current_position(observation)
        target = self._target_of(action)
        if target is None:
            # A planner is untrusted input: a language model will sometimes
            # emit "1.0" instead of 1.0, or a null. The kernel must return a
            # verdict for that, not raise - an exception here propagates out
            # of the governance layer and no decision is recorded at all.
            return SafetyDecision(
                outcome=PolicyOutcome.DENY,
                reason=f"invalid move parameters: {action.parameters!r} "
                "(x, y and z must be numbers)",
            )

        violation = self._validate_path(origin, target, observation)
        if violation is not None:
            return violation

        clamp = self._check_velocity(origin, target)
        if clamp is None:
            return SafetyDecision(outcome=PolicyOutcome.ALLOW)

        # Re-validate what MODIFY would actually execute. The clamped target
        # lies on the already-validated segment, so for a convex workspace
        # and the swept collision test above this cannot fail - except when
        # the robot is already outside the workspace, where `origin` itself
        # is unsafe and no sub-segment of it is validated. Checking rather
        # than asserting keeps MODIFY sound if a future check is added that
        # is not closed under taking sub-segments.
        clamped = clamp.modified_parameters or {}
        clamped_target = (clamped.get("x", 0.0), clamped.get("y", 0.0), clamped.get("z", 0.0))
        residual = self._validate_path(origin, clamped_target, observation)
        if residual is not None:
            return SafetyDecision(
                outcome=PolicyOutcome.DENY,
                reason=f"clamped target {clamped_target} is still unsafe: {residual.reason}",
            )
        return clamp

    @staticmethod
    def _target_of(action: Action) -> Point | None:
        """The requested target, or None if the parameters are not numeric.
        Numeric strings are accepted and coerced: rejecting "1.0" outright
        would deny a well-meant action over formatting, but it must never
        reach a comparison as a string."""
        coerced = []
        for axis in ("x", "y", "z"):
            value = action.parameters.get(axis, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return None
            try:
                coerced.append(float(value))
            except (TypeError, ValueError):
                return None
        return (coerced[0], coerced[1], coerced[2])

    def _validate_path(
        self, origin: Point, target: Point, observation: Observation
    ) -> SafetyDecision | None:
        """The checks a proposed motion must pass to be executable at all.
        Both are closed under shortening the path, which is what lets MODIFY
        clamp a target without reopening either question."""
        workspace_violation = self._check_workspace(target)
        if workspace_violation is not None:
            return workspace_violation
        return self._check_collision(origin, target, observation)

    def _current_position(self, observation: Observation) -> Point:
        current = observation.robot_state.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
        return (current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0))

    def _check_collision(
        self, origin: Point, target: Point, observation: Observation
    ) -> SafetyDecision | None:
        """Flags a move whose straight-line path passes within
        collision_margin of any detected object - including one that appeared
        after the plan was made, which is what turns a Safety Kernel denial
        into a genuine re-planning trigger (see Runtime.run_task and Use Case
        4 in the PAR spec).

        The test sweeps the whole segment, not just the endpoint: a target
        clear of every obstacle can still only be reached by driving through
        one. This bounds straight-line Cartesian motion only; a manipulator
        whose end effector follows a joint-space arc traverses points this
        segment does not cover.
        """
        for detection in observation.detections:
            position = detection.get("position")
            if not position:
                continue
            obstacle_point = (position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0))
            clearance = segment_point_distance(origin, target, obstacle_point)
            if clearance < self.profile.collision_margin:
                return SafetyDecision(
                    outcome=PolicyOutcome.DENY,
                    reason=f"collision risk: path {origin} -> {target} passes within "
                    f"{clearance:.3f}m of detected object '{detection.get('name')}' "
                    f"(margin {self.profile.collision_margin}m)",
                )
        return None

    def _check_workspace(self, target: Point) -> SafetyDecision | None:
        if self.profile.workspace.contains(*target):
            return None
        return SafetyDecision(
            outcome=PolicyOutcome.DENY,
            reason=f"target {target} is outside workspace bounds "
            f"x={self.profile.workspace.x} y={self.profile.workspace.y} z={self.profile.workspace.z}",
        )

    def _check_velocity(self, origin: Point, target: Point) -> SafetyDecision | None:
        distance = math.dist(origin, target)
        implied_speed = distance / self.profile.action_timeout_seconds

        if implied_speed <= self.profile.max_velocity:
            return None

        max_distance = self.profile.max_velocity * self.profile.action_timeout_seconds
        scale = max_distance / distance if distance else 0.0
        clamped = {
            axis: origin[i] + (target[i] - origin[i]) * scale
            for i, axis in enumerate(("x", "y", "z"))
        }
        return SafetyDecision(
            outcome=PolicyOutcome.MODIFY,
            reason=f"implied speed {implied_speed:.2f} m/s exceeds max_velocity "
            f"{self.profile.max_velocity} m/s; clamped to reachable distance",
            modified_parameters=clamped,
        )
