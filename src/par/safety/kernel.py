from __future__ import annotations

import math

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.core.observation import Observation
from par.safety.environment import EnvironmentProfile
from par.safety.policy import PolicyOutcome, SafetyDecision


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

    def _check_move(self, action: Action, observation: Observation) -> SafetyDecision:
        target = (
            action.parameters.get("x", 0.0),
            action.parameters.get("y", 0.0),
            action.parameters.get("z", 0.0),
        )
        if not self.profile.workspace.contains(*target):
            return SafetyDecision(
                outcome=PolicyOutcome.DENY,
                reason=f"target {target} is outside workspace bounds "
                f"x={self.profile.workspace.x} y={self.profile.workspace.y} z={self.profile.workspace.z}",
            )

        current = observation.robot_state.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
        current_point = (current["x"], current["y"], current["z"])
        distance = math.dist(current_point, target)
        implied_speed = distance / self.profile.action_timeout_seconds

        if implied_speed > self.profile.max_velocity:
            max_distance = self.profile.max_velocity * self.profile.action_timeout_seconds
            scale = max_distance / distance if distance else 0.0
            clamped = {
                axis: current_point[i] + (target[i] - current_point[i]) * scale
                for i, axis in enumerate(("x", "y", "z"))
            }
            return SafetyDecision(
                outcome=PolicyOutcome.MODIFY,
                reason=f"implied speed {implied_speed:.2f} m/s exceeds max_velocity "
                f"{self.profile.max_velocity} m/s; clamped to reachable distance",
                modified_parameters=clamped,
            )

        return SafetyDecision(outcome=PolicyOutcome.ALLOW)
