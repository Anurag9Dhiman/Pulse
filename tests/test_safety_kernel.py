from datetime import datetime, timezone

import pytest

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.core.observation import Observation
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome

_PROFILE = EnvironmentProfile(
    name="test",
    workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
    max_velocity=0.5,
    action_timeout_seconds=1.0,
    approval_required=False,
)

_MOVE_CAPABILITY = Capability(name="move", description="move", env_profiles=["test"])


def _observation(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Observation:
    return Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": position[0], "y": position[1], "z": position[2]}},
    )


def _move_action(x: float, y: float, z: float) -> Action:
    return Action(
        action_id="a1",
        skill_name="move",
        parameters={"x": x, "y": y, "z": z},
        created_at=datetime.now(timezone.utc),
    )


def test_admit_denies_capability_not_registered_for_profile():
    kernel = SafetyKernel(_PROFILE)
    other = Capability(name="fly", description="flies", env_profiles=["other_profile"])
    assert kernel.admit(other).outcome == PolicyOutcome.DENY


def test_admit_allows_registered_capability():
    kernel = SafetyKernel(_PROFILE)
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


def test_check_denies_move_outside_workspace():
    kernel = SafetyKernel(_PROFILE)
    action = _move_action(5.0, 0.0, 0.0)  # workspace x bound is [-1, 1]
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.DENY


def test_check_allows_slow_move_within_bounds():
    kernel = SafetyKernel(_PROFILE)  # max_velocity=0.5, timeout=1.0s -> max distance 0.5
    action = _move_action(0.3, 0.0, 0.0)
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.ALLOW


def test_check_modifies_move_exceeding_velocity_but_within_workspace():
    kernel = SafetyKernel(_PROFILE)
    action = _move_action(0.9, 0.0, 0.0)  # inside workspace, but distance 0.9 > max 0.5
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.MODIFY
    assert decision.modified_parameters is not None
    assert decision.modified_parameters["x"] == pytest.approx(0.5, abs=0.01)


def test_emergency_stop_denies_everything():
    kernel = SafetyKernel(_PROFILE)
    kernel.emergency_stop()
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.DENY
    action = _move_action(0.1, 0.0, 0.0)
    assert kernel.check(action, _observation(), _MOVE_CAPABILITY).outcome == PolicyOutcome.DENY


def test_clear_emergency_stop_restores_normal_operation():
    kernel = SafetyKernel(_PROFILE)
    kernel.emergency_stop()
    kernel.clear_emergency_stop()
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


def test_check_denies_move_that_collides_with_detected_object():
    kernel = SafetyKernel(_PROFILE)
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "obstacle_1", "position": {"x": 0.5, "y": 0.0, "z": 0.0}}],
    )
    action = _move_action(0.5, 0.0, 0.0)  # lands exactly on the obstacle
    decision = kernel.check(action, obs, _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.DENY
    assert "collision" in decision.reason


def test_check_allows_move_far_from_detected_objects():
    kernel = SafetyKernel(_PROFILE)
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "obstacle_1", "position": {"x": -0.9, "y": -0.9, "z": 0.0}}],
    )
    action = _move_action(0.3, 0.0, 0.0)  # far from the obstacle, within velocity limit
    decision = kernel.check(action, obs, _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.ALLOW


def test_check_escalates_high_risk_action_when_approval_required():
    profile = EnvironmentProfile(
        name="test-approval",
        workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
        max_velocity=0.5,
        action_timeout_seconds=1.0,
        approval_required=True,
    )
    kernel = SafetyKernel(profile)
    capability = Capability(name="pick", description="pick", risk=RiskLevel.HIGH, env_profiles=["test-approval"])
    action = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "red_object"},
        created_at=datetime.now(timezone.utc),
    )
    decision = kernel.check(action, _observation(), capability)
    assert decision.outcome == PolicyOutcome.ESCALATE
