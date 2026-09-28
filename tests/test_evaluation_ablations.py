from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.core.observation import Observation
from par.evaluation.ablations import (
    NoAdmissionKernel,
    NoCollisionCheckKernel,
    NoEscalateKernel,
    NoModifyKernel,
    NoPolicyGuardKernel,
    NoVelocityCheckKernel,
    NoWorkspaceCheckKernel,
)
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.policy import PolicyOutcome

_PROFILE = EnvironmentProfile(
    name="test",
    workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
    max_velocity=0.5,
    action_timeout_seconds=1.0,
    approval_required=True,
)


def _obs(detections=None) -> Observation:
    return Observation(
        observation_id="o",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=detections or [],
    )


def _move(x: float, y: float = 0.0, z: float = 0.0) -> Action:
    return Action(action_id="a", skill_name="move", parameters={"x": x, "y": y, "z": z}, created_at=datetime.now(timezone.utc))


def test_no_admission_kernel_always_admits_unregistered_capability():
    kernel = NoAdmissionKernel(_PROFILE)
    other = Capability(name="fly", description="flies", env_profiles=["some_other_profile"])
    assert kernel.admit(other).outcome == PolicyOutcome.ALLOW


def test_no_workspace_check_allows_out_of_bounds_move():
    kernel = NoWorkspaceCheckKernel(_PROFILE)
    capability = Capability(name="move", description="move", env_profiles=["test"])
    decision = kernel.check(_move(100.0), _obs(), capability)
    assert decision.outcome != PolicyOutcome.DENY  # workspace can't deny it; velocity might still MODIFY


def test_no_collision_check_allows_move_onto_obstacle():
    kernel = NoCollisionCheckKernel(_PROFILE)
    capability = Capability(name="move", description="move", env_profiles=["test"])
    obs = _obs(detections=[{"name": "obstacle", "position": {"x": 0.3, "y": 0.0, "z": 0.0}}])
    decision = kernel.check(_move(0.3), obs, capability)
    assert decision.outcome == PolicyOutcome.ALLOW  # would DENY (collision) under the real kernel


def test_no_velocity_check_allows_fast_move_unmodified():
    kernel = NoVelocityCheckKernel(_PROFILE)  # max reachable distance = 0.5
    capability = Capability(name="move", description="move", env_profiles=["test"])
    decision = kernel.check(_move(0.9), _obs(), capability)  # in workspace, exceeds velocity budget
    assert decision.outcome == PolicyOutcome.ALLOW
    assert decision.modified_parameters is None


def test_no_modify_denies_what_would_have_been_clamped():
    kernel = NoModifyKernel(_PROFILE)
    capability = Capability(name="move", description="move", env_profiles=["test"])
    decision = kernel.check(_move(0.9), _obs(), capability)  # real kernel would MODIFY this
    assert decision.outcome == PolicyOutcome.DENY


def test_no_escalate_runs_high_risk_action_unmodified_if_otherwise_safe():
    kernel = NoEscalateKernel(_PROFILE)  # approval_required=True on this profile
    capability = Capability(name="pick", description="pick", risk=RiskLevel.HIGH, env_profiles=["test"])
    action = Action(action_id="a", skill_name="pick", parameters={"object": "x"}, created_at=datetime.now(timezone.utc))
    decision = kernel.check(action, _obs(), capability)
    assert decision.outcome == PolicyOutcome.ALLOW  # real kernel would ESCALATE


def test_no_policy_guard_allows_everything_admission_lets_through():
    kernel = NoPolicyGuardKernel(_PROFILE)
    capability = Capability(name="move", description="move", env_profiles=["test"])
    # Even a wildly out-of-bounds move: Policy Guard is skipped entirely.
    decision = kernel.check(_move(1000.0), _obs(), capability)
    assert decision.outcome == PolicyOutcome.ALLOW


def test_ablated_kernels_still_respect_emergency_stop():
    for cls in (NoAdmissionKernel, NoWorkspaceCheckKernel, NoVelocityCheckKernel, NoModifyKernel, NoEscalateKernel, NoPolicyGuardKernel):
        kernel = cls(_PROFILE)
        kernel.emergency_stop()
        capability = Capability(name="move", description="move", env_profiles=["test"])
        assert kernel.check(_move(0.0), _obs(), capability).outcome == PolicyOutcome.DENY, f"{cls.__name__} ignored emergency stop"
