from __future__ import annotations

from par.core.agent import Agent
from par.core.capability import Capability, RiskLevel
from par.core.planner import Planner
from par.core.runtime import Runtime
from par.core.skill import ParameterizedSkill, SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel


def _profile(**overrides) -> EnvironmentProfile:
    defaults = dict(
        name="simulation",
        workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
        max_velocity=2.0,
        action_timeout_seconds=1.0,
        approval_required=True,
    )
    defaults.update(overrides)
    return EnvironmentProfile(**defaults)


def _high_risk_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(
        ParameterizedSkill(
            Capability(name="pick", description="pick", risk=RiskLevel.HIGH, env_profiles=["simulation"]),
            required_params={"object"},
        )
    )
    return registry


def test_telemetry_marks_escalated_even_when_resolved_to_allow(robot):
    # Regression: Runtime._step reassigns `decision` to the *resolved*
    # outcome before logging, so safety_decision alone can never show
    # "escalate" - it always shows what escalation resolved to (allow/deny).
    # escalated must be tracked separately, from the *original* decision.
    class _PickPlanner(Planner):
        def propose(self, goal, observation, capabilities):
            return "pick", {"object": "red_object"}

    agent = Agent(_high_risk_registry(), planner=_PickPlanner())
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(
        agent, robot, safety_kernel=SafetyKernel(_profile()), telemetry=telemetry,
        human_approval=lambda action, reason: True,  # approved -> resolves to ALLOW
    )

    result = runtime.run_once("pick red_object")

    assert result is not None
    assert result.success is True
    assert telemetry.events[0].safety_decision == "allow"  # resolved outcome
    assert telemetry.events[0].escalated is True  # but it WAS an escalation
    runtime.close()


def test_telemetry_does_not_mark_escalated_for_plain_allow(robot, agent):
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(_profile(approval_required=False)), telemetry=telemetry)

    runtime.run_once("pick the red_object")

    assert telemetry.events[0].safety_decision == "allow"
    assert telemetry.events[0].escalated is False
    runtime.close()
