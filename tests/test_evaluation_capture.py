from __future__ import annotations

from par.core.agent import Agent
from par.core.runtime import Runtime
from par.evaluation.capture import CapturingTelemetryLogger
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


def test_capturing_logger_records_events_and_outcome_counts(robot):
    from par.core.skill import SkillRegistry

    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    agent = Agent(registry)
    profile = EnvironmentProfile(
        name="simulation",
        workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
        max_velocity=2.0,
        action_timeout_seconds=1.0,
    )
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(profile), telemetry=telemetry)

    runtime.run_once("pick the red_object")

    assert len(telemetry.events) == 1
    assert telemetry.events[0].skill == "pick"
    assert telemetry.outcome_counts()["allow"] == 1
    assert len(telemetry.latencies()) == 1
    assert telemetry.latencies()[0] >= 0.0
    runtime.close()
