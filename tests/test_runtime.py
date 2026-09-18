from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest

from par.core.action import Action, ActionResult
from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.core.capability import Capability, RiskLevel
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import ParameterizedSkill, SkillRegistry
from par.robots.base import RobotInterface
from par.robots.mock import MockRobot
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


class _ScriptedPlanner(Planner):
    """Returns a fixed sequence of (skill_name, params), no LLM needed."""

    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


class _SlowRobot(RobotInterface):
    def __init__(self, delay: float) -> None:
        self._delay = delay

    def get_observation(self) -> Observation:
        return Observation(observation_id="o", timestamp=datetime.now(timezone.utc), source="mock")

    def execute(self, action: Action) -> ActionResult:
        time.sleep(self._delay)
        return ActionResult(
            action_id=action.action_id, success=True, message="done", completed_at=datetime.now(timezone.utc)
        )


def _registry_with_builtins() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def _profile(**overrides: Any) -> EnvironmentProfile:
    defaults: dict[str, Any] = dict(
        name="simulation",  # matches builtin_skills()' default Capability.env_profiles
        workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
        max_velocity=2.0,
        action_timeout_seconds=1.0,
        approval_required=False,
    )
    defaults.update(overrides)
    return EnvironmentProfile(**defaults)


def test_full_loop_observation_to_result(agent, robot):
    runtime = Runtime(agent, robot)
    result = runtime.run_once("pick the red_object")

    assert result is not None
    assert result.success is True
    assert agent.state.status == AgentStatus.EXECUTING
    assert agent.state.history[-1]["result"]["success"] is True


def test_run_task_stops_on_task_complete(robot: MockRobot):
    planner = _ScriptedPlanner(
        [
            ("pick", {"object": "red_object"}),
            ("place", {"target": "blue_container"}),
            (TASK_COMPLETE, {"message": "done"}),
        ]
    )
    agent = Agent(_registry_with_builtins(), planner=planner)
    runtime = Runtime(agent, robot)

    results = runtime.run_task("pick red_object and place in blue_container", max_steps=10)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert agent.state.status == AgentStatus.DONE


def test_run_task_stops_on_failure(robot: MockRobot):
    planner = _ScriptedPlanner(
        [
            ("pick", {"object": "nonexistent_object"}),
            ("place", {"target": "blue_container"}),
        ]
    )
    agent = Agent(_registry_with_builtins(), planner=planner)
    runtime = Runtime(agent, robot)

    results = runtime.run_task("pick nonexistent_object", max_steps=10)

    assert len(results) == 1
    assert results[0].success is False
    assert agent.state.status == AgentStatus.FAILED


def test_safety_kernel_denies_move_outside_workspace_and_task_continues(robot: MockRobot):
    planner = _ScriptedPlanner(
        [
            ("move", {"x": 100.0, "y": 0.0, "z": 0.0}),  # denied: outside workspace
            (TASK_COMPLETE, {"message": "done"}),
        ]
    )
    agent = Agent(_registry_with_builtins(), planner=planner)
    safety = SafetyKernel(_profile())
    runtime = Runtime(agent, robot, safety_kernel=safety)

    results = runtime.run_task("move far away", max_steps=5)

    assert len(results) == 1
    assert results[0].success is False
    assert "workspace" in results[0].message
    # A safety denial is recoverable - the loop kept going to task_complete,
    # it did not abort the way a genuine execution failure would.
    assert agent.state.status == AgentStatus.DONE


def test_safety_kernel_modifies_fast_move_before_execution(robot: MockRobot):
    planner = _ScriptedPlanner([("move", {"x": 0.9, "y": 0.0, "z": 0.0})])
    agent = Agent(_registry_with_builtins(), planner=planner)
    safety = SafetyKernel(_profile(max_velocity=0.5))
    runtime = Runtime(agent, robot, safety_kernel=safety)

    result = runtime.run_once("move")

    assert result is not None
    assert result.success is True
    final_x = robot.get_observation().robot_state["position"]["x"]
    assert final_x == pytest.approx(0.5, abs=0.05)  # clamped, not the original 0.9


def test_emergency_stop_blocks_all_actions(robot: MockRobot):
    planner = _ScriptedPlanner([("pick", {"object": "red_object"})])
    agent = Agent(_registry_with_builtins(), planner=planner)
    safety = SafetyKernel(_profile())
    safety.emergency_stop()
    runtime = Runtime(agent, robot, safety_kernel=safety)

    result = runtime.run_once("pick red_object")

    assert result is not None
    assert result.success is False
    assert "emergency stop" in result.message


def test_escalated_action_denied_by_default_human_approval(robot: MockRobot):
    registry = SkillRegistry()
    registry.register(
        ParameterizedSkill(
            Capability(name="pick", description="pick", risk=RiskLevel.HIGH, env_profiles=["simulation"]),
            required_params={"object"},
        )
    )
    planner = _ScriptedPlanner([("pick", {"object": "red_object"})])
    agent = Agent(registry, planner=planner)
    safety = SafetyKernel(_profile(approval_required=True))
    runtime = Runtime(agent, robot, safety_kernel=safety)

    result = runtime.run_once("pick red_object")

    assert result is not None
    assert result.success is False
    assert "human denied" in result.message


def test_action_timeout_produces_failed_result():
    planner = _ScriptedPlanner([("stop", {})])
    agent = Agent(_registry_with_builtins(), planner=planner)
    safety = SafetyKernel(_profile(action_timeout_seconds=0.05))
    runtime = Runtime(agent, _SlowRobot(delay=0.3), safety_kernel=safety)

    result = runtime.run_once("stop")

    assert result is not None
    assert result.success is False
    assert "timed out" in result.message
    runtime.close()
