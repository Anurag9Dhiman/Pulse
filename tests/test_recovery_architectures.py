from __future__ import annotations

from typing import Any

from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.core.planner import Planner
from par.core.skill import SkillRegistry
from par.evaluation.recovery_architectures import FixedRecoveryRuntime, NoReplanRuntime
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


class _FixedMovePlanner(Planner):
    def __init__(self, target: tuple[float, float, float]) -> None:
        self._target = target

    def propose(self, goal, observation, capabilities) -> tuple[str, dict[str, Any]]:
        return "move", {"x": self._target[0], "y": self._target[1], "z": self._target[2]}


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def test_no_replan_runtime_marks_denial_as_terminal_failure():
    robot = MockRobot()
    robot.add_obstacle("obstacle", 1.0, 0.0, 0.0)
    agent = Agent(_registry(), planner=_FixedMovePlanner((1.0, 0.0, 0.0)))
    runtime = NoReplanRuntime(agent, robot, safety_kernel=SafetyKernel(load_profile("simulation")))

    result = runtime.run_once("move to obstacle")

    assert result is not None
    assert result.success is False
    assert agent.state.status == AgentStatus.FAILED
    runtime.close()


def test_no_replan_runtime_stops_run_task_loop_after_one_denial():
    robot = MockRobot()
    robot.add_obstacle("obstacle", 1.0, 0.0, 0.0)
    agent = Agent(_registry(), planner=_FixedMovePlanner((1.0, 0.0, 0.0)))
    runtime = NoReplanRuntime(agent, robot, safety_kernel=SafetyKernel(load_profile("simulation")))

    results = runtime.run_task("move to obstacle", max_steps=10)

    assert len(results) == 1  # loop stopped immediately, unlike the recoverable-denial default
    runtime.close()


def test_fixed_recovery_runtime_succeeds_by_scaling_down_the_move():
    robot = MockRobot()
    robot.add_obstacle("obstacle", 1.0, 0.0, 0.0)  # collision_margin=0.3 on "simulation"
    agent = Agent(_registry(), planner=_FixedMovePlanner((1.0, 0.0, 0.0)))
    runtime = FixedRecoveryRuntime(
        agent, robot, safety_kernel=SafetyKernel(load_profile("simulation")), max_recovery_attempts=3, backoff=0.5
    )

    result = runtime.run_once("move to obstacle")

    assert result is not None
    assert result.success is True  # attempt 1: 0.5 distance to obstacle > 0.3 margin -> clears it
    final_x = robot.get_observation().robot_state["position"]["x"]
    assert 0.0 < final_x < 1.0  # moved, but not all the way to the (unsafe) original target
    runtime.close()


def test_fixed_recovery_runtime_reports_failure_when_recovery_exhausted():
    robot = MockRobot()
    # Obstacle placed exactly at the origin means every scaled-down retry
    # (which shrinks toward the origin) stays within the collision margin.
    robot.add_obstacle("obstacle", 0.0, 0.0, 0.0)
    agent = Agent(_registry(), planner=_FixedMovePlanner((0.2, 0.0, 0.0)))
    runtime = FixedRecoveryRuntime(
        agent, robot, safety_kernel=SafetyKernel(load_profile("simulation")), max_recovery_attempts=3, backoff=0.5
    )

    result = runtime.run_once("move near the obstacle")

    assert result is not None
    assert result.success is False
    assert "RecoveryModule" in result.message
    assert "exhausted" in result.message
    runtime.close()
