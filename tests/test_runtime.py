from __future__ import annotations

from typing import Any

from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.skills import builtin_skills


class _ScriptedPlanner(Planner):
    """Returns a fixed sequence of (skill_name, params), no LLM needed."""

    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


def _agent_with_registry() -> tuple[Agent, SkillRegistry]:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return Agent(registry), registry


def test_full_loop_observation_to_result(agent, robot):
    runtime = Runtime(agent, robot)
    result = runtime.run_once("pick the red_object")

    assert result is not None
    assert result.success is True
    assert agent.state.status == AgentStatus.EXECUTING
    assert agent.state.history[-1]["result"]["success"] is True


def test_run_task_stops_on_task_complete(robot: MockRobot):
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    planner = _ScriptedPlanner(
        [
            ("pick", {"object": "red_object"}),
            ("place", {"target": "blue_container"}),
            (TASK_COMPLETE, {"message": "done"}),
        ]
    )
    agent = Agent(registry, planner=planner)
    runtime = Runtime(agent, robot)

    results = runtime.run_task("pick red_object and place in blue_container", max_steps=10)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert agent.state.status == AgentStatus.DONE


def test_run_task_stops_on_failure(robot: MockRobot):
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    planner = _ScriptedPlanner(
        [
            ("pick", {"object": "nonexistent_object"}),
            ("place", {"target": "blue_container"}),
        ]
    )
    agent = Agent(registry, planner=planner)
    runtime = Runtime(agent, robot)

    results = runtime.run_task("pick nonexistent_object", max_steps=10)

    assert len(results) == 1
    assert results[0].success is False
    assert agent.state.status == AgentStatus.FAILED
