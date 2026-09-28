from __future__ import annotations

from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.core.planner import TASK_COMPLETE, Planner
from par.core.skill import SkillRegistry
from par.skills import builtin_skills


class _CompletionPlanner(Planner):
    def __init__(self, params: dict) -> None:
        self._params = params

    def propose(self, goal, observation, capabilities):
        return TASK_COMPLETE, self._params


def _agent(params: dict) -> Agent:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return Agent(registry, planner=_CompletionPlanner(params))


def test_task_complete_captures_message_and_explicit_success(robot):
    agent = _agent({"message": "done", "success": True})
    agent.set_goal("do it")
    action = agent.propose_action(robot.get_observation())

    assert action is None
    assert agent.state.status == AgentStatus.DONE
    assert agent.state.completion_message == "done"
    assert agent.state.reported_success is True


def test_task_complete_captures_explicit_failure(robot):
    agent = _agent({"message": "blocked by obstacle, no safe alternative", "success": False})
    agent.set_goal("do it")
    agent.propose_action(robot.get_observation())

    assert agent.state.reported_success is False
    assert "blocked" in agent.state.completion_message


def test_task_complete_defaults_success_true_when_omitted(robot):
    # Backward compatibility: scripted planners/tests written before Experiment
    # 22 only ever passed {"message": ...}. Absence must not look like failure.
    agent = _agent({"message": "done"})
    agent.set_goal("do it")
    agent.propose_action(robot.get_observation())

    assert agent.state.reported_success is True


def test_set_goal_resets_completion_state(robot):
    agent = _agent({"message": "done", "success": True})
    agent.set_goal("first")
    agent.propose_action(robot.get_observation())
    assert agent.state.reported_success is True

    agent.set_goal("second")
    assert agent.state.completion_message is None
    assert agent.state.reported_success is None
