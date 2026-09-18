import pytest

from par.core.agent import Agent, AgentError


def test_agent_requires_goal_before_proposing(agent: Agent, robot):
    obs = robot.get_observation()
    with pytest.raises(AgentError):
        agent.propose_action(obs)


def test_agent_proposes_pick_action(agent: Agent, robot):
    agent.set_goal("pick the red_object")
    obs = robot.get_observation()
    action = agent.propose_action(obs)
    assert action.skill_name == "pick"
    assert action.parameters == {"object": "red_object"}
    assert agent.state.step_count == 1
