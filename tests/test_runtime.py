from par.core.agent_state import AgentStatus
from par.core.runtime import Runtime


def test_full_loop_observation_to_result(agent, robot):
    runtime = Runtime(agent, robot)
    result = runtime.run_once("pick the red_object")

    assert result.success is True
    assert agent.state.status == AgentStatus.DONE
    assert agent.state.history[-1]["result"]["success"] is True
