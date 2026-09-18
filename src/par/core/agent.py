from __future__ import annotations

from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.agent_state import AgentState, AgentStatus
from par.core.observation import Observation
from par.core.planner import Planner, RuleBasedPlanner
from par.core.skill import SkillRegistry


class AgentError(Exception):
    pass


class Agent:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        planner: Planner | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.skill_registry = skill_registry
        self.planner = planner or RuleBasedPlanner()
        self.state = AgentState(agent_id=agent_id or str(uuid4()))

    def set_goal(self, goal: str) -> None:
        self.state.goal = goal
        self.state.status = AgentStatus.PLANNING

    def propose_action(self, observation: Observation) -> Action:
        if self.state.goal is None:
            raise AgentError("agent has no active goal")
        skill_name, parameters = self.planner.propose(
            self.state.goal, observation, self.skill_registry.list_names()
        )
        skill = self.skill_registry.get(skill_name)
        action = skill.build_action(parameters)
        self.state.step_count += 1
        return action

    def record_result(self, action: Action, result: ActionResult) -> None:
        self.state.history.append(
            {
                "action": action.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
            }
        )
        self.state.status = AgentStatus.DONE if result.success else AgentStatus.FAILED
