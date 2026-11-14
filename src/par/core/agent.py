from __future__ import annotations

from typing import Any
from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.agent_state import AgentState, AgentStatus
from par.core.capability import Capability
from par.core.decision import Decision, DecisionEngine
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner, RuleBasedPlanner
from par.core.skill import SkillRegistry


class AgentError(Exception):
    pass


class Agent:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        planner: Planner | None = None,
        agent_id: str | None = None,
        decision_engine: DecisionEngine | None = None,
        decision_confidence_threshold: float = 0.8,
    ) -> None:
        self.skill_registry = skill_registry
        self.planner = planner or RuleBasedPlanner()
        self.state = AgentState(agent_id=agent_id or str(uuid4()))
        self.decision_engine = decision_engine
        self.decision_confidence_threshold = decision_confidence_threshold
        self.last_decision: Decision | None = None  # most recent DecisionEngine
        # output, whether or not it was acted on - a plain Agent attribute,
        # not part of AgentState, so AgentState's own pydantic serialization
        # is unaffected by it.

    def set_goal(self, goal: str) -> None:
        self.state.goal = goal
        self.state.status = AgentStatus.PLANNING
        self.state.completion_message = None
        self.state.reported_success = None
        self.planner.reset()

    def propose_action(self, observation: Observation) -> Action | None:
        """Returns the next Action to take, or None if the planner signaled task completion."""
        if self.state.goal is None:
            raise AgentError("agent has no active goal")
        capabilities = self.skill_registry.capabilities()
        skill_name, parameters = self._propose(observation, capabilities)
        if skill_name == TASK_COMPLETE:
            self.state.status = AgentStatus.DONE
            self.state.completion_message = parameters.get("message")
            # Older scripted/test planners may omit "success"; absence means
            # "the planner didn't say otherwise", not "it failed" - default True.
            self.state.reported_success = bool(parameters.get("success", True))
            return None
        skill = self.skill_registry.get(skill_name)
        action = skill.build_action(parameters)
        self.state.step_count += 1
        return action

    def _propose(
        self, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        """Tries the DecisionEngine (System 1) first when configured and
        confident; otherwise falls through to the Planner (System 2) -
        exactly the pre-Decision-layer call when decision_engine is None,
        so Agent(registry, planner=X) with no decision_engine is unchanged."""
        self.last_decision = None
        if self.decision_engine is not None:
            decision = self.decision_engine.decide(self.state.goal, observation, capabilities)
            self.last_decision = decision
            if decision.skill_name is not None and decision.confidence >= self.decision_confidence_threshold:
                return decision.skill_name, decision.parameters
        return self.planner.propose(self.state.goal, observation, capabilities)

    def record_result(self, action: Action, result: ActionResult) -> None:
        self.state.history.append(
            {
                "action": action.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
            }
        )
        # A single successful step doesn't mean the task is done - only the
        # planner's explicit task_complete signal (see propose_action) does.
        self.state.status = AgentStatus.FAILED if not result.success else AgentStatus.EXECUTING

    def record_rejection(self, action: Action, reason: str) -> None:
        """A Safety Kernel denial is recoverable: the next loop step re-plans,
        so - unlike record_result() - this must not flip status to FAILED."""
        self.state.history.append(
            {
                "action": action.model_dump(mode="json"),
                "result": {"success": False, "message": reason, "rejected_by_safety": True},
            }
        )
