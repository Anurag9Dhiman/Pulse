from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from par.core.agent import Agent
from par.core.decision import Decision, RuleDecisionEngine
from par.core.observation import Observation
from par.core.planner import Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry


class _UnreachablePlanner(Planner):
    """Proves System 1 fully short-circuits System 2: any call is a bug."""

    def propose(self, goal, observation, capabilities):
        raise AssertionError("planner.propose() must not be called when the DecisionEngine is confident")


class _FixedDecisionEngine:
    """Always returns the same Decision, regardless of input - a DecisionEngine
    is a Protocol (structural typing), so this doesn't need to inherit from it."""

    def __init__(self, decision: Decision) -> None:
        self._decision = decision

    def decide(self, goal, observation, capabilities) -> Decision:
        return self._decision


class _ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


def _obs() -> Observation:
    return Observation(observation_id="o", timestamp=datetime.now(timezone.utc), source="mock")


def test_rule_decision_engine_confidently_resolves_unambiguous_zero_parameter_goal(skill_registry: SkillRegistry):
    capabilities = skill_registry.capabilities()
    decision = RuleDecisionEngine().decide("stop the robot now", _obs(), capabilities)
    assert decision.skill_name == "stop"
    assert decision.confidence == 1.0


def test_rule_decision_engine_defers_when_goal_requires_parameters(skill_registry: SkillRegistry):
    capabilities = skill_registry.capabilities()
    decision = RuleDecisionEngine().decide("pick the red_object", _obs(), capabilities)
    assert decision.skill_name is None
    assert decision.confidence == 0.0


def test_rule_decision_engine_defers_on_ambiguous_multi_capability_match(skill_registry: SkillRegistry):
    capabilities = skill_registry.capabilities()
    decision = RuleDecisionEngine().decide("detect then stop the robot", _obs(), capabilities)
    assert decision.skill_name is None
    assert decision.confidence == 0.0


def test_rule_decision_engine_defers_when_no_capability_matches(skill_registry: SkillRegistry):
    capabilities = skill_registry.capabilities()
    decision = RuleDecisionEngine().decide("xyzzy nonsense", _obs(), capabilities)
    assert decision.skill_name is None
    assert decision.confidence == 0.0


def test_agent_uses_decision_engine_choice_and_never_consults_planner_when_confident(
    skill_registry: SkillRegistry, robot
):
    decision_engine = _FixedDecisionEngine(Decision(skill_name="stop", confidence=1.0, reason="fixed"))
    agent = Agent(skill_registry, planner=_UnreachablePlanner(), decision_engine=decision_engine)
    agent.set_goal("stop the robot")

    action = agent.propose_action(robot.get_observation())

    assert action.skill_name == "stop"
    assert agent.last_decision is not None
    assert agent.last_decision.confidence == 1.0


def test_agent_falls_back_to_planner_when_decision_engine_confidence_below_threshold(
    skill_registry: SkillRegistry, robot
):
    decision_engine = _FixedDecisionEngine(Decision(skill_name=None, confidence=0.0, reason="fixed, not confident"))
    planner = _ScriptedPlanner([("pick", {"object": "red_object"})])
    agent = Agent(skill_registry, planner=planner, decision_engine=decision_engine)
    agent.set_goal("pick the red_object")

    action = agent.propose_action(robot.get_observation())

    assert action.skill_name == "pick"
    assert action.parameters == {"object": "red_object"}
    # The engine was consulted (and its verdict recorded) even though its
    # low-confidence answer wasn't acted on - distinguishes "no engine
    # configured" from "engine configured but deferred".
    assert agent.last_decision is not None
    assert agent.last_decision.confidence == 0.0


def test_agent_without_decision_engine_has_none_configured_and_unchanged_behavior(
    skill_registry: SkillRegistry, robot
):
    """Explicit regression anchor: Agent(skill_registry) with no decision_engine
    behaves exactly as it did before this milestone (see test_agent.py's
    test_agent_proposes_pick_action, asserted identically here)."""
    agent = Agent(skill_registry)
    assert agent.decision_engine is None

    agent.set_goal("pick the red_object")
    action = agent.propose_action(robot.get_observation())

    assert action.skill_name == "pick"
    assert action.parameters == {"object": "red_object"}
    assert agent.state.step_count == 1
    assert agent.last_decision is None


def test_runtime_executes_decision_engine_chosen_action_and_records_history_like_planner_chosen_one(
    skill_registry: SkillRegistry, robot
):
    """Integration test through the real, unmodified Runtime and Safety
    Kernel pipeline - proves the new seam works end-to-end, not just at the
    Agent unit level."""
    agent = Agent(skill_registry, decision_engine=RuleDecisionEngine())
    runtime = Runtime(agent, robot)

    result = runtime.run_once("stop the robot")

    assert result is not None
    assert result.success is True
    assert agent.state.history[-1]["action"]["skill_name"] == "stop"
    assert agent.last_decision is not None
    assert agent.last_decision.confidence == 1.0
