from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from par.core.capability import Capability
from par.core.observation import Observation


class Decision(BaseModel):
    """A DecisionEngine's proposed choice over the bounded vocabulary given
    by SkillRegistry.capabilities(), with a confidence score. Deliberately
    not named to collide with the pre-existing SafetyDecision (a Safety
    Kernel outcome) - this is a System-1 proposal, produced before Safety
    ever sees the action."""

    skill_name: str | None = None  # None = "no proposal"; distinct from low
    # confidence, so a buggy engine can't get routed on confidence alone
    # without also naming a real skill.
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""  # short justification, for telemetry/debugging and later benchmarking


class DecisionEngine(Protocol):
    def decide(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> Decision:
        """Best-effort (skill_name, parameters) choice for this step, with a
        confidence score. Mirrors Planner.propose()'s argument shape so Agent
        can call either without either side knowing about the other; named
        `decide` rather than `propose` to keep "fast System-1 decision" and
        "slow System-2 proposal" distinguishable at call sites.

        Must always return a Decision, never raise: a low-confidence (or
        skill_name=None) Decision is the expected, routine way to say "defer
        to the Planner" - not an error condition requiring exception
        handling by the caller. Unlike Planner.propose() (which legitimately
        raises PlannerError when nothing matches), this contract never does;
        error handling for a real external engine (timeouts, bad responses)
        is deferred to the Jev-integration milestone.
        """
        ...


class RuleDecisionEngine:
    """Deterministic, local, zero-latency reference DecisionEngine - the
    "local floor" this milestone exists to establish before touching Jev.

    Deliberately narrow: confidently resolves only unambiguous, zero-
    parameter goals (a single capability name appears as a keyword and that
    capability takes no input parameters, e.g. "stop", "detect" among the
    builtin skills). Anything requiring parameter extraction, or an
    ambiguous/absent keyword match, gets confidence 0.0 so Agent falls
    through to the existing Planner - this is intentionally not a
    sophisticated router; the point is a measurable, trustworthy floor.
    """

    def decide(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> Decision:
        goal_lower = goal.lower()
        matches = [c for c in capabilities if c.name in goal_lower]
        if len(matches) == 1 and not matches[0].input_schema:
            capability = matches[0]
            return Decision(
                skill_name=capability.name,
                parameters={},
                confidence=1.0,
                reason=f"unambiguous zero-parameter match on '{capability.name}'",
            )
        return Decision(
            skill_name=None,
            parameters={},
            confidence=0.0,
            reason="no unambiguous zero-parameter capability match",
        )
