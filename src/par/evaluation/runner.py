from __future__ import annotations

from dataclasses import dataclass

from par.core.skill import SkillError, SkillRegistry
from par.evaluation.benchmarks import BenchmarkCase
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome, SafetyDecision


@dataclass
class CaseOutcome:
    case: BenchmarkCase
    actual_outcome: str
    decision: SafetyDecision | None
    correct: bool


def evaluate_case(case: BenchmarkCase, registry: SkillRegistry, safety_kernel: SafetyKernel) -> CaseOutcome:
    """Runs one benchmark case through the real pipeline: Skill.build_action
    (parameter validation) -> Admission -> Policy Guard. Per-case env_profiles
    / risk overrides are applied to a copy of the skill's capability, never
    mutating the shared registry between cases."""
    skill = registry.get(case.skill_name)
    capability = skill.capability
    overrides = {}
    if case.env_profiles is not None:
        overrides["env_profiles"] = case.env_profiles
    if case.risk is not None:
        overrides["risk"] = case.risk
    if overrides:
        capability = capability.model_copy(update=overrides)

    try:
        action = skill.build_action(case.parameters)
    except SkillError:
        actual = "invalid_parameters"
        return CaseOutcome(case=case, actual_outcome=actual, decision=None, correct=actual == case.expected_outcome)

    admission = safety_kernel.admit(capability)
    if admission.outcome != PolicyOutcome.ALLOW:
        actual = admission.outcome.value
        return CaseOutcome(case=case, actual_outcome=actual, decision=admission, correct=actual == case.expected_outcome)

    decision = safety_kernel.check(action, case.observation, capability)
    actual = decision.outcome.value
    return CaseOutcome(case=case, actual_outcome=actual, decision=decision, correct=actual == case.expected_outcome)


def evaluate_cases(
    cases: list[BenchmarkCase], registry: SkillRegistry, safety_kernel: SafetyKernel
) -> list[CaseOutcome]:
    return [evaluate_case(case, registry, safety_kernel) for case in cases]
