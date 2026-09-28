from __future__ import annotations

import pytest

from par.core.skill import SkillRegistry
from par.evaluation.benchmarks import (
    excess_velocity_target,
    generate_capability_admission_benchmark,
    generate_four_outcome_scenarios,
    generate_safety_interception_benchmark,
)
from par.evaluation.runner import evaluate_cases
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def test_excess_velocity_target_raises_when_workspace_is_the_tighter_constraint():
    # Both shipped baseline profiles have this property - documented in
    # benchmarks.py's module docstring, not just an incidental test detail.
    for name in ("simulation", "real_robot"):
        with pytest.raises(ValueError, match="workspace bound"):
            excess_velocity_target(load_profile(name))


def test_safety_interception_benchmark_matches_expected_outcomes():
    # strict_simulation: roomy workspace, low max_velocity - the only shipped
    # profile where an in-workspace excess-velocity case is constructible.
    profile = load_profile("strict_simulation")
    kernel = SafetyKernel(profile)
    cases = generate_safety_interception_benchmark(profile, seed=1, n_safe=10, n_unsafe=10)
    outcomes = evaluate_cases(cases, _registry(), kernel)

    accuracy = sum(1 for o in outcomes if o.correct) / len(outcomes)
    mismatches = [(o.case.category, o.case.parameters, o.actual_outcome) for o in outcomes if not o.correct]
    assert accuracy == 1.0, f"benchmark/runner disagree on: {mismatches}"

    categories = {o.case.category for o in outcomes}
    assert categories == {
        "safe_move",
        "workspace_violation",
        "collision",
        "excess_velocity",
        "unavailable_capability",
        "invalid_parameters",
    }


def test_safety_interception_benchmark_is_reproducible_given_same_seed():
    profile = load_profile("strict_simulation")
    a = generate_safety_interception_benchmark(profile, seed=42)
    b = generate_safety_interception_benchmark(profile, seed=42)
    assert [c.parameters for c in a] == [c.parameters for c in b]
    assert [c.category for c in a] == [c.category for c in b]


def test_four_outcome_scenarios_hit_all_four_outcomes_on_high_safety_profile():
    # high_safety is the one shipped profile where both preconditions hold at
    # once: approval_required=True (needed for escalate) AND
    # max_velocity*action_timeout < workspace bound (needed for modify).
    profile = load_profile("high_safety")
    kernel = SafetyKernel(profile)
    cases = generate_four_outcome_scenarios(profile)
    outcomes = evaluate_cases(cases, _registry(), kernel)

    actual = {o.case.category: o.actual_outcome for o in outcomes}
    assert actual == {
        "clearly_safe": "allow",
        "soft_violation_modifiable": "modify",
        "hard_violation": "deny",
        "high_risk_needs_approval": "escalate",
    }


def test_capability_admission_benchmark():
    kernel = SafetyKernel(load_profile("simulation"))
    cases = generate_capability_admission_benchmark(
        registered_profile="simulation", other_profile="not_simulation", n_trials=10, seed=7
    )
    outcomes = evaluate_cases(cases, _registry(), kernel)

    assert all(o.correct for o in outcomes)
    registered = [o for o in outcomes if o.case.category == "registered"]
    unregistered = [o for o in outcomes if o.case.category == "unregistered"]
    assert all(o.actual_outcome == "allow" for o in registered)
    assert all(o.actual_outcome == "deny" for o in unregistered)
