"""Experiment 2: Four-Outcome Safety Kernel Evaluation (RQ2).

Does the four-outcome Safety Kernel (ALLOW, MODIFY, DENY, ESCALATE) give
correct decisions across the four action classes it's meant to distinguish?

Uses high_safety: the one shipped profile where approval_required=True
(needed for escalate) AND max_velocity*action_timeout < workspace bound
(needed for modify) hold simultaneously - see benchmarks.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from par.core.skill import SkillRegistry
from par.evaluation.benchmarks import EXPECTED_OUTCOMES, generate_four_outcome_scenarios
from par.evaluation.metrics import classification_accuracy, confusion_matrix
from par.evaluation.runner import evaluate_cases
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def run() -> dict:
    profile = load_profile("high_safety")
    kernel = SafetyKernel(profile)
    cases = generate_four_outcome_scenarios(profile)
    outcomes = evaluate_cases(cases, _registry(), kernel)

    expected = [o.case.expected_outcome for o in outcomes]
    actual = [o.actual_outcome for o in outcomes]

    return {
        "experiment": "2_four_outcome_evaluation",
        "profile": profile.name,
        "n_cases": len(outcomes),
        "outcome_classification_accuracy": classification_accuracy(expected, actual),
        "confusion_matrix": confusion_matrix(expected, actual, list(EXPECTED_OUTCOMES)),
        "cases": [
            {
                "category": o.case.category,
                "skill": o.case.skill_name,
                "parameters": o.case.parameters,
                "expected": o.case.expected_outcome,
                "actual": o.actual_outcome,
                "correct": o.correct,
                "reason": o.decision.reason if o.decision else None,
            }
            for o in outcomes
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp02_four_outcome_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
