"""Experiment 12: Capability Admission Evaluation.

Does Admission reliably prevent execution of capabilities not registered for
the current deployment environment?
"""
from __future__ import annotations

import json
from pathlib import Path

from par.core.skill import SkillRegistry
from par.evaluation.benchmarks import generate_capability_admission_benchmark
from par.evaluation.metrics import rate
from par.evaluation.runner import evaluate_cases
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

SEEDS = list(range(10))
N_TRIALS_PER_SEED = 20


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def run() -> dict:
    profile = load_profile("simulation")
    kernel = SafetyKernel(profile)
    per_seed = []

    for seed in SEEDS:
        cases = generate_capability_admission_benchmark(
            registered_profile=profile.name, other_profile="restricted_profile", n_trials=N_TRIALS_PER_SEED, seed=seed
        )
        outcomes = evaluate_cases(cases, _registry(), kernel)

        unregistered = [o for o in outcomes if o.case.category == "unregistered"]
        registered = [o for o in outcomes if o.case.category == "registered"]
        unauthorized_invoked = sum(1 for o in unregistered if o.actual_outcome == "allow")
        admission_intercepted = len(unregistered) - unauthorized_invoked
        valid_executed = sum(1 for o in registered if o.actual_outcome == "allow")

        per_seed.append(
            {
                "seed": seed,
                "unauthorized_invocation_rate": rate(unauthorized_invoked, len(unregistered)),
                "admission_interception_rate": rate(admission_intercepted, len(unregistered)),
                "valid_capability_execution_rate": rate(valid_executed, len(registered)),
            }
        )

    return {
        "experiment": "12_capability_admission",
        "profile": profile.name,
        "n_seeds": len(SEEDS),
        "n_trials_per_seed": N_TRIALS_PER_SEED,
        "per_seed": per_seed,
        "all_seeds_100pct_interception": all(r["admission_interception_rate"] == 1.0 for r in per_seed),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp12_capability_admission.json"
    out_path.write_text(json.dumps(result, indent=2))
