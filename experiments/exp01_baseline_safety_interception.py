"""Experiment 1: Baseline Safety Interception (RQ1).

Does PAR prevent unsafe actions proposed by an LLM-directed agent from
reaching the robot execution layer?

Compares two arms on the same benchmark of safe/unsafe action proposals,
across multiple seeded benchmark draws:
  direct:   Observation -> Planner -> Robot            (no governance)
  governed: Observation -> Planner -> SafetyKernel -> Robot

Uses strict_simulation: the only shipped profile where all five unsafe
categories (workspace violation, collision, excess velocity, unavailable
capability, invalid parameters) are simultaneously constructible - see
par.evaluation.benchmarks module docstring for why simulation/real_robot
can't produce a genuine excess-velocity case.
"""
from __future__ import annotations

import json
from pathlib import Path

from par.core.skill import SkillRegistry
from par.evaluation.benchmarks import generate_safety_interception_benchmark
from par.evaluation.metrics import rate, summarize
from par.evaluation.runner import evaluate_cases
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

SEEDS = list(range(10))
N_SAFE = 25
N_UNSAFE = 25
METRIC_KEYS = ("unsafe_execution_rate", "interception_rate", "safe_execution_rate", "false_denial_rate")


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def _run_arm(profile, governed: bool, seed: int) -> dict:
    cases = generate_safety_interception_benchmark(profile, seed=seed, n_safe=N_SAFE, n_unsafe=N_UNSAFE)
    kernel = SafetyKernel(profile) if governed else None
    outcomes = evaluate_cases(cases, _registry(), kernel)

    unsafe = [o for o in outcomes if o.case.is_unsafe]
    safe = [o for o in outcomes if not o.case.is_unsafe]

    unsafe_executed = sum(1 for o in unsafe if o.actual_outcome == "allow")
    unsafe_intercepted = len(unsafe) - unsafe_executed
    safe_executed = sum(1 for o in safe if o.actual_outcome == "allow")
    safe_falsely_denied = sum(1 for o in safe if o.actual_outcome == "deny")

    return {
        "seed": seed,
        "unsafe_execution_rate": rate(unsafe_executed, len(unsafe)),
        "interception_rate": rate(unsafe_intercepted, len(unsafe)),
        "safe_execution_rate": rate(safe_executed, len(safe)),
        "false_denial_rate": rate(safe_falsely_denied, len(safe)),
        "n_unsafe": len(unsafe),
        "n_safe": len(safe),
    }


def _summary_dict(runs: list[dict], key: str) -> dict:
    stats = summarize([r[key] for r in runs])
    return {"n": stats.n, "mean": stats.mean, "median": stats.median, "stdev": stats.stdev, "ci95_half_width": stats.ci95_half_width}


def run() -> dict:
    profile = load_profile("strict_simulation")
    direct_runs = [_run_arm(profile, governed=False, seed=s) for s in SEEDS]
    governed_runs = [_run_arm(profile, governed=True, seed=s) for s in SEEDS]

    return {
        "experiment": "1_baseline_safety_interception",
        "profile": profile.name,
        "n_seeds": len(SEEDS),
        "n_safe_per_seed": N_SAFE,
        "n_unsafe_per_seed": N_UNSAFE,
        "direct": {k: _summary_dict(direct_runs, k) for k in METRIC_KEYS},
        "governed": {k: _summary_dict(governed_runs, k) for k in METRIC_KEYS},
        "per_seed": {"direct": direct_runs, "governed": governed_runs},
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))
    out_path = Path(__file__).parent / "results" / "exp01_baseline_safety_interception.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull results (incl. per-seed data) written to {out_path}")
