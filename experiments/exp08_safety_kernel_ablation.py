"""Experiment 8: Safety Kernel Ablation (RQ8).

Which Safety Kernel components contribute most to safety and task
completion? Runs all 8 named ablations (plus full PAR as the baseline)
against (a) the Experiment 1 benchmark and (b) a fixed pick/place/move task.

Recovery rate and #replanning-steps per ablation are intentionally not
measured here - they need a denial-then-recovery scenario, which is what
Experiments 3/4 build; duplicating that machinery here would just be the
same infrastructure built twice. This experiment measures what's ablation-
specific: how much each mechanism changes unsafe-execution/interception and
whether the fixed task still completes and is still verifiably correct.
"""
from __future__ import annotations

import json
from pathlib import Path

from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.ablations import ABLATIONS
from par.evaluation.benchmarks import generate_safety_interception_benchmark
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import rate
from par.evaluation.runner import evaluate_cases
from par.evaluation.verifiers import verify_pick_and_place
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.skills import builtin_skills

N_SAFE = 25
N_UNSAFE = 25
BENCHMARK_SEED = 1


class _FixedTaskPlanner(Planner):
    def __init__(self) -> None:
        self._steps = [
            ("pick", {"object": "red_object"}),
            ("place", {"target": "blue_container"}),
            ("move", {"x": 0.3, "y": 0.3, "z": 0.0}),
            (TASK_COMPLETE, {"message": "done", "success": True}),
        ]

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


def _registry(profile_name: str) -> SkillRegistry:
    """Registers builtin skills for `profile_name` specifically. Builtin
    skills default to env_profiles=["simulation"] - registering them
    unmodified would make Admission fail for every ablation, including
    full_par (unablated), the moment profile_name isn't "simulation"."""
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def run() -> dict:
    profile = load_profile("strict_simulation")
    goal_verifier = verify_pick_and_place("red_object", "blue_container")
    results = []

    for name, kernel_cls in ABLATIONS.items():
        # (a) benchmark: unsafe/safe interception behavior (each BenchmarkCase
        # sets its own env_profiles override, so this part is unaffected)
        cases = generate_safety_interception_benchmark(profile, seed=BENCHMARK_SEED, n_safe=N_SAFE, n_unsafe=N_UNSAFE)
        kernel_for_benchmark = kernel_cls(profile)
        outcomes = evaluate_cases(cases, _registry(profile.name), kernel_for_benchmark)
        unsafe = [o for o in outcomes if o.case.is_unsafe]
        safe = [o for o in outcomes if not o.case.is_unsafe]
        unsafe_executed = sum(1 for o in unsafe if o.actual_outcome == "allow")
        safe_falsely_denied = sum(1 for o in safe if o.actual_outcome == "deny")

        # (b) fixed task: does it still complete, and is completion real?
        registry = _registry(profile.name)
        robot = MockRobot()
        agent = Agent(registry, planner=_FixedTaskPlanner())
        telemetry = CapturingTelemetryLogger()
        kernel_for_task = kernel_cls(profile)
        runtime = Runtime(agent, robot, safety_kernel=kernel_for_task, telemetry=telemetry)
        task_results = runtime.run_task("pick, place, move", max_steps=10)
        runtime.close()

        results.append(
            {
                "ablation": name,
                "benchmark": {
                    "unsafe_execution_rate": rate(unsafe_executed, len(unsafe)),
                    "interception_rate": rate(len(unsafe) - unsafe_executed, len(unsafe)),
                    "false_denial_rate": rate(safe_falsely_denied, len(safe)),
                },
                "fixed_task": {
                    "n_actions_attempted": len(task_results),
                    "n_succeeded": sum(1 for r in task_results if r.success),
                    "agent_status": agent.state.status.value,
                    "verified_success": goal_verifier(runtime.world_state.latest_observation),
                    "safety_outcome_counts": telemetry.outcome_counts(),
                },
            }
        )

    baseline = next(r for r in results if r["ablation"] == "full_par")
    return {
        "experiment": "8_safety_kernel_ablation",
        "profile": profile.name,
        "n_safe": N_SAFE,
        "n_unsafe": N_UNSAFE,
        "baseline": "full_par",
        "results_by_ablation": results,
        "unsafe_execution_rate_delta_vs_baseline": {
            r["ablation"]: r["benchmark"]["unsafe_execution_rate"] - baseline["benchmark"]["unsafe_execution_rate"]
            for r in results
        },
        "caveats": [
            "no_workspace_check shows 0.0 delta not because workspace-checking is "
            "unimportant, but because this benchmark's workspace_violation case "
            "(10x the workspace bound) also violates the velocity check so severely "
            "that velocity catches it anyway (as MODIFY, still counted as "
            "'intercepted' since only outcome=='allow' counts as unsafe execution). "
            "A workspace-only violation (far out of bounds but reachable within the "
            "velocity budget from a non-origin start) would isolate this properly.",
            "no_escalate shows 0.0 delta because this benchmark's 5 unsafe categories "
            "never include a high-risk/escalate-worthy case - see Experiment 2's "
            "four-outcome scenario for an escalate-sensitive benchmark instead.",
            "fixed_task's verified_success only checks pick+place (see "
            "verify_pick_and_place), not the final move - an ablation that only "
            "changes how the move step is handled (no_modify, no_velocity_check) "
            "can still show verified_success=True even though that step's outcome "
            "changed, because _FixedTaskPlanner is non-adaptive (scripted, not "
            "reasoning) and moves on regardless of the move step's result.",
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp08_safety_kernel_ablation.json"
    out_path.write_text(json.dumps(result, indent=2))
