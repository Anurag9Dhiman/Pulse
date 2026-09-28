"""Experiment 9: Runtime Latency and Overhead (RQ9).

Decomposes T_total = T_LLM + T_Admission + T_Policy + T_Execution + T_Logging
and compares direct execution vs. PAR-Admission-only vs. full PAR.

T_LLM is NOT re-measured here: it would mean burning live API quota purely
to reconfirm a number already measured for real. Uses the paper's reported
live Gemini figures (gemini-3.1-flash-lite, ~0.8-1.4s/call) instead, clearly
labeled as sourced from that measurement, not this run. Everything PAR
actually controls (Admission, Policy Guard, execution dispatch, logging) is
measured directly here, many times over, since these are cheap and local.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from par.core.skill import SkillRegistry
from par.evaluation.metrics import latency_stats
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

N_TRIALS = 500
T_LLM_SOURCE = "paper/par_paper.tex Table I - live gemini-3.1-flash-lite, not re-measured here"
T_LLM_RANGE_SECONDS = (0.8, 1.4)


def run() -> dict:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    skill = registry.get("move")
    capability = skill.capability
    profile = load_profile("simulation")
    kernel = SafetyKernel(profile)
    robot = MockRobot()
    action = skill.build_action({"x": 0.1, "y": 0.0, "z": 0.0})
    observation = robot.get_observation()

    t_admission = []
    t_policy = []
    t_execution = []
    t_logging = []

    for _ in range(N_TRIALS):
        start = time.perf_counter()
        kernel.admit(capability)
        t_admission.append(time.perf_counter() - start)

        start = time.perf_counter()
        kernel.check(action, observation, capability)
        t_policy.append(time.perf_counter() - start)

        start = time.perf_counter()
        robot.execute(action)
        t_execution.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = json.dumps({"skill": action.skill_name, "parameters": action.parameters})  # logging serialization cost
        t_logging.append(time.perf_counter() - start)

    par_overhead_mean = (
        latency_stats(t_admission)["mean"] + latency_stats(t_policy)["mean"] + latency_stats(t_logging)["mean"]
    )

    return {
        "experiment": "9_runtime_latency_overhead",
        "n_trials": N_TRIALS,
        "t_admission": latency_stats(t_admission),
        "t_policy": latency_stats(t_policy),
        "t_execution_mock_robot": latency_stats(t_execution),
        "t_logging_serialization": latency_stats(t_logging),
        "par_overhead_seconds_mean": par_overhead_mean,
        "t_llm_reference": {
            "source": T_LLM_SOURCE,
            "range_seconds": T_LLM_RANGE_SECONDS,
            "note": "not measured in this run - see source",
        },
        "par_overhead_as_fraction_of_t_llm": {
            "vs_fastest_llm_call": par_overhead_mean / T_LLM_RANGE_SECONDS[0],
            "vs_slowest_llm_call": par_overhead_mean / T_LLM_RANGE_SECONDS[1],
        },
        "configurations": {
            "direct_execution": "T_execution only (no Admission, no Policy Guard, no PAR logging)",
            "par_admission_only": "T_Admission + T_execution + T_logging",
            "full_par": "T_Admission + T_Policy + T_execution + T_logging",
        },
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp09_runtime_latency_overhead.json"
    out_path.write_text(json.dumps(result, indent=2))
