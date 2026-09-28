"""Experiment 3: Safety Denial and Re-planning (RQ3).

Can an agent recover from a Safety Kernel denial by using the denial reason
as feedback and re-planning through the same reasoning loop?

Uses RecoveringMovePlanner (see module docstring there for why: statistical
volume without spending live-LLM API quota to reconfirm a mechanism already
validated qualitatively with a real model in the paper). Each trial places
an obstacle at a randomized distance from a fixed target; some trials never
trigger a denial at all (obstacle far away), isolating RecoveryRate to
exactly the trials where recovery was actually needed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from par.core.agent import Agent
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import rate, summarize
from par.evaluation.recovering_planner import RecoveringMovePlanner
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

N_TRIALS = 200
TARGET = (1.0, 1.0, 0.0)
MAX_ATTEMPTS = 6
SEED = 20


def _registry(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _run_trial(rng: random.Random, profile_name: str) -> dict:
    profile = load_profile(profile_name)
    robot = MockRobot()
    # Obstacle placed at a randomized offset from the target - sometimes far
    # enough away to never trigger a denial, sometimes squarely on it.
    offset = rng.uniform(-0.3, 0.3)
    robot.add_obstacle("obstacle", TARGET[0] + offset, TARGET[1] + offset, TARGET[2])

    planner = RecoveringMovePlanner(target=TARGET, max_attempts=MAX_ATTEMPTS)
    agent = Agent(_registry(profile_name), planner=planner)
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(profile), telemetry=telemetry)

    task_results = runtime.run_task("move to target", max_steps=MAX_ATTEMPTS + 2)
    runtime.close()

    experienced_denial = planner.denial_count > 0
    final_success = bool(agent.state.reported_success) and (task_results[-1].success if task_results else False)

    return {
        "experienced_denial": experienced_denial,
        "n_denials": planner.denial_count,
        "n_actions_attempted": len(task_results),
        "final_success": final_success,
        "repeated_unsafe_proposals": sum(1 for e in telemetry.events if e.safety_decision == "deny"),
    }


def run() -> dict:
    rng = random.Random(SEED)
    trials = [_run_trial(rng, "strict_simulation") for _ in range(N_TRIALS)]

    with_denial = [t for t in trials if t["experienced_denial"]]
    without_denial = [t for t in trials if not t["experienced_denial"]]
    recovered = [t for t in with_denial if t["final_success"]]

    return {
        "experiment": "3_safety_denial_and_replanning",
        "profile": "strict_simulation",
        "n_trials": N_TRIALS,
        "n_trials_with_denial": len(with_denial),
        "n_trials_without_denial": len(without_denial),
        "recovery_rate": rate(len(recovered), len(with_denial)),
        "final_task_success_rate_overall": rate(sum(1 for t in trials if t["final_success"]), N_TRIALS),
        "replanning_steps_when_denied": summarize([t["n_denials"] for t in with_denial]).__dict__ if with_denial else None,
        "denials_per_successful_task": summarize([t["n_denials"] for t in recovered]).__dict__ if recovered else None,
        "notes": [
            "Uses a deterministic rule-based re-planner for statistical volume, "
            "not a live LLM - see recovering_planner.py docstring. Live LLM "
            "re-planning (Gemini) is validated qualitatively with a real example "
            "in the paper, not re-run here at volume to avoid spending API quota "
            "reconfirming an already-established result.",
            "recovery_rate=1.0 here reflects this scenario's geometry, not a "
            "general guarantee: the planner's max cumulative nudge "
            f"({MAX_ATTEMPTS} x its step) comfortably exceeds the obstacle's "
            "randomized offset range, so eventual recovery is close to "
            "guaranteed by construction. Whether recovery holds under harder "
            "conditions (larger/multiple obstacles, tighter workspace) is what "
            "Experiments 7 and 20 stress-test, not a claim made here.",
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp03_safety_denial_and_replanning.json"
    out_path.write_text(json.dumps(result, indent=2))
