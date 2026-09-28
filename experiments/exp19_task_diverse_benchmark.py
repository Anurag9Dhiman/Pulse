"""Experiment 19: Task-Diverse Benchmark.

Does PAR's governance and recovery behavior generalize across diverse
embodied tasks, not just the pick/place/move-near-an-obstacle scenario
every other experiment in this suite uses? 8 task families, each mapped to
a concrete scenario using PAR's actual 6 builtin skills - there is no 7th
skill to invent tasks from, so diversity here means diversity of *how*
those 6 skills get composed and constrained, not a larger skill set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import rate
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

N_TRIALS_PER_FAMILY = 20
SEED = 70


class _ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        if not self._steps:
            return TASK_COMPLETE, {"message": "exhausted script", "success": False}
        return self._steps.pop(0)


def _registry(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _build_trial(family: str, rng: random.Random) -> tuple[Planner, str]:
    """Returns (planner, profile_name) for one trial of the given family."""
    # MockRobot always has red_object at (0.5, 0.2, 0) and blue_container at
    # (-0.3, 0.4, 0) - every "move" target/waypoint below is chosen to stay
    # outside collision_margin of both, confirmed by direct calculation, not
    # just "looks far enough". A target that coincidentally lands near a
    # fixture produces a real collision denial for a reason unrelated to
    # whatever the family is meant to test - this bit 3 experiments already
    # (10, 11, and an earlier draft of this one) before being fixed here.
    if family == "navigation":
        waypoints = [(-0.1 * i, -0.1 * i, 0.0) for i in range(1, 4)]  # away from both fixtures
        steps = [("move", {"x": x, "y": y, "z": z}) for x, y, z in waypoints]
        steps.append((TASK_COMPLETE, {"message": "navigated", "success": True}))
        return _ScriptedPlanner(steps), "simulation"

    if family == "reaching":
        x = rng.uniform(-0.3, -0.05)  # negative range: measured safe distance from both fixtures
        return _ScriptedPlanner([("move", {"x": x, "y": 0.0, "z": 0.0}), (TASK_COMPLETE, {"message": "reached", "success": True})]), "simulation"

    if family == "object_detection":
        return _ScriptedPlanner([("detect", {}), (TASK_COMPLETE, {"message": "detected", "success": True})]), "simulation"

    if family == "pick_and_place":
        return (
            _ScriptedPlanner(
                [
                    ("pick", {"object": "red_object"}),
                    ("place", {"target": "blue_container"}),
                    (TASK_COMPLETE, {"message": "placed", "success": True}),
                ]
            ),
            "simulation",
        )

    if family == "inspection":
        return _ScriptedPlanner([("inspect", {"target": "blue_container"}), (TASK_COMPLETE, {"message": "inspected", "success": True})]), "simulation"

    if family == "object_transport":
        return (
            _ScriptedPlanner(
                [
                    ("pick", {"object": "red_object"}),
                    ("move", {"x": -0.2, "y": -0.2, "z": 0.0}),  # measured safe distance from both fixtures
                    ("place", {"target": "blue_container"}),
                    (TASK_COMPLETE, {"message": "transported", "success": True}),
                ]
            ),
            "simulation",
        )

    if family == "workspace_constrained_manipulation":
        # SafetyKernel only position-checks the "move" skill (pick/place act
        # on named objects, not raw coordinates, so they never reach
        # _check_move) - a pick+place-only task under a tight profile
        # wouldn't actually exercise any workspace constraint at all. This
        # needs a genuine move step to test what its name claims.
        return (
            _ScriptedPlanner(
                [
                    ("pick", {"object": "red_object"}),
                    # (-0.4, -0.4, 0): within high_safety's +/-0.5 workspace,
                    # >0.5m from both red_object and blue_container (measured:
                    # 1.08m and 0.81m respectively), and its ~0.57m distance
                    # from the origin exceeds high_safety's 0.1m velocity
                    # budget - triggers MODIFY on velocity grounds specifically,
                    # not an accidental DENY on collision grounds.
                    ("move", {"x": -0.4, "y": -0.4, "z": 0.0}),
                    ("place", {"target": "blue_container"}),
                    (TASK_COMPLETE, {"message": "placed under tight constraints", "success": True}),
                ]
            ),
            "high_safety",
        )

    if family == "capability_restricted":
        # "stop" is deliberately left unregistered for this profile via a
        # separate registry built below - this planner just proposes it.
        return _ScriptedPlanner([("stop", {}), (TASK_COMPLETE, {"message": "n/a", "success": False})]), "simulation"

    raise ValueError(family)


def _run_trial(family: str, rng: random.Random) -> dict:
    planner, profile_name = _build_trial(family, rng)
    profile = load_profile(profile_name)
    robot = MockRobot()
    registry = _registry(profile.name)
    if family == "capability_restricted":
        registry.get("stop").capability.env_profiles = ["a_profile_not_in_use"]

    agent = Agent(registry, planner=planner)
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(profile), telemetry=telemetry)

    task_results = runtime.run_task(f"{family} task", max_steps=8)
    runtime.close()

    return {
        "goal": f"{family} task",
        "n_actions_proposed": len(telemetry.events),
        "n_executions": sum(1 for e in telemetry.events if e.safety_decision in ("allow", "modify")),
        "n_denials": sum(1 for e in telemetry.events if e.safety_decision == "deny"),
        "safety_outcomes": telemetry.outcome_counts(),
        "final_outcome": {
            "agent_status": agent.state.status.value,
            "reported_success": agent.state.reported_success,
            "last_action_success": task_results[-1].success if task_results else None,
        },
    }


FAMILIES = [
    "navigation",
    "reaching",
    "object_detection",
    "pick_and_place",
    "inspection",
    "object_transport",
    "workspace_constrained_manipulation",
    "capability_restricted",
]


def run() -> dict:
    rng = random.Random(SEED)
    results = {}
    for family in FAMILIES:
        trials = [_run_trial(family, rng) for _ in range(N_TRIALS_PER_FAMILY)]
        n = len(trials)
        results[family] = {
            "n_trials": n,
            "task_success_rate": rate(
                sum(1 for t in trials if t["final_outcome"]["reported_success"]), n
            ),
            "mean_actions_proposed": sum(t["n_actions_proposed"] for t in trials) / n,
            "mean_denials": sum(t["n_denials"] for t in trials) / n,
            "sample_trial": trials[0],
        }

    return {
        "experiment": "19_task_diverse_benchmark",
        "n_trials_per_family": N_TRIALS_PER_FAMILY,
        "families": FAMILIES,
        "results_by_family": results,
        "note": (
            "Tasks are composed from PAR's actual 6 builtin skills (detect, "
            "move, pick, place, stop, inspect) - there is no 7th skill to draw "
            "genuinely different task types from, so 'diverse' here means "
            "diverse composition and constraint (single-step vs multi-step, "
            "generous vs tight workspace, registered vs restricted capability), "
            "not a larger skill vocabulary. capability_restricted is expected "
            "to show 0% task_success_rate by design (the one capability this "
            "planner is scripted to use is deliberately unregistered) - that's "
            "the correct outcome being verified, not a failure of the suite."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp19_task_diverse_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
