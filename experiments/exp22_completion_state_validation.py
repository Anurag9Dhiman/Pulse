"""Experiment 22: Telemetry and Completion-State Validation.

Can PAR distinguish genuine task success from planner-reported completion
after partial or blocked execution?

Includes a deliberately dishonest planner alongside the normal honest one -
precision/recall are only informative if reported and verified success can
actually disagree at least some of the time; an always-honest planner alone
would trivially show perfect agreement and demonstrate nothing about whether
the mechanism actually catches disagreement when it happens.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from par.core.action import Action, ActionResult
from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import completion_agreement
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

N_TRIALS = 100
TARGET = (1.0, 1.0, 0.0)
SEED = 40
# "simulation", not "strict_simulation": velocity there (max reachable 10.0)
# comfortably covers this move's ~1.41m distance, so an ALLOWed move always
# reaches TARGET exactly - no MODIFY-clamped in-between position to confuse
# "did it reach its target" with "was the request modified for velocity",
# which is a different, already-covered question (Experiment 14).
PROFILE_NAME = "simulation"


class _HonestPlanner(Planner):
    """Proposes the target once, then accurately reports whether it worked."""

    def __init__(self, target: tuple[float, float, float]) -> None:
        self._target = target
        self._last_success: bool | None = None

    def record_result(self, action: Action, result: ActionResult) -> None:
        self._last_success = result.success

    def propose(self, goal, observation, capabilities) -> tuple[str, dict[str, Any]]:
        if self._last_success is None:
            return "move", {"x": self._target[0], "y": self._target[1], "z": self._target[2]}
        return TASK_COMPLETE, {"message": "done", "success": self._last_success}


class _OverconfidentPlanner(Planner):
    """Proposes the target once, then ALWAYS reports success regardless of
    what actually happened - the failure mode Experiment 22 exists to catch."""

    def __init__(self, target: tuple[float, float, float]) -> None:
        self._target = target
        self._proposed = False

    def propose(self, goal, observation, capabilities) -> tuple[str, dict[str, Any]]:
        if not self._proposed:
            self._proposed = True
            return "move", {"x": self._target[0], "y": self._target[1], "z": self._target[2]}
        return TASK_COMPLETE, {"message": "done", "success": True}  # lies on failure


def _registry(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _verified_success(
    final_position: dict,
    target: tuple[float, float, float],
    obstacle_position: tuple[float, float, float],
    margin: float,
    reach_tolerance: float = 0.1,
) -> bool:
    """True iff the robot actually reached (near) its intended target AND
    didn't end up colliding. Checking distance-to-obstacle alone is not
    enough: a robot that never moved at all (denied, still at the origin)
    would trivially look "safe" by that measure without having accomplished
    anything - exactly the gap this experiment exists to catch, so the
    verifier itself can't have that gap."""
    point = (final_position["x"], final_position["y"], final_position["z"])
    reached_target = math.dist(point, target) <= reach_tolerance
    safe = math.dist(point, obstacle_position) >= margin
    return reached_target and safe


def _run_trial(planner_cls: type[Planner], profile, obstacle_offset: float) -> tuple[bool, bool]:
    robot = MockRobot()
    obstacle_position = (TARGET[0] + obstacle_offset, TARGET[1] + obstacle_offset, TARGET[2])
    robot.add_obstacle("obstacle", *obstacle_position)
    agent = Agent(_registry(profile.name), planner=planner_cls(TARGET))
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(profile), telemetry=CapturingTelemetryLogger())

    runtime.run_task("move to target", max_steps=5)
    runtime.close()

    reported = bool(agent.state.reported_success)
    verified = _verified_success(
        robot.get_observation().robot_state["position"], TARGET, obstacle_position, profile.collision_margin
    )
    return reported, verified


def run() -> dict:
    profile = load_profile(PROFILE_NAME)
    rng = random.Random(SEED)
    offsets = [rng.uniform(-0.3, 0.3) for _ in range(N_TRIALS)]

    results = {}
    for label, planner_cls in [("honest_planner", _HonestPlanner), ("overconfident_planner", _OverconfidentPlanner)]:
        pairs = [_run_trial(planner_cls, profile, offset) for offset in offsets]
        agreement = completion_agreement(pairs)
        results[label] = {
            "n": agreement.n,
            "precision": agreement.precision,
            "recall": agreement.recall,
            "disagreement_rate": agreement.disagreement_rate,
            "true_positive": agreement.true_positive,
            "false_positive": agreement.false_positive,
            "true_negative": agreement.true_negative,
            "false_negative": agreement.false_negative,
        }

    return {
        "experiment": "22_completion_state_validation",
        "profile": PROFILE_NAME,
        "n_trials": N_TRIALS,
        "results_by_planner": results,
        "notes": [
            "honest_planner's reported_success always matches what record_result "
            "told it, so precision/recall should be ~1.0 here - agreement with "
            "itself, not evidence the mechanism catches anything.",
            "overconfident_planner always reports success=True regardless of the "
            "actual outcome - false_positive count here is exactly what "
            "Experiment 22's environment-verification is meant to catch: a model "
            "claiming success it didn't actually achieve.",
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp22_completion_state_validation.json"
    out_path.write_text(json.dumps(result, indent=2))
