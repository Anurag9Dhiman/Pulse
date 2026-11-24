"""Experiment 4: where should recovery from a Safety Kernel denial live?

Four configurations over the same seeded obstacle placements (paired):

  1. direct                  - no Safety Kernel at all
  2. safety_no_replan        - kernel denies; nothing recovers
  3. safety_recovery_module  - kernel denies; a hand-coded module recovers
  4. par_feedback            - kernel denies; the denial is fed to the planner,
                               which re-plans in its own loop

Configurations 3 and 4 run the *identical* recovery strategy object
(DetourStrategy), so the only thing that differs between them is where the
recovery lives. An earlier version of this experiment let the planner extend
its target while the module shortened it; with an endpoint-only collision
test both appeared to reach 100%, which hid the fact that the two arms were
running different algorithms. Configuration 5 keeps the old planner strategy
as an explicit negative control.

Scenario. The obstacle is placed *on the path* between the origin and the
goal (at a random fraction along it, with a small lateral jitter), never on
the goal itself. That makes the direct move always unsafe and the goal always
reachable - so "did the robot get there" is a meaningful question. The
previous scenario put the obstacle next to the goal, which made the goal
unreachable in most trials and forced a success metric ("moved at all without
ending in contact") that a strategy could satisfy by retreating and stopping.

Ground truth is computed here, independently of the Safety Kernel, by
sweeping every executed motion segment against the obstacle - not by checking
the final position, which would score a robot that drove straight through an
obstacle on its way to the goal as safe.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from par.core.agent import Agent
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.metrics import summarize
from par.evaluation.recovering_planner import FixedGoalPlanner, GoalSeekingPlanner
from par.evaluation.recovery_architectures import FixedRecoveryRuntime, NoReplanRuntime
from par.evaluation.recovery_strategies import DetourStrategy, ExtendStrategy
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel, segment_point_distance
from par.skills import builtin_skills
from par.telemetry.logger import TelemetryLogger

N_TRIALS = 150
ORIGIN = (0.0, 0.0, 0.0)
GOAL = (1.0, 1.0, 0.0)
MAX_STEPS = 10
SEED = 30
PROFILE_NAME = "simulation"
TOLERANCE = 0.05

CONFIGS = [
    "direct",
    "safety_no_replan",
    "safety_recovery_module",
    "par_feedback",
    "par_feedback_extend_control",
]


class _TrackingRobot(MockRobot):
    """Records every motion segment actually executed, so safety can be judged
    from the path travelled rather than from where the robot stopped."""

    def __init__(self) -> None:
        super().__init__()
        # MockRobot ships a demo scene (red_object, blue_container) that also
        # sits near this goal line. Left in place it would deny every trial
        # for reasons unrelated to the obstacle under study.
        self._objects.clear()
        self.segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    def execute(self, action):
        before = tuple(self._position[a] for a in ("x", "y", "z"))
        result = super().execute(action)
        after = tuple(self._position[a] for a in ("x", "y", "z"))
        if action.skill_name == "move" and before != after:
            self.segments.append((before, after))
        return result


def _registry(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _obstacle_for(rng: random.Random) -> tuple[float, float, float]:
    """A point on the origin->goal segment, nudged sideways. Blocks the direct
    path; leaves the goal itself clear."""
    fraction = rng.uniform(0.35, 0.65)
    jitter = rng.uniform(-0.15, 0.15)
    direction = tuple(GOAL[i] - ORIGIN[i] for i in range(3))
    length = math.dist(ORIGIN, GOAL)
    perpendicular = (-direction[1] / length, direction[0] / length, 0.0)
    return tuple(
        ORIGIN[i] + direction[i] * fraction + perpendicular[i] * jitter for i in range(3)
    )


def _build(config: str, agent, robot, profile, telemetry):
    kernel = SafetyKernel(profile)
    if config == "direct":
        return Runtime(agent, robot, safety_kernel=None, telemetry=telemetry)
    if config == "safety_no_replan":
        return NoReplanRuntime(agent, robot, safety_kernel=kernel, telemetry=telemetry)
    if config == "safety_recovery_module":
        return FixedRecoveryRuntime(
            agent,
            robot,
            safety_kernel=kernel,
            telemetry=telemetry,
            strategy=DetourStrategy(),
            max_recovery_attempts=6,
        )
    return Runtime(agent, robot, safety_kernel=kernel, telemetry=telemetry)


def _planner_for(config: str):
    # Only the feedback configurations re-plan; the recovery-module arm must
    # not, or both arms would be recovering and the comparison would be void.
    if config == "par_feedback":
        return GoalSeekingPlanner(goal=GOAL, strategy=DetourStrategy())
    if config == "par_feedback_extend_control":
        return GoalSeekingPlanner(goal=GOAL, strategy=ExtendStrategy())
    return FixedGoalPlanner(goal=GOAL)


def _run_trial(config: str, profile, obstacle: tuple[float, float, float]) -> dict:
    robot = _TrackingRobot()
    robot.add_obstacle("obstacle", *obstacle)
    planner = _planner_for(config)
    agent = Agent(_registry(profile.name), planner=planner)
    telemetry = TelemetryLogger()
    runtime = _build(config, agent, robot, profile, telemetry)

    runtime.run_task("move to the goal", max_steps=MAX_STEPS)
    runtime.close()

    final = robot.get_observation().robot_state["position"]
    final_point = (final["x"], final["y"], final["z"])

    # Ground truth: did any executed segment pass within the margin of any
    # object actually in the scene? Checking only the placed obstacle would
    # miss anything else the robot could hit.
    scene = [
        (d["position"]["x"], d["position"]["y"], d["position"]["z"])
        for d in robot.get_observation().detections
        if d.get("position")
    ]
    unsafe = any(
        segment_point_distance(start, end, point) < profile.collision_margin
        for start, end in robot.segments
        for point in scene
    )
    reached = math.dist(final_point, GOAL) <= TOLERANCE

    return {
        "n_moves_executed": len(robot.segments),
        "n_denials": getattr(planner, "denial_count", 0),
        "unsafe_path": unsafe,
        "reached_goal": reached,
        "task_success": reached and not unsafe,
    }


def run() -> dict:
    profile = load_profile(PROFILE_NAME)
    rng = random.Random(SEED)
    obstacles = [_obstacle_for(rng) for _ in range(N_TRIALS)]

    results = {}
    for config in CONFIGS:
        trials = [_run_trial(config, profile, o) for o in obstacles]
        success = [1.0 if t["task_success"] else 0.0 for t in trials]
        unsafe = [1.0 if t["unsafe_path"] else 0.0 for t in trials]
        reached = [1.0 if t["reached_goal"] else 0.0 for t in trials]
        results[config] = {
            "task_success_rate": sum(success) / len(success),
            "goal_reached_rate": sum(reached) / len(reached),
            "unsafe_path_rate": sum(unsafe) / len(unsafe),
            "mean_moves_executed": sum(t["n_moves_executed"] for t in trials) / len(trials),
            "mean_denials": sum(t["n_denials"] for t in trials) / len(trials),
            "task_success_stats": summarize(success),
            "unsafe_path_stats": summarize(unsafe),
        }

    return {
        "experiment": "4_recovery_architecture_comparison",
        "profile": PROFILE_NAME,
        "n_trials": N_TRIALS,
        "origin": ORIGIN,
        "goal": GOAL,
        "seed": SEED,
        "results_by_config": results,
        "notes": (
            "Obstacle lies on the origin->goal path (fraction 0.35-0.65, lateral jitter "
            "+/-0.15), so the direct move is always unsafe and the goal is always reachable. "
            "safety_recovery_module and par_feedback run the identical DetourStrategy, "
            "differing only in where recovery lives. par_feedback_extend_control keeps the "
            "superseded extend strategy as a negative control. 'unsafe_path' is ground truth "
            "computed by sweeping each executed segment against the obstacle, independently "
            "of the Safety Kernel."
        ),
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))
    path = Path(__file__).parent / "results" / "exp04_recovery_architecture_comparison.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWritten to {path}")
