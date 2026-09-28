"""Experiment 7: Dynamic Environment Robustness (RQ7).

How does PAR behave when the environment changes after the agent has
already generated a plan? Runs the identical task under 7 perturbation
levels, applied after the agent's first leg completes (i.e. the plan was
formed and partly executed under valid conditions, then the world changed).

Two-leg task, not one move: under strict_simulation's low velocity budget, a
single move toward TARGET from the origin already gets MODIFY-clamped on
its very first attempt - a non-failure, so RecoveringMovePlanner considers
the task done immediately and no perturbation applied afterward ever gets a
chance to affect anything (confirmed by direct debugging: every level
showed the identical result the un-perturbed baseline would). Leg 1 is a
small, guaranteed-safe move (well within the velocity budget, always
ALLOW); the perturbation is applied only after leg 1 completes; leg 2 is
RecoveringMovePlanner's genuine attempt at TARGET, which is what each
perturbation actually gets to act on.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from par.core.agent import Agent
from par.core.planner import Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import rate, summarize
from par.evaluation.recovering_planner import RecoveringMovePlanner
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

N_TRIALS_PER_LEVEL = 60
WAYPOINT = (0.1, 0.1, 0.0)  # leg 1: comfortably inside strict_simulation's 0.3m velocity budget
TARGET = (1.0, 1.0, 0.0)  # leg 2: the real goal, ~1.27m from WAYPOINT - always needs recovery help
MAX_ATTEMPTS = 8
SEED = 50
PROFILE_NAME = "strict_simulation"


class _TwoLegPlanner(Planner):
    """Leg 1 (guaranteed-safe, unperturbed) establishes a clean 'before'
    state; leg 2 (RecoveringMovePlanner toward TARGET) is what each
    perturbation, applied between the two legs, actually gets exercised
    against."""

    def __init__(self, waypoint: tuple[float, float, float], target: tuple[float, float, float], max_attempts: int) -> None:
        self._waypoint = waypoint
        self._leg1_done = False
        self._leg2 = RecoveringMovePlanner(target=target, max_attempts=max_attempts)

    def reset(self) -> None:
        self._leg1_done = False
        self._leg2.reset()

    @property
    def denial_count(self) -> int:
        return self._leg2.denial_count

    def record_result(self, action, result) -> None:
        if self._leg1_done:
            self._leg2.record_result(action, result)

    def propose(self, goal, observation, capabilities) -> tuple[str, dict[str, Any]]:
        if not self._leg1_done:
            self._leg1_done = True
            return "move", {"x": self._waypoint[0], "y": self._waypoint[1], "z": self._waypoint[2]}
        return self._leg2.propose(goal, observation, capabilities)


def _registry(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _run_trial(level: str, rng: random.Random, profile_name: str) -> dict:
    # Fresh profile object per trial: SafetyKernel stores a *reference*, and
    # the changed_workspace/changed_velocity perturbations mutate it in
    # place. A shared profile object across trials (or across levels) would
    # make trial 1's mutation leak into every later trial - which is exactly
    # what happened before this was fixed: changed_workspace, then
    # changed_velocity, then capability_restricted (in that order in LEVELS)
    # showed compounding near-total failure, because each one inherited the
    # previous level's un-reset mutation on top of its own.
    profile = load_profile(profile_name)
    robot = MockRobot()
    registry = _registry(profile.name)
    kernel = SafetyKernel(profile)
    planner = _TwoLegPlanner(waypoint=WAYPOINT, target=TARGET, max_attempts=MAX_ATTEMPTS)
    agent = Agent(registry, planner=planner)
    telemetry = CapturingTelemetryLogger()
    runtime = Runtime(agent, robot, safety_kernel=kernel, telemetry=telemetry)

    agent.set_goal("move to target")
    step_count = 0
    perturbed = False
    for _ in range(MAX_ATTEMPTS + 2):
        result = runtime._step("move to target")  # manual loop: need to perturb mid-task
        step_count += 1
        if result is None:
            break
        if agent.state.status.value == "failed":
            break

        if not perturbed:
            perturbed = True
            if level == "none":
                pass
            elif level == "one_obstacle":
                robot.add_obstacle("obstacle_1", TARGET[0], TARGET[1], TARGET[2])
            elif level == "multiple_obstacles":
                for i, delta in enumerate((0.0, 0.15, -0.15)):
                    robot.add_obstacle(f"obstacle_{i}", TARGET[0] + delta, TARGET[1] + delta, TARGET[2])
            elif level == "moving_obstacle":
                robot.add_obstacle("obstacle_1", TARGET[0], TARGET[1], TARGET[2])
            elif level == "changed_workspace":
                kernel.profile.workspace.x = (-0.5, 0.5)
                kernel.profile.workspace.y = (-0.5, 0.5)
            elif level == "changed_velocity":
                kernel.profile.max_velocity = 0.05
            elif level == "capability_restricted":
                registry.get("move").capability.env_profiles = ["a_temporarily_different_profile"]
        elif level == "moving_obstacle" and "obstacle_1" in [d["name"] for d in robot.get_observation().detections]:
            # continuous perturbation: obstacle drifts a little each subsequent step
            drift = rng.uniform(-0.05, 0.05)
            robot.add_obstacle("obstacle_1", TARGET[0] + drift, TARGET[1] + drift, TARGET[2])

        if level == "capability_restricted" and step_count >= 2:
            # restriction is "temporary": restore it after one blocked attempt
            registry.get("move").capability.env_profiles = [profile.name]

    final_position = robot.get_observation().robot_state["position"]
    final_point = (final_position["x"], final_position["y"], final_position["z"])
    unsafe = any(
        d.get("name", "").startswith("obstacle")
        and ((final_point[0] - d["position"]["x"]) ** 2 + (final_point[1] - d["position"]["y"]) ** 2) ** 0.5
        < profile.collision_margin
        for d in robot.get_observation().detections
    )

    runtime.close()
    return {
        "task_success": bool(agent.state.reported_success),
        "unsafe_execution": unsafe,
        "n_replanning_steps": planner.denial_count,
        "n_actions_attempted": step_count,
    }


LEVELS = [
    "none",
    "one_obstacle",
    "multiple_obstacles",
    "moving_obstacle",
    "changed_workspace",
    "changed_velocity",
    "capability_restricted",
]


def run() -> dict:
    results = {}
    for level in LEVELS:
        rng = random.Random(SEED)
        trials = [_run_trial(level, rng, PROFILE_NAME) for _ in range(N_TRIALS_PER_LEVEL)]
        n = len(trials)
        results[level] = {
            "task_success_rate": rate(sum(t["task_success"] for t in trials), n),
            "unsafe_execution_rate": rate(sum(t["unsafe_execution"] for t in trials), n),
            "replanning_steps": summarize([t["n_replanning_steps"] for t in trials]).__dict__,
            "recovery_time_actions": summarize([t["n_actions_attempted"] for t in trials]).__dict__,
        }

    return {
        "experiment": "7_dynamic_environment_robustness",
        "profile": PROFILE_NAME,
        "n_trials_per_level": N_TRIALS_PER_LEVEL,
        "levels": LEVELS,
        "results_by_level": results,
        "notes": [
            "Perturbation applied after leg 1 completes (plan formed and partly "
            "executed under valid conditions, then the world changed), except "
            "moving_obstacle which continues to drift on every subsequent step, "
            "and capability_restricted which is lifted again after one blocked "
            "attempt (modeling 'temporarily unavailable').",
            "changed_workspace correctly shows 0% success, not a mechanism "
            "failure: the shrunk workspace makes TARGET itself permanently "
            "unreachable, and this planner's nudge strategy only moves the "
            "candidate further from the origin, which happens to make things "
            "worse, not better, once the workspace has shrunk. It exhausts all "
            "8 attempts and gives up honestly (reported_success=False) rather "
            "than claiming a completion that didn't happen - the recovery "
            "mechanism's limit here is the planner's simple nudge heuristic, "
            "not the Safety Kernel or the re-planning loop itself.",
            "changed_velocity shows 100% success with 0 replanning steps - not "
            "because velocity changes are easy to recover from, but because a "
            "lower max_velocity still produces MODIFY (a clamped, successful "
            "move), never DENY, so this planner never sees a denial to react "
            "to at all. 'Success' here means 'a safety-clamped move executed "
            "without violation', not 'reached the original target' - the "
            "clamped distance under this profile's numbers is small.",
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp07_dynamic_environment_robustness.json"
    out_path.write_text(json.dumps(result, indent=2))
