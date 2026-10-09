"""Experiment 18, run against Reach's real Webots e-puck simulation rather
than left as the undoable "needs physical hardware" placeholder.

This is NOT physical hardware, and this script does not claim otherwise:
there is still no real robot, no real sensor noise beyond what Webots'
physics models, and no real-world unpredictability. What it IS: a real,
externally-bridged, physically-simulated robot (differential-drive dynamics,
real collision geometry, a real separate OS process reached over a real
WebSocket, not an in-process MockRobot call) - a genuinely stronger tier
than every other experiment in this suite, even if short of RQ12's full
claim. See `honest_scope` in the written result for exactly what each of
the 6 skills does and doesn't physically mean on this platform (the e-puck
has no gripper - par_bridge.py's own pick/place handlers say so).

Requires experiments/webots_live/launch.sh already running (Webots +
par_bridge.py + computer_arm_bridge.py all up) - see that script.

Design:
  - Profile: real_robot (workspace [-1,1]x[-1,1]x[0,1.2], max_velocity 0.5,
    collision_margin 0.5, action_timeout 3.0s - same as Experiment 17).
  - All 6 builtin skills registered for "real_robot" at the start of every
    task's trial block (capability_restriction trials remove+restore just
    that one task's registration around the trial, not globally).
  - 3 conditions per task x roughly 5 trials each: baseline, obstacle_insertion
    (physically relocates blue_container, which Experiment 7's MockRobot
    version only ever simulated as an observation fixture - here it's an
    actual Supervisor-driven relocation of a real Solid), capability_restriction.
    Only `move` has a geometric Safety Kernel check at all (see
    safety/kernel.py's check()) - obstacle_insertion is structurally a null
    condition for the other 5 skills, reported as such rather than hidden.
  - Every DENY'd trial is followed by one untimed recovery-check: retry the
    same action once the perturbation is lifted, confirming PAR's
    feed-back-to-planner recoverability (Experiment 3's claim) holds here too.
  - One dedicated emergency-stop sequence (RQ11-adjacent, named in this
    experiment's own metric list) since nothing else in this run exercises it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from par.core.action import Action, ActionResult
from par.core.agent import Agent
from par.core.observation import Observation
from par.core.planner import Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.integrations.webots import WebotsBridge
from par.robots.webots_bridge import WebotsRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

PROFILE_NAME = "real_robot"
TASKS = ["move", "detect", "pick", "place", "inspect", "stop"]
N_BASELINE = 5
N_OBSTACLE = 5
N_CAPABILITY = 3

_RED_OBJECT = (0.5, 0.2, 0.0)
_BLUE_CONTAINER_HOME = (-0.3, 0.4, 0.0)
_MOVE_WAYPOINTS = [(0.0, -0.3, 0.0), (0.0, 0.0, 0.0)]  # safe hop, clear of both objects


class _OneShotPlanner(Planner):
    """Driven imperatively by this script, not a fixed step list - each
    trial sets .next_step right before the one run_once() call that uses it."""

    def __init__(self) -> None:
        self.next_step: tuple[str, dict[str, Any]] | None = None

    def propose(self, goal: str, observation: Observation, capabilities) -> tuple[str, dict[str, Any]]:
        assert self.next_step is not None, "driver bug: run_once() called without setting next_step"
        return self.next_step

    def record_result(self, action: Action, result: ActionResult) -> None:
        pass


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [PROFILE_NAME]
        registry.register(skill)
    return registry


def _params_for(task: str) -> dict[str, Any]:
    return {
        "move": {"x": _MOVE_WAYPOINTS[0][0], "y": _MOVE_WAYPOINTS[0][1], "z": _MOVE_WAYPOINTS[0][2]},
        "detect": {},
        "pick": {"object": "red_object"},
        "place": {"target": "blue_container"},
        "inspect": {"target": "red_object"},
        "stop": {},
    }[task]


def _ensure_gripper_state(robot: WebotsRobot, want_holding: bool) -> None:
    holding = robot.get_observation().robot_state.get("holding")
    if want_holding and holding is None:
        robot.execute(Action(action_id="setup-pick", skill_name="pick", parameters={"object": "red_object"}, created_at=_now()))
    elif not want_holding and holding is not None:
        robot.execute(Action(action_id="setup-place", skill_name="place", parameters={"target": "blue_container"}, created_at=_now()))


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _relocate(bridge: WebotsBridge, name: str, xyz: tuple[float, float, float]) -> None:
    bridge.send_action({"action_id": f"relocate-{name}", "skill_name": "relocate_object", "parameters": {"name": name, "x": xyz[0], "y": xyz[1], "z": xyz[2]}})


def _run_trial(agent: Agent, runtime: Runtime, planner: _OneShotPlanner, task: str, params: dict[str, Any]) -> dict[str, Any]:
    if task == "pick":
        _ensure_gripper_state(runtime.robot, want_holding=False)
    elif task == "place":
        _ensure_gripper_state(runtime.robot, want_holding=True)

    planner.next_step = (task, params)
    started = time.monotonic()
    result = runtime.run_once(f"experiment 18 trial: {task}")
    wall_seconds = time.monotonic() - started

    event = runtime.telemetry.events[-1]
    return {
        "task": task,
        "parameters": params,
        "outcome": event.safety_decision.upper(),
        "reason": event.rejection_reason,
        "success": result.success if result else None,
        "message": result.message if result else None,
        "latency_seconds": event.latency_seconds,
        "wall_seconds": round(wall_seconds, 3),
    }


def _recovery_check(agent: Agent, runtime: Runtime, planner: _OneShotPlanner, task: str, params: dict[str, Any]) -> bool:
    """Retried once the perturbation is lifted - the actual claim this
    experiment (and Experiment 3) is making: a denial doesn't end the task,
    the same skill tried again under normal conditions succeeds."""
    trial = _run_trial(agent, runtime, planner, task, params)
    return trial["outcome"] in ("ALLOW", "MODIFY")


def run_task_trials(agent: Agent, runtime: Runtime, planner: _OneShotPlanner, registry: SkillRegistry, bridge: WebotsBridge, task: str) -> dict[str, Any]:
    params = _params_for(task)
    capability = registry.get(task).capability
    waypoint_i = 0

    baseline, obstacle, capability_restriction = [], [], []

    for _ in range(N_BASELINE):
        if task == "move":
            params = {"x": _MOVE_WAYPOINTS[waypoint_i][0], "y": _MOVE_WAYPOINTS[waypoint_i][1], "z": _MOVE_WAYPOINTS[waypoint_i][2]}
            waypoint_i = 1 - waypoint_i
        baseline.append(_run_trial(agent, runtime, planner, task, params))

    for _ in range(N_OBSTACLE):
        if task == "move":
            target = (_MOVE_WAYPOINTS[waypoint_i][0], _MOVE_WAYPOINTS[waypoint_i][1], _MOVE_WAYPOINTS[waypoint_i][2])
            params = {"x": target[0], "y": target[1], "z": target[2]}
            _relocate(bridge, "blue_container", target)  # obstacle appears exactly on the planned target
        else:
            current = runtime.robot.get_observation().robot_state.get("position", {"x": 0, "y": 0, "z": 0})
            _relocate(bridge, "blue_container", (current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0)))
        trial = _run_trial(agent, runtime, planner, task, params)
        trial["applicable"] = task == "move"
        if not trial["applicable"]:
            trial["note"] = "SafetyKernel.check() has no geometric check for this skill - no denial possible regardless of a nearby object"
        _relocate(bridge, "blue_container", _BLUE_CONTAINER_HOME)
        if trial["outcome"] == "DENY":
            trial["recovered"] = _recovery_check(agent, runtime, planner, task, params)
            if task == "move":
                waypoint_i = 1 - waypoint_i
        obstacle.append(trial)

    for _ in range(N_CAPABILITY):
        original_profiles = list(capability.env_profiles)
        capability.env_profiles = []
        if task == "move":
            params = {"x": _MOVE_WAYPOINTS[waypoint_i][0], "y": _MOVE_WAYPOINTS[waypoint_i][1], "z": _MOVE_WAYPOINTS[waypoint_i][2]}
            waypoint_i = 1 - waypoint_i
        trial = _run_trial(agent, runtime, planner, task, params)
        capability.env_profiles = original_profiles
        if trial["outcome"] == "DENY":
            trial["recovered"] = _recovery_check(agent, runtime, planner, task, params)
        capability_restriction.append(trial)

    return {"task": task, "baseline": baseline, "obstacle_insertion": obstacle, "capability_restriction": capability_restriction}


def run_emergency_stop_sequence(agent: Agent, runtime: Runtime, planner: _OneShotPlanner) -> dict[str, Any]:
    before = _run_trial(agent, runtime, planner, "move", {"x": _MOVE_WAYPOINTS[0][0], "y": _MOVE_WAYPOINTS[0][1], "z": 0.0})
    runtime.emergency_stop()
    during = _run_trial(agent, runtime, planner, "move", {"x": _MOVE_WAYPOINTS[1][0], "y": _MOVE_WAYPOINTS[1][1], "z": 0.0})
    runtime.clear_emergency_stop()
    after = _run_trial(agent, runtime, planner, "move", {"x": _MOVE_WAYPOINTS[1][0], "y": _MOVE_WAYPOINTS[1][1], "z": 0.0})
    return {
        "before_estop": before,
        "during_estop": during,
        "after_clearing_estop": after,
        "estop_blocked_action": during["outcome"] == "DENY" and during["reason"] == "emergency stop engaged",
        "recovered_after_clearing": after["outcome"] in ("ALLOW", "MODIFY"),
    }


def main() -> None:
    registry = _registry()
    planner = _OneShotPlanner()
    agent = Agent(registry, planner=planner)
    telemetry = CapturingTelemetryLogger()
    bridge = WebotsBridge()
    robot = WebotsRobot(bridge)
    safety = SafetyKernel(load_profile(PROFILE_NAME))
    runtime = Runtime(agent, robot, safety_kernel=safety, telemetry=telemetry)
    agent.set_goal("experiment 18 trial run")  # idempotently re-set per run_once() anyway

    _relocate(bridge, "blue_container", _BLUE_CONTAINER_HOME)  # known-clean starting state

    started = time.monotonic()
    results_by_task = [run_task_trials(agent, runtime, planner, registry, bridge, task) for task in TASKS]
    estop = run_emergency_stop_sequence(agent, runtime, planner)
    wall_clock_seconds = time.monotonic() - started

    def _outcome_counts(trials: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in trials:
            counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
        return counts

    all_trials = [t for task_result in results_by_task for cond in ("baseline", "obstacle_insertion", "capability_restriction") for t in task_result[cond]]
    denied_trials = [t for t in all_trials if t["outcome"] == "DENY"]
    recovery_checked = [t for t in denied_trials if "recovered" in t]

    result = {
        "experiment": "18_physical_robot_validation",
        "status": "RUN - against a real Webots physics simulation (Reach/webots), not physical hardware",
        "honest_scope": {
            "what_is_physically_real": "move and stop drive a real differential-drive control loop against real Webots physics; detect and inspect query real simulated object positions; the WebSocket transport and par_bridge.py are a genuine separate OS process, not an in-process mock.",
            "what_is_not_physically_real": "pick and place are state-only bookkeeping in par_bridge.py's own words ('state only - e-puck has no gripper') - this e-puck platform cannot physically grasp anything, regardless of what this experiment does.",
            "what_this_still_does_not_test": "real sensor noise, real actuator slip beyond what Webots' physics models, real-world unpredictability, or any consequence of a genuine physical collision - RQ12's full claim (physical hardware) remains untested after this run.",
        },
        "profile": PROFILE_NAME,
        "trial_design": f"{N_BASELINE} baseline + {N_OBSTACLE} obstacle_insertion + {N_CAPABILITY} capability_restriction per task, each denial followed by one recovery-check retry, across {len(TASKS)} tasks, plus one dedicated emergency-stop sequence",
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "results_by_task": results_by_task,
        "emergency_stop_sequence": estop,
        "summary": {
            "total_trials": len(all_trials),
            "outcome_counts": _outcome_counts(all_trials),
            "n_denied": len(denied_trials),
            "n_denied_with_recovery_checked": len(recovery_checked),
            "recovery_rate": (sum(1 for t in recovery_checked if t["recovered"]) / len(recovery_checked)) if recovery_checked else None,
            "obstacle_insertion_applicable_only_to": "move (see per-trial 'applicable'/'note' fields for the other 5 tasks)",
            "emergency_stop_worked": estop["estop_blocked_action"] and estop["recovered_after_clearing"],
        },
        "metrics_not_captured_this_run": {
            "physical_collision_occurrence": "no real collision is possible to measure - this is a simulation; the Safety Kernel's collision_margin DENY count is the closest proxy and is in outcome_counts above",
            "perception_to_policy_latency": "Runtime._step() doesn't expose a perception/policy sub-breakdown hook; only total per-step latency_seconds is available without modifying runtime.py, which was out of scope for this experiment",
        },
    }

    out_path = Path(__file__).parent.parent / "results" / "exp18_physical_robot_validation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
