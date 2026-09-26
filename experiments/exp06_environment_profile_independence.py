"""Experiment 6: Environment Profile Independence (RQ6).

Can the same agent, planner, and skill implementation operate under
different deployment constraints by changing only the environment profile?

Runs the identical scripted task through the identical Agent/Runtime code
against all 4 shipped profiles (relaxed simulation, strict simulation,
physical-robot, high-safety) - only the SafetyKernel's profile changes.
Skills are registered for all 4 profiles at once (representing a capability
package deployed across multiple environments), not re-registered per run.
"""
from __future__ import annotations

import json
from pathlib import Path

from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.evaluation.capture import CapturingTelemetryLogger
from par.evaluation.metrics import latency_stats
from par.evaluation.verifiers import verify_pick_and_place
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

PROFILE_NAMES = ["simulation", "strict_simulation", "real_robot", "high_safety"]
ALL_PROFILES_LIST = PROFILE_NAMES  # capability registration target


class _FixedTaskPlanner(Planner):
    """Same scripted task for every profile - only the SafetyKernel differs."""

    def __init__(self) -> None:
        self._steps = [
            ("pick", {"object": "red_object"}),
            ("place", {"target": "blue_container"}),
            ("move", {"x": 0.3, "y": 0.3, "z": 0.0}),
            (TASK_COMPLETE, {"message": "done", "success": True}),
        ]

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


def _registry_for_all_profiles() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = list(ALL_PROFILES_LIST)
        registry.register(skill)
    return registry


def run() -> dict:
    goal_verifier = verify_pick_and_place("red_object", "blue_container")
    results = []

    for profile_name in PROFILE_NAMES:
        profile = load_profile(profile_name)
        registry = _registry_for_all_profiles()
        robot = MockRobot()
        agent = Agent(registry, planner=_FixedTaskPlanner())
        telemetry = CapturingTelemetryLogger()
        runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(profile), telemetry=telemetry)

        task_results = runtime.run_task("pick red_object, place in blue_container, move to (0.3, 0.3, 0)", max_steps=10)
        runtime.close()

        results.append(
            {
                "profile": profile_name,
                "n_actions_attempted": len(task_results),
                "n_succeeded": sum(1 for r in task_results if r.success),
                "n_failed": sum(1 for r in task_results if not r.success),
                "safety_outcome_counts": telemetry.outcome_counts(),
                "latency_stats": latency_stats(telemetry.latencies()),
                "agent_status": agent.state.status.value,
                "reported_success": agent.state.reported_success,
                "verified_success": goal_verifier(runtime.world_state.latest_observation),
                "final_position": robot.get_observation().robot_state.get("position"),
            }
        )

    return {
        "experiment": "6_environment_profile_independence",
        "profiles_tested": PROFILE_NAMES,
        "results_by_profile": results,
        "same_pick_place_succeeded_all_profiles": all(r["verified_success"] for r in results),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp06_environment_profile_independence.json"
    out_path.write_text(json.dumps(result, indent=2))
