from __future__ import annotations

from par.core.action import ActionResult
from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

GOAL = "Pick the red_object and place it in the blue_container."


class _DemoPlanner(Planner):
    """Pre-scripted golden-path-then-obstacle sequence for a live, hardware-free
    walkthrough of the full PAR loop. This stands in for the LLM/VLM planner
    (Week 2) exactly like RuleBasedPlanner does elsewhere in the codebase -
    the mechanism being demonstrated (Safety Kernel denial -> re-planning) is
    real; only "what to try next" is scripted here instead of reasoned live.
    """

    def __init__(self) -> None:
        self._steps: list[tuple[str, dict]] = [
            ("pick", {"object": "red_object"}),
            ("place", {"target": "blue_container"}),
            ("move", {"x": 1.0, "y": 1.0, "z": 0.0}),  # will be blocked by an obstacle
            ("move", {"x": 1.0, "y": -1.0, "z": 0.0}),  # re-planned, safe alternative
            (TASK_COMPLETE, {"message": "demo complete"}),
        ]

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


def _print_step(index: int, result: ActionResult) -> None:
    status = "APPROVED" if result.success else "REJECTED"
    print(f"  [{index}] {status:8} {result.message}")


def run() -> int:
    print("Physical Agent Runtime - Live Demo\n")
    print(f'Task: "{GOAL}"\n')

    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)

    robot = MockRobot()
    safety = SafetyKernel(load_profile("simulation"))
    agent = Agent(registry, planner=_DemoPlanner())
    runtime = Runtime(agent, robot, safety_kernel=safety)

    approved = 0
    rejected = 0

    def step(index: int) -> ActionResult:
        nonlocal approved, rejected
        result = runtime.run_once(GOAL)
        _print_step(index, result)
        if result.success:
            approved += 1
        else:
            rejected += 1
        return result

    print("Golden path:")
    step(1)  # pick
    step(2)  # place

    print("\nEnvironment change: an obstacle appears at (1.0, 1.0, 0.0)")
    robot.add_obstacle("obstacle_1", 1.0, 1.0, 0.0)

    print("\nAgent attempts its planned waypoint - now blocked by the obstacle:")
    step(3)  # move, denied: collision

    print("\nRe-planned to a safe alternative waypoint:")
    step(4)  # move, approved

    final = runtime.run_once(GOAL)  # task_complete -> None
    assert final is None

    print("\n" + "=" * 50)
    print("TRACEABILITY")
    print(f"  actions requested: {approved + rejected}")
    print(f"  approved:          {approved}")
    print(f"  rejected:          {rejected}")
    print(f"  result:            {'SUCCESS' if rejected <= 1 else 'PARTIAL'}")
    print("=" * 50)

    runtime.close()
    return 0
