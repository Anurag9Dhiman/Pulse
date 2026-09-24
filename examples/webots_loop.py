"""PAR driving a real physically-simulated robot (Webots e-puck) instead of
MockRobot, so `move` and the Safety Kernel's behavior can be watched live.

Requires:
    pip install -e ".[webots]"
    WEBOTS_BRIDGE_URL (optional) - default ws://localhost:6001

And, running beforehand (see Reach/webots/README.md):
    1. Webots open on Reach/webots/worlds/par_arena.wbt, simulation running.
    2. Reach/webots/controllers/par_bridge/par_bridge.py running in its own
       terminal (with WEBOTS_HOME set) - it waits for this script to connect.

No API key needed - uses a small scripted planner rather than the default
RuleBasedPlanner, because RuleBasedPlanner._extract_params("move", ...)
always returns (0.0, 0.0, 0.0) (see par/core/planner.py); it can't turn
"move to red_object" into that object's real coordinates. The scripted steps
below target real arena coordinates instead, including one move placed
exactly on top of red_object so the Safety Kernel's collision-margin check
denies it live - the same Use Case 4 behavior examples/safety_demo.py shows
against MockRobot, now against a real simulated robot.
"""
from datetime import datetime, timezone
from typing import Any

from par.core.action import Action, ActionResult
from par.core.agent import Agent
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.webots_bridge import WebotsRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills

# e-puck's real driving speed is much slower than simulation.yaml's 5s
# action_timeout_seconds assumes for an instant MockRobot move - override it
# for this example only, via the same per-capability mechanism used for
# use_computer. builtin.py's default stays correct for MockRobot.
_MOVE_TIMEOUT_SECONDS = 20.0

# Arena coordinates match MockRobot's defaults (par/robots/mock.py) and
# Reach/webots/worlds/par_arena.wbt's object placement.
_RED_OBJECT = (0.5, 0.2, 0.0)
_BLUE_CONTAINER = (-0.3, 0.4, 0.0)

_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("detect", {}),
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1] + 0.4, "z": 0.0}),  # near red_object: allowed
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1], "z": 0.0}),  # onto red_object: denied (collision)
    ("move", {"x": _BLUE_CONTAINER[0], "y": _BLUE_CONTAINER[1] + 0.4, "z": 0.0}),  # near blue_container: allowed
    ("stop", {}),
    (TASK_COMPLETE, {"message": "toured the arena"}),
]


class _ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal: str, observation: Observation, capabilities) -> tuple[str, dict[str, Any]]:
        return self._steps.pop(0)

    def record_result(self, action: Action, result: ActionResult) -> None:
        print(f"  {action.skill_name}({action.parameters}) -> success={result.success}  {result.message}")


def main() -> None:
    registry = SkillRegistry()
    for skill in builtin_skills():
        if skill.name == "move":
            skill.capability.execution_timeout_seconds = _MOVE_TIMEOUT_SECONDS
        registry.register(skill)

    robot = WebotsRobot()
    agent = Agent(registry, planner=_ScriptedPlanner(_STEPS))
    safety = SafetyKernel(load_profile("simulation"))
    runtime = Runtime(agent, robot, safety_kernel=safety)

    runtime.run_task("tour the arena", max_steps=len(_STEPS))


if __name__ == "__main__":
    main()
