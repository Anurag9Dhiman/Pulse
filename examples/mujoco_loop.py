"""PAR driving a Franka Emika Panda arm in MuJoCo (instead of MockRobot) so
`move`, the Safety Kernel's behavior, and a real use_computer delegation to
CollectiveOS can all be watched live.

Replaces examples/webots_loop.py as the project's physical-simulation
example. Same tour-and-delegate shape (detect -> near red -> collision
denial -> near blue -> use_computer -> stop -> task_complete), only the
simulator and robot change: a 7-DOF arm swings to each target instead of
an e-puck rolling to it, and the dock-at-computer step moves the arm's
end-effector to the laptop prop instead of lighting an LED.

Requires:
    pip install -e ".[mujoco]"
    MUJOCO_BRIDGE_URL (optional) - default ws://localhost:6003

And, running beforehand (see Reach/mujoco/README.md):
    1. Reach/mujoco/bridge/mujoco_bridge.py running in its own terminal.
       Interactive mode (what shows the arena): launch with `mjpython ...`,
       not plain `python`, on macOS; headless mode (--headless) works with
       plain `python` and is used by CI.

The last two tour steps delegate to CollectiveOS via use_computer, wrapped
in ComputerAugmentedRobot. MuJoCoRobot.begin_/end_computer_use() drive the
arm's end-effector to the simulated laptop prop and back - CollectiveOS
does not need to be running for the physical docking to happen; if it isn't,
run_task simply comes back as a failed ActionResult and the robot still
undocks and continues.

No API key needed - uses a small scripted planner rather than the default
RuleBasedPlanner, because RuleBasedPlanner._extract_params("move", ...)
always returns (0.0, 0.0, 0.0) (see par/core/planner.py); it can't turn
"move to red_object" into that object's real coordinates. The scripted steps
below target real arena coordinates instead, including one move placed
exactly on top of red_object so the Safety Kernel's collision-margin check
denies it live - the same Use Case 4 behavior examples/safety_demo.py shows
against MockRobot, now against a real simulated arm.
"""
from datetime import datetime, timezone
from typing import Any

from par.core.action import Action, ActionResult
from par.core.agent import Agent
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.computer_bridge import ComputerAugmentedRobot
from par.robots.mujoco_bridge import MuJoCoRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills, computer_use_skill

# The Panda's joint-space moves take 1-2s in the viewer (real-time pacing)
# versus the ~5s simulation.yaml allows for an instant MockRobot move -
# override as e-puck's example does, same per-capability mechanism.
_MOVE_TIMEOUT_SECONDS = 20.0

# Arena coordinates match MockRobot's defaults (par/robots/mock.py) and
# the authored positions in Reach/mujoco/scenes/par_arena.py.
_RED_OBJECT = (0.5, 0.2, 0.0)
_BLUE_CONTAINER = (-0.3, 0.4, 0.0)

_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("detect", {}),
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1] + 0.4, "z": 0.0}),  # near red_object: allowed
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1], "z": 0.0}),  # onto red_object: denied (collision)
    ("move", {"x": _BLUE_CONTAINER[0], "y": _BLUE_CONTAINER[1] + 0.4, "z": 0.0}),  # near blue_container: allowed
    # Real manipulation: arm picks the red cube up, carries it, drops it in the blue bowl.
    ("pick_and_place", {"object": "red_object", "target": "blue_container"}),
    # Physical task done - now delegate a digital subtask. Watch the arm's
    # end-effector move to the laptop prop; whether CollectiveOS itself
    # succeeds depends on whether it's actually running.
    ("use_computer", {"task": "which application is currently in the foreground?"}),
    ("stop", {}),
    (TASK_COMPLETE, {"message": "toured, picked and placed the red cube, delegated a digital subtask"}),
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
    registry.register(computer_use_skill())

    robot = ComputerAugmentedRobot(MuJoCoRobot())
    agent = Agent(registry, planner=_ScriptedPlanner(_STEPS))
    safety = SafetyKernel(load_profile("simulation"))
    runtime = Runtime(agent, robot, safety_kernel=safety)

    runtime.run_task("tour the arena", max_steps=len(_STEPS))


if __name__ == "__main__":
    main()
