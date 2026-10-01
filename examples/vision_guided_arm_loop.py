"""PAR driving computer_arm's physically-real gantry for use_computer, with
Gemini deciding WHICH button to press by looking at a real camera image of
the kiosk panel - not keyword-matching the task text (see
examples/simulated_arm_loop.py for that cheaper, offline alternative). Same
physical arena tour as simulated_arm_loop.py; only the button-choice logic
differs.

Requires:
    pip install -e ".[webots]"
    GEMINI_API_KEY - this path spends real Gemini quota (gemini-3.1-flash-lite,
        one call per use_computer delegation), unlike simulated_arm_loop.py.
    WEBOTS_BRIDGE_URL (optional) - default ws://localhost:6001
    COMPUTER_ARM_BRIDGE_URL (optional) - default ws://localhost:6002

And, running beforehand (see Reach/webots/README.md):
    1. Webots open on Reach/webots/worlds/par_arena.wbt, simulation running.
    2. Reach/webots/controllers/par_bridge/par_bridge.py running in its own
       terminal.
    3. Reach/webots/controllers/computer_arm_bridge/computer_arm_bridge.py
       running in its own terminal.
    Both (2) and (3) must be connected before the simulation will step at
    all - Webots holds the whole world at t=0 until every extern-controller
    robot has connected, not just the one you're testing (see
    computer_arm_bridge.py's module docstring).

PAR_COMPUTER_USE_MODE=vision_guided_arm selects VisionGuidedArmBridge as the
use_computer bridge (see par/robots/computer_bridge.py::_default_bridge) -
set as this process's own env var below rather than requiring the caller to
export it, so this example is copy-paste runnable on its own.
"""
import os

os.environ.setdefault("PAR_COMPUTER_USE_MODE", "vision_guided_arm")

from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

from par.core.action import Action, ActionResult  # noqa: E402
from par.core.agent import Agent  # noqa: E402
from par.core.observation import Observation  # noqa: E402
from par.core.planner import TASK_COMPLETE, Planner  # noqa: E402
from par.core.runtime import Runtime  # noqa: E402
from par.core.skill import SkillRegistry  # noqa: E402
from par.robots.computer_bridge import ComputerAugmentedRobot  # noqa: E402
from par.robots.webots_bridge import WebotsRobot  # noqa: E402
from par.safety.environment import load_profile  # noqa: E402
from par.safety.kernel import SafetyKernel  # noqa: E402
from par.skills import builtin_skills, computer_use_skill  # noqa: E402

_MOVE_TIMEOUT_SECONDS = 20.0
_RED_OBJECT = (0.5, 0.2, 0.0)
_BLUE_CONTAINER = (-0.3, 0.4, 0.0)

_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("detect", {}),
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1] + 0.4, "z": 0.0}),  # near red_object: allowed
    ("move", {"x": _RED_OBJECT[0], "y": _RED_OBJECT[1], "z": 0.0}),  # onto red_object: denied (collision)
    ("move", {"x": _BLUE_CONTAINER[0], "y": _BLUE_CONTAINER[1] + 0.4, "z": 0.0}),  # near blue_container: allowed
    # Physical task done - now delegate a digital subtask. Watch the e-puck
    # dock at computer_arm, then watch Gemini look at a real photo of the
    # kiosk panel and decide which button to press, before the gantry
    # presses it for real, confirmed by its TouchSensor.
    ("use_computer", {"task": "check whether any maintenance alerts are open"}),
    ("stop", {}),
    (TASK_COMPLETE, {"message": "toured the arena and let Gemini vision choose and press a real button"}),
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

    robot = ComputerAugmentedRobot(WebotsRobot())
    agent = Agent(registry, planner=_ScriptedPlanner(_STEPS))
    safety = SafetyKernel(load_profile("simulation"))
    runtime = Runtime(agent, robot, safety_kernel=safety)

    runtime.run_task("tour the arena", max_steps=len(_STEPS))


if __name__ == "__main__":
    main()
