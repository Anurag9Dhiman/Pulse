from __future__ import annotations

import os

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.integrations.collectiveos import CollectiveOSBridge
from par.integrations.simulated_arm_computer_use import SimulatedArmBridge
from par.integrations.vision_guided_arm import VisionGuidedArmBridge
from par.robots.base import RobotInterface
from par.skills.computer_use import DEFAULT_TIMEOUT_SECONDS

_COMPUTER_USE_SKILL = "use_computer"

_ComputerUseBridge = CollectiveOSBridge | SimulatedArmBridge | VisionGuidedArmBridge


def _default_bridge() -> _ComputerUseBridge:
    """Which bridge a use_computer delegation actually reaches, when the
    caller doesn't pass one explicitly - PAR_COMPUTER_USE_MODE=simulated_arm
    routes to computer_arm's physical gantry with keyword-matched button
    choice, vision_guided_arm routes to the same gantry but with Gemini
    deciding the button from a real camera image. Defaults to "collectiveos"
    so existing callers/examples are unaffected."""
    mode = os.environ.get("PAR_COMPUTER_USE_MODE", "collectiveos")
    if mode == "simulated_arm":
        return SimulatedArmBridge()
    if mode == "vision_guided_arm":
        return VisionGuidedArmBridge()
    return CollectiveOSBridge()


class ComputerAugmentedRobot(RobotInterface):
    """Wraps a RobotInterface, routing `use_computer` actions to a
    computer-use bridge (real CollectiveOS by default, or computer_arm's
    physical gantry - see _default_bridge) and everything else to the
    wrapped robot unchanged."""

    def __init__(self, robot: RobotInterface, bridge: _ComputerUseBridge | None = None) -> None:
        self._robot = robot
        self._bridge = bridge or _default_bridge()

    def get_observation(self) -> Observation:
        return self._robot.get_observation()

    def execute(self, action: Action) -> ActionResult:
        if action.skill_name != _COMPUTER_USE_SKILL:
            return self._robot.execute(action)

        # Optional hook: robots that can physically represent "at the
        # computer" (e.g. WebotsRobot docking near a laptop prop) implement
        # begin_/end_computer_use(); robots that can't (MockRobot, ROS2Robot)
        # simply don't have the attribute, so this is a no-op for them.
        begin_computer_use = getattr(self._robot, "begin_computer_use", None)
        if begin_computer_use is not None:
            begin_computer_use()

        task = action.parameters.get("task", "")
        result = self._bridge.run_task(action, task, timeout=DEFAULT_TIMEOUT_SECONDS)

        end_computer_use = getattr(self._robot, "end_computer_use", None)
        if end_computer_use is not None:
            end_computer_use()

        return result
