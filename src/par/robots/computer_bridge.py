from __future__ import annotations

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.integrations.collectiveos import CollectiveOSBridge
from par.robots.base import RobotInterface
from par.skills.computer_use import DEFAULT_TIMEOUT_SECONDS

_COMPUTER_USE_SKILL = "use_computer"


class ComputerAugmentedRobot(RobotInterface):
    """Wraps a RobotInterface, routing `use_computer` actions to CollectiveOS
    and everything else to the wrapped robot unchanged."""

    def __init__(self, robot: RobotInterface, bridge: CollectiveOSBridge | None = None) -> None:
        self._robot = robot
        self._bridge = bridge or CollectiveOSBridge()

    def get_observation(self) -> Observation:
        return self._robot.get_observation()

    def execute(self, action: Action) -> ActionResult:
        if action.skill_name != _COMPUTER_USE_SKILL:
            return self._robot.execute(action)
        task = action.parameters.get("task", "")
        return self._bridge.run_task(action, task, timeout=DEFAULT_TIMEOUT_SECONDS)
