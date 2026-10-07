from __future__ import annotations

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.integrations.collectiveos import CollectiveOSBridge
from par.robots.base import RobotInterface
from par.skills.computer_use import DEFAULT_TIMEOUT_SECONDS

_COMPUTER_USE_SKILL = "use_computer"


class ComputerAugmentedRobot(RobotInterface):
    """Wraps a RobotInterface, routing `use_computer` actions to a
    CollectiveOS bridge and everything else to the wrapped robot unchanged.

    Earlier revisions supported a PAR_COMPUTER_USE_MODE env var that could
    swap in alternative "physically-real" bridges (gantry-pressing-buttons
    variants) alongside CollectiveOS; those bridges and the mode switch
    were removed when Webots was retired - the current MuJoCo arm
    integration keeps the governance+delegation story clean by using one
    bridge (real CollectiveOS) end to end, with the arm's visible dock-at-
    laptop keyframe standing in for the physical hand-off.
    """

    def __init__(self, robot: RobotInterface, bridge: CollectiveOSBridge | None = None) -> None:
        self._robot = robot
        self._bridge = bridge or CollectiveOSBridge()

    def get_observation(self) -> Observation:
        return self._robot.get_observation()

    def execute(self, action: Action) -> ActionResult:
        if action.skill_name != _COMPUTER_USE_SKILL:
            return self._robot.execute(action)

        # Optional hook: robots that can physically represent "at the
        # computer" (e.g. MuJoCoRobot docking its end-effector at the
        # laptop prop) implement begin_/end_computer_use(); robots that
        # can't (MockRobot, ROS2Robot) simply don't have the attribute,
        # so this is a no-op for them.
        begin_computer_use = getattr(self._robot, "begin_computer_use", None)
        if begin_computer_use is not None:
            begin_computer_use()

        task = action.parameters.get("task", "")
        result = self._bridge.run_task(action, task, timeout=DEFAULT_TIMEOUT_SECONDS)

        end_computer_use = getattr(self._robot, "end_computer_use", None)
        if end_computer_use is not None:
            end_computer_use()

        return result
