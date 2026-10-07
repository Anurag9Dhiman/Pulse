from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.integrations.mujoco import MuJoCoBridge
from par.robots.base import RobotInterface
from par.robots.ros2_mapping import action_to_payload, payload_to_observation


class MuJoCoRobot(RobotInterface):
    """Drives a Franka Emika Panda arm in MuJoCo via the mujoco_bridge
    (Reach/mujoco/bridge/mujoco_bridge.py), instead of MockRobot's in-memory
    stand-in.

    Reuses par.robots.ros2_mapping's payload<->Observation/Action mapping
    unchanged - the wire schema matches what ROS 2 would carry, only the
    transport (WebSocket) differs. The robot's reported "position" is the
    Panda's end-effector world position (the thing that would actually
    collide with a detected object), so the Safety Kernel's collision
    check behaves the same way it did against the e-puck's base.
    """

    def __init__(self, bridge: MuJoCoBridge | None = None) -> None:
        self._bridge = bridge or MuJoCoBridge()

    def get_observation(self) -> Observation:
        return payload_to_observation(self._bridge.get_observation_payload())

    def execute(self, action: Action) -> ActionResult:
        reply = self._bridge.send_action(action_to_payload(action))
        return ActionResult(
            action_id=action.action_id,
            success=reply["success"],
            message=reply["message"],
            completed_at=datetime.now(timezone.utc),
        )

    def begin_computer_use(self) -> None:
        """Best-effort: dock the Panda's end-effector at the laptop prop so
        a use_computer delegation is visibly happening in the simulation.
        Never raises and its result is not checked - a missing laptop prop
        or unresponsive bridge must never block the actual (safety-relevant)
        computer-use call this wraps, only skip its physical visualization."""
        self._bridge.send_action({"action_id": str(uuid4()), "skill_name": "dock_at_laptop", "parameters": {}})

    def end_computer_use(self) -> None:
        self._bridge.send_action(
            {"action_id": str(uuid4()), "skill_name": "undock_from_laptop", "parameters": {}}
        )
