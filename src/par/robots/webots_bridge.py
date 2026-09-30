from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.integrations.webots import WebotsBridge
from par.robots.base import RobotInterface
from par.robots.ros2_mapping import action_to_payload, payload_to_observation


class WebotsRobot(RobotInterface):
    """Drives a real physically-simulated robot (e-puck, by default) via the
    Webots extern-controller bridge (Reach/webots/controllers/par_bridge),
    instead of MockRobot's in-memory stand-in.

    Reuses par.robots.ros2_mapping's payload<->Observation/Action mapping
    unchanged - the wire schema matches what real ROS 2 would carry, only
    the transport (WebSocket, not ROS 2 topics) differs.
    """

    def __init__(self, bridge: WebotsBridge | None = None) -> None:
        self._bridge = bridge or WebotsBridge()

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
        """Best-effort: drive to the world's "laptop" prop and light an LED,
        so a use_computer delegation is visible in the simulation, not just a
        WebSocket call happening invisibly off to the side. Never raises and
        its result is not checked - a missing prop or unresponsive bridge
        must never block the actual (safety-relevant) computer-use call this
        wraps, only skip its physical visualization."""
        self._bridge.send_action({"action_id": str(uuid4()), "skill_name": "dock_at_computer", "parameters": {}})

    def end_computer_use(self) -> None:
        self._bridge.send_action({"action_id": str(uuid4()), "skill_name": "undock_from_computer", "parameters": {}})
