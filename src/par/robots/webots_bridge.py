from __future__ import annotations

from datetime import datetime, timezone

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
