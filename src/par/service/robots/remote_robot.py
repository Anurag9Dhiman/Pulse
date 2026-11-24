from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.robots.base import RobotInterface
from par.robots.ros2_mapping import action_to_payload, payload_to_observation
from par.service.robots.connection_registry import RobotConnection, RobotConnectionError


class RemoteRobot(RobotInterface):
    """A robot reached over an inbound WebSocket: the robot's bridge dials
    the service, the reverse of WebotsBridge/CollectiveOSBridge.

    Reuses the ros2_mapping wire schema unchanged. Never raises: a dead or
    silent connection becomes an empty observation with raw.error set, or a
    failed ActionResult, the same convention WebotsBridge follows. That
    empty observation now also carries both perception channels marked
    unavailable (inferred from raw.error by ros2_mapping), so the Safety
    Kernel denies motion on it rather than reading a dropped connection as
    a clear workspace.
    """

    def __init__(self, connection: RobotConnection, request_timeout: float = 30.0) -> None:
        self._connection = connection
        self._request_timeout = request_timeout

    def get_observation(self) -> Observation:
        try:
            reply = self._connection.send_request({"type": "get_observation"}, self._request_timeout)
        except RobotConnectionError as exc:
            return payload_to_observation({"raw": {"error": str(exc)}})
        return payload_to_observation(
            {
                "robot_state": reply.get("robot_state") or {},
                "detections": reply.get("detections") or [],
                # Forwarded, not dropped: a tenant's bridge is the least
                # trustworthy sensor source in the system, so when it
                # declares a channel dead the Safety Kernel has to hear it.
                # Dropping the block left a remote robot with a failed
                # camera reporting an empty, apparently healthy world.
                "channels": reply.get("channels") or {},
                "raw": reply.get("raw") or {},
            }
        )

    def execute(self, action: Action) -> ActionResult:
        try:
            reply = self._connection.send_request(
                {"type": "action", **action_to_payload(action)}, self._request_timeout
            )
            success, message = bool(reply.get("success", False)), str(reply.get("message", ""))
        except RobotConnectionError as exc:
            success, message = False, str(exc)
        return ActionResult(
            action_id=action.action_id,
            success=success,
            message=message,
            completed_at=datetime.now(timezone.utc),
        )
