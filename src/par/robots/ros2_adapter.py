from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.observation import Observation
from par.robots.base import RobotInterface
from par.robots.ros2_mapping import action_to_payload, payload_to_observation

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - exercised only where ROS 2 isn't installed
    rclpy = None
    Node = None
    String = None


class ROS2Robot(RobotInterface):
    """Bridges PAR to ROS 2 over JSON-encoded std_msgs/String topics.

    Message types are intentionally minimal for the MVP: the exact
    sensor/actuator message types depend on the physical robot chosen in
    Week 3. Only the pure JSON mapping (ros2_mapping.py) is unit-tested in
    this environment - no ROS 2 installation is available here, so the
    pub/sub node lifecycle below must be verified on a machine with ROS 2
    Jazzy or Humble installed before it's trusted.
    """

    def __init__(
        self,
        state_topic: str = "/par/robot_state",
        detections_topic: str = "/par/detections",
        command_topic: str = "/par/action",
        node_name: str = "par_robot_adapter",
        observation_timeout: float = 2.0,
    ) -> None:
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 (rclpy) is not installed or not sourced. Install ROS 2 "
                "Jazzy/Humble and source its setup.bash before using ROS2Robot."
            )
        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node(node_name)
        self._observation_timeout = observation_timeout
        self._latest_payload: dict = {"robot_state": {}, "detections": []}

        self._node.create_subscription(String, state_topic, self._on_state, 10)
        self._node.create_subscription(String, detections_topic, self._on_detections, 10)
        self._command_pub = self._node.create_publisher(String, command_topic, 10)

    def _on_state(self, msg: "String") -> None:
        self._latest_payload["robot_state"] = json.loads(msg.data)

    def _on_detections(self, msg: "String") -> None:
        self._latest_payload["detections"] = json.loads(msg.data)

    def get_observation(self) -> Observation:
        deadline = time.monotonic() + self._observation_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        return payload_to_observation(self._latest_payload)

    def execute(self, action: Action) -> ActionResult:
        msg = String(data=json.dumps(action_to_payload(action)))
        self._command_pub.publish(msg)
        # Fire-and-forget for the MVP: no completion-acknowledgement channel yet.
        # Real success/failure confirmation arrives with the physical robot (Week 3).
        return ActionResult(
            action_id=action.action_id,
            success=True,
            message="published to ROS 2",
            completed_at=datetime.now(timezone.utc),
        )

    def shutdown(self) -> None:
        self._node.destroy_node()
