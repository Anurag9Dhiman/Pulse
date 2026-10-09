"""Standalone rclpy node standing in for the physical robot side of
Experiment 17's live ROS 2 integration test.

Runs as a separate OS process from the PAR side (par_driver.py) - the two
communicate only over real ROS 2 topics/DDS, not in-process function calls -
so this is a genuine transport-boundary test, not a mocked one dressed up to
look like it. Publishes a fixed robot_state and one detected obstacle on a
timer (mirroring MockRobot's fixture shape so payload_to_observation() has
real data to parse), and records every message it receives on /par/action to
a JSONL trace file, which par_driver.py cross-checks afterward to verify
nothing PAR denied ever actually reached this process.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

OBSTACLE_POSITION = {"x": 0.5, "y": 0.5, "z": 0.0}


class MockRos2Robot(Node):
    def __init__(self, trace_path: Path) -> None:
        super().__init__("mock_ros2_robot")
        self._trace_path = trace_path
        self._trace_path.write_text("")  # truncate any stale trace from a prior run

        self._state_pub = self.create_publisher(String, "/par/robot_state", 10)
        self._detections_pub = self.create_publisher(String, "/par/detections", 10)
        self.create_subscription(String, "/par/action", self._on_action, 10)
        self.create_timer(0.2, self._publish_environment)
        self.get_logger().info(f"mock_ros2_robot up, tracing received actions to {trace_path}")

    def _publish_environment(self) -> None:
        state_msg = String(data=json.dumps({"position": {"x": 0.0, "y": 0.0, "z": 0.0}}))
        detections_msg = String(data=json.dumps([{"name": "obstacle_1", "position": OBSTACLE_POSITION}]))
        self._state_pub.publish(state_msg)
        self._detections_pub.publish(detections_msg)

    def _on_action(self, msg: String) -> None:
        record = {"received_at": datetime.now(timezone.utc).isoformat(), "payload": json.loads(msg.data)}
        with self._trace_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.get_logger().info(f"received action on /par/action: {record['payload']}")


def main() -> None:
    trace_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/par_ros2_action_trace.jsonl")
    rclpy.init()
    node = MockRos2Robot(trace_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
