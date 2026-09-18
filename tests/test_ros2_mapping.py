from datetime import datetime, timezone

import pytest

from par.core.action import Action
from par.robots.ros2_mapping import action_to_payload, payload_to_observation


def test_payload_to_observation_maps_fields():
    obs = payload_to_observation({"robot_state": {"x": 1.0}, "detections": [{"name": "red_object"}]})
    assert obs.source == "ros2"
    assert obs.robot_state == {"x": 1.0}
    assert obs.detections == [{"name": "red_object"}]


def test_payload_to_observation_defaults_missing_fields():
    obs = payload_to_observation({})
    assert obs.robot_state == {}
    assert obs.detections == []


def test_action_to_payload_round_trip_fields():
    action = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "red_object"},
        created_at=datetime.now(timezone.utc),
    )
    payload = action_to_payload(action)
    assert payload == {
        "action_id": "a1",
        "skill_name": "pick",
        "parameters": {"object": "red_object"},
    }


def test_ros2_robot_raises_clear_error_without_rclpy():
    import par.robots.ros2_adapter as ros2_adapter

    if ros2_adapter.rclpy is not None:
        pytest.skip("rclpy is installed in this environment")
    with pytest.raises(RuntimeError, match="ROS 2"):
        ros2_adapter.ROS2Robot()
