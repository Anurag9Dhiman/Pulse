from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action
from par.core.observation import Observation
from par.robots.mujoco_bridge import MuJoCoRobot


def _action(skill_name: str, parameters: dict) -> Action:
    return Action(action_id="a1", skill_name=skill_name, parameters=parameters, created_at=datetime.now(timezone.utc))


class _FakeBridge:
    def __init__(self, observation_payload: dict, action_reply: dict) -> None:
        self._observation_payload = observation_payload
        self._action_reply = action_reply
        self.sent_actions: list[dict] = []

    def get_observation_payload(self) -> dict:
        return self._observation_payload

    def send_action(self, payload: dict) -> dict:
        self.sent_actions.append(payload)
        return self._action_reply


def test_get_observation_routes_through_ros2_mapping():
    bridge = _FakeBridge(
        observation_payload={
            "robot_state": {"position": {"x": 0.55, "y": 0.0, "z": 0.62}},  # Panda EE at home
            "detections": [{"name": "red_object"}, {"name": "blue_container"}, {"name": "laptop"}],
        },
        action_reply={},
    )
    robot = MuJoCoRobot(bridge=bridge)

    observation = robot.get_observation()

    assert isinstance(observation, Observation)
    assert observation.source == "ros2"  # same schema real ROS 2 would carry
    assert observation.robot_state == {"position": {"x": 0.55, "y": 0.0, "z": 0.62}}
    assert len(observation.detections) == 3


def test_get_observation_surfaces_bridge_error_via_raw():
    bridge = _FakeBridge(observation_payload={"raw": {"error": "MuJoCo bridge unreachable"}}, action_reply={})
    robot = MuJoCoRobot(bridge=bridge)

    observation = robot.get_observation()

    assert observation.robot_state == {}
    assert observation.raw == {"error": "MuJoCo bridge unreachable"}


def test_execute_sends_action_payload_and_builds_result():
    bridge = _FakeBridge(
        observation_payload={},
        action_reply={"success": True, "message": "moved toward (0.50, 0.20) via 'par_near_red'"},
    )
    robot = MuJoCoRobot(bridge=bridge)

    result = robot.execute(_action("move", {"x": 0.5, "y": 0.2, "z": 0.0}))

    assert bridge.sent_actions == [{"action_id": "a1", "skill_name": "move", "parameters": {"x": 0.5, "y": 0.2, "z": 0.0}}]
    assert result.success is True
    assert "par_near_red" in result.message


def test_execute_reports_failure_from_bridge():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": False, "message": "MuJoCo bridge connection failed"})
    robot = MuJoCoRobot(bridge=bridge)

    result = robot.execute(_action("stop", {}))

    assert result.success is False
    assert result.message == "MuJoCo bridge connection failed"


def test_begin_computer_use_sends_dock_at_laptop_action():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": True, "message": "docked at laptop"})
    robot = MuJoCoRobot(bridge=bridge)

    robot.begin_computer_use()

    assert len(bridge.sent_actions) == 1
    sent = bridge.sent_actions[0]
    assert sent["skill_name"] == "dock_at_laptop"
    assert sent["parameters"] == {}
    assert "action_id" in sent


def test_end_computer_use_sends_undock_from_laptop_action():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": True, "message": "undocked (returned to home)"})
    robot = MuJoCoRobot(bridge=bridge)

    robot.end_computer_use()

    assert len(bridge.sent_actions) == 1
    assert bridge.sent_actions[0]["skill_name"] == "undock_from_laptop"


def test_dock_hooks_never_raise_even_if_bridge_reports_failure():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": False, "message": "'laptop' prop not found"})
    robot = MuJoCoRobot(bridge=bridge)
    robot.begin_computer_use()
    robot.end_computer_use()
