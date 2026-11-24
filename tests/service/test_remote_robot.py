from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from par.core.action import Action
from par.service.robots.connection_registry import RobotConnectionError
from par.service.robots.remote_robot import RemoteRobot


class _FakeConnection:
    def __init__(self, replies: list[dict[str, Any]] | None = None, error: str | None = None) -> None:
        self._replies = list(replies or [])
        self._error = error
        self.requests: list[dict[str, Any]] = []

    def send_request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.requests.append(payload)
        if self._error is not None:
            raise RobotConnectionError(self._error)
        return self._replies.pop(0)


def _action() -> Action:
    return Action(
        action_id="a1",
        skill_name="move",
        parameters={"x": 0.1, "y": 0.0, "z": 0.0},
        created_at=datetime.now(timezone.utc),
    )


def test_get_observation_maps_the_bridge_reply_through_the_shared_wire_schema():
    connection = _FakeConnection(
        [
            {
                "type": "observation",
                "robot_state": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
                "detections": [{"name": "red_object"}],
            }
        ]
    )

    observation = RemoteRobot(connection).get_observation()

    assert connection.requests == [{"type": "get_observation"}]
    assert observation.robot_state == {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}
    assert observation.detections == [{"name": "red_object"}]


def test_execute_sends_the_action_payload_and_returns_the_bridge_result():
    connection = _FakeConnection([{"type": "action_result", "success": True, "message": "reached"}])

    result = RemoteRobot(connection).execute(_action())

    assert connection.requests == [
        {"type": "action", "action_id": "a1", "skill_name": "move", "parameters": {"x": 0.1, "y": 0.0, "z": 0.0}}
    ]
    assert result.success is True
    assert result.message == "reached"
    assert result.action_id == "a1"


def test_execute_reports_a_lost_connection_as_a_failed_result_not_an_exception():
    # Runtime treats a failed ActionResult as a task failure; an exception
    # here would instead kill the task's thread with no result recorded.
    result = RemoteRobot(_FakeConnection(error="robot disconnected")).execute(_action())

    assert result.success is False
    assert "disconnected" in result.message


def test_get_observation_reports_a_lost_connection_in_raw_error_not_an_exception():
    observation = RemoteRobot(_FakeConnection(error="robot disconnected")).get_observation()

    assert observation.robot_state == {}
    assert "disconnected" in observation.raw["error"]


def test_a_reply_without_a_success_field_counts_as_failure():
    result = RemoteRobot(_FakeConnection([{"type": "action_result"}])).execute(_action())

    assert result.success is False


def test_a_remote_bridge_declaring_a_dead_sensor_has_that_declaration_forwarded():
    # A tenant's robot is the least trustworthy sensor source in the
    # system. The channels block used to be dropped on the way in, so a
    # remote robot with a failed camera arrived looking like one reporting
    # an empty, healthy workspace.
    connection = _FakeConnection(
        [
            {
                "type": "observation",
                "robot_state": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "detections": [],
                "channels": {"detections": {"available": False, "detail": "depth camera offline"}},
            }
        ]
    )

    observation = RemoteRobot(connection).get_observation()

    assert observation.channels["detections"].available is False
    assert observation.channels["detections"].detail == "depth camera offline"


def test_a_dropped_connection_yields_an_observation_both_channels_declare_unknown():
    observation = RemoteRobot(_FakeConnection(error="socket closed")).get_observation()

    assert observation.channels["robot_state"].available is False
    assert observation.channels["detections"].available is False
    assert "socket closed" in observation.channels["detections"].detail


def test_the_safety_kernel_denies_a_move_for_a_robot_whose_connection_dropped():
    """The hosted-service version of the fail-open bug: the robot is
    unreachable, so the service knows nothing about its surroundings, and
    a move must be refused rather than approved against an empty world."""
    from par.core.capability import Capability
    from par.safety.environment import EnvironmentProfile, Workspace
    from par.safety.kernel import SafetyKernel
    from par.safety.policy import PolicyOutcome

    observation = RemoteRobot(_FakeConnection(error="socket closed")).get_observation()
    kernel = SafetyKernel(
        EnvironmentProfile(
            name="simulation",
            workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
            max_velocity=10.0,
            action_timeout_seconds=1.0,
        )
    )

    decision = kernel.check(
        _action(), observation, Capability(name="move", description="move", env_profiles=["simulation"])
    )

    assert decision.outcome == PolicyOutcome.DENY
    assert "socket closed" in decision.reason
