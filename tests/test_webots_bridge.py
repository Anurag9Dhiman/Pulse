from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action
from par.core.observation import Observation
from par.robots.webots_bridge import WebotsRobot


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
        observation_payload={"robot_state": {"position": {"x": 0.5, "y": 0.0, "z": 0.0}}, "detections": [{"name": "red_object"}]},
        action_reply={},
    )
    robot = WebotsRobot(bridge=bridge)

    observation = robot.get_observation()

    assert isinstance(observation, Observation)
    assert observation.source == "ros2"  # same schema real ROS 2 would carry
    assert observation.robot_state == {"position": {"x": 0.5, "y": 0.0, "z": 0.0}}
    assert observation.detections == [{"name": "red_object"}]


def test_get_observation_surfaces_bridge_error_via_raw():
    bridge = _FakeBridge(observation_payload={"raw": {"error": "Webots bridge unreachable"}}, action_reply={})
    robot = WebotsRobot(bridge=bridge)

    observation = robot.get_observation()

    assert observation.robot_state == {}
    assert observation.raw == {"error": "Webots bridge unreachable"}


def test_execute_sends_action_payload_and_builds_result():
    bridge = _FakeBridge(
        observation_payload={},
        action_reply={"success": True, "message": "reached target"},
    )
    robot = WebotsRobot(bridge=bridge)

    result = robot.execute(_action("move", {"x": 0.5, "y": 0.2, "z": 0.0}))

    assert bridge.sent_actions == [{"action_id": "a1", "skill_name": "move", "parameters": {"x": 0.5, "y": 0.2, "z": 0.0}}]
    assert result.success is True
    assert result.message == "reached target"


def test_execute_reports_failure_from_bridge():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": False, "message": "Webots bridge connection failed"})
    robot = WebotsRobot(bridge=bridge)

    result = robot.execute(_action("stop", {}))

    assert result.success is False
    assert result.message == "Webots bridge connection failed"


def test_begin_computer_use_sends_dock_action():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": True, "message": "docked near 'laptop'"})
    robot = WebotsRobot(bridge=bridge)

    robot.begin_computer_use()

    assert len(bridge.sent_actions) == 1
    sent = bridge.sent_actions[0]
    assert sent["skill_name"] == "dock_at_computer"
    assert sent["parameters"] == {}
    assert "action_id" in sent  # a fresh id - this call has no originating PAR Action


def test_end_computer_use_sends_undock_action():
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": True, "message": "undocked from computer"})
    robot = WebotsRobot(bridge=bridge)

    robot.end_computer_use()

    assert len(bridge.sent_actions) == 1
    assert bridge.sent_actions[0]["skill_name"] == "undock_from_computer"


def test_dock_hooks_never_raise_even_if_bridge_reports_failure():
    # begin_/end_computer_use's result is intentionally not checked by callers
    # (see ComputerAugmentedRobot) - a docking failure must never surface as
    # an exception that could block the actual use_computer delegation.
    bridge = _FakeBridge(observation_payload={}, action_reply={"success": False, "message": "'laptop' prop not found"})
    robot = WebotsRobot(bridge=bridge)

    robot.begin_computer_use()
    robot.end_computer_use()


def test_a_bridge_failure_marks_both_channels_unavailable_instead_of_reporting_an_empty_world():
    # WebotsBridge returns an empty robot_state/detections on a transport
    # failure and always has. What is new is that the emptiness is
    # labelled: unlabelled, it was indistinguishable from a healthy bridge
    # reporting a clear workspace.
    bridge = _FakeBridge(
        observation_payload={
            "robot_state": {},
            "detections": [],
            "channels": {
                "robot_state": {"available": False, "detail": "webots bridge: timed out"},
                "detections": {"available": False, "detail": "webots bridge: timed out"},
            },
            "raw": {"error": "timed out"},
        },
        action_reply={},
    )

    observation = WebotsRobot(bridge=bridge).get_observation()

    assert observation.channels["detections"].available is False
    assert "timed out" in observation.channels["detections"].detail


def test_a_bridge_that_only_reports_raw_error_still_has_its_channels_inferred_as_unavailable():
    # Belt and braces for any bridge predating the channels block: the
    # presence of raw.error is itself proof that nothing was observed.
    bridge = _FakeBridge(observation_payload={"raw": {"error": "connection refused"}}, action_reply={})

    observation = WebotsRobot(bridge=bridge).get_observation()

    assert observation.channels["robot_state"].available is False
    assert observation.channels["detections"].available is False


def test_the_safety_kernel_denies_a_move_built_on_a_failed_bridge_observation():
    """End to end, the case that used to be approved: the bridge is down,
    so PAR knows nothing about the world, and the kernel must refuse to
    certify motion through it."""
    from par.core.capability import Capability
    from par.safety.environment import EnvironmentProfile, Workspace
    from par.safety.kernel import SafetyKernel
    from par.safety.policy import PolicyOutcome

    bridge = _FakeBridge(observation_payload={"raw": {"error": "connection refused"}}, action_reply={})
    observation = WebotsRobot(bridge=bridge).get_observation()
    kernel = SafetyKernel(
        EnvironmentProfile(
            name="test",
            workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
            max_velocity=10.0,
            action_timeout_seconds=1.0,
        )
    )

    decision = kernel.check(
        _action("move", {"x": 0.2, "y": 0.0, "z": 0.0}),
        observation,
        Capability(name="move", description="move", env_profiles=["test"]),
    )

    assert decision.outcome == PolicyOutcome.DENY
    assert "connection refused" in decision.reason
