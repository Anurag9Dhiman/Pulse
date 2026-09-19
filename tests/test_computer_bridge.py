from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.capability import RiskLevel
from par.core.observation import Observation
from par.robots.computer_bridge import ComputerAugmentedRobot
from par.robots.mock import MockRobot
from par.skills.computer_use import DEFAULT_TIMEOUT_SECONDS, computer_use_skill


def test_computer_use_skill_capability_shape():
    skill = computer_use_skill()
    cap = skill.capability

    assert cap.name == "use_computer"
    assert cap.risk == RiskLevel.HIGH
    assert cap.supports_rollback is False
    assert set(cap.env_profiles) == {"simulation", "real_robot"}
    assert cap.execution_timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert cap.input_schema == {"task": "str"}


class _FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
        self.calls.append((task, timeout))
        return ActionResult(
            action_id=action.action_id, success=True, message="ok", completed_at=datetime.now(timezone.utc)
        )


def _action(skill_name: str, parameters: dict) -> Action:
    return Action(action_id="a1", skill_name=skill_name, parameters=parameters, created_at=datetime.now(timezone.utc))


def test_computer_augmented_robot_routes_use_computer_to_bridge():
    bridge = _FakeBridge()
    robot = ComputerAugmentedRobot(MockRobot(), bridge=bridge)

    result = robot.execute(_action("use_computer", {"task": "look up today's date"}))

    assert result.success is True
    assert bridge.calls == [("look up today's date", DEFAULT_TIMEOUT_SECONDS)]


def test_computer_augmented_robot_delegates_other_skills_to_wrapped_robot():
    bridge = _FakeBridge()
    mock = MockRobot()
    robot = ComputerAugmentedRobot(mock, bridge=bridge)

    result = robot.execute(_action("pick", {"object": "red_object"}))

    assert result.success is True
    assert bridge.calls == []
    assert mock._holding == "red_object"


def test_computer_augmented_robot_delegates_get_observation():
    mock = MockRobot()
    robot = ComputerAugmentedRobot(mock, bridge=_FakeBridge())

    observation = robot.get_observation()

    assert isinstance(observation, Observation)
    assert observation.source == "mock"
