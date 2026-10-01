from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.capability import RiskLevel
from par.core.observation import Observation
from par.integrations.collectiveos import CollectiveOSBridge
from par.integrations.simulated_arm_computer_use import SimulatedArmBridge
from par.integrations.vision_guided_arm import VisionGuidedArmBridge
from par.robots.computer_bridge import ComputerAugmentedRobot, _default_bridge
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


class _RobotWithDockHooks(MockRobot):
    """Stands in for WebotsRobot, which implements these hooks to drive to a
    laptop prop and light an LED - see robots/webots_bridge.py."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def begin_computer_use(self) -> None:
        self.calls.append("begin")

    def end_computer_use(self) -> None:
        self.calls.append("end")


def test_computer_augmented_robot_calls_dock_hooks_around_delegation_when_present():
    robot_impl = _RobotWithDockHooks()
    robot = ComputerAugmentedRobot(robot_impl, bridge=_FakeBridge())

    result = robot.execute(_action("use_computer", {"task": "look up today's date"}))

    assert result.success is True
    assert robot_impl.calls == ["begin", "end"]


def test_computer_augmented_robot_calls_end_hook_even_when_delegation_fails():
    class _FailingBridge:
        def run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
            return ActionResult(
                action_id=action.action_id, success=False, message="nope", completed_at=datetime.now(timezone.utc)
            )

    robot_impl = _RobotWithDockHooks()
    robot = ComputerAugmentedRobot(robot_impl, bridge=_FailingBridge())

    result = robot.execute(_action("use_computer", {"task": "look up today's date"}))

    assert result.success is False
    assert robot_impl.calls == ["begin", "end"]


def test_computer_augmented_robot_skips_dock_hooks_when_wrapped_robot_lacks_them():
    # MockRobot has no begin_/end_computer_use - must not raise AttributeError.
    robot = ComputerAugmentedRobot(MockRobot(), bridge=_FakeBridge())

    result = robot.execute(_action("use_computer", {"task": "look up today's date"}))

    assert result.success is True


def test_use_computer_is_admitted_under_both_profiles_and_escalates_when_approval_required():
    from par.safety.environment import load_profile
    from par.safety.kernel import SafetyKernel
    from par.safety.policy import PolicyOutcome

    capability = computer_use_skill().capability
    observation = MockRobot().get_observation()
    action = _action("use_computer", {"task": "t"})

    for profile in (load_profile("simulation"), load_profile("real_robot")):
        kernel = SafetyKernel(profile.model_copy(update={"approval_required": True}))
        assert kernel.admit(capability).outcome == PolicyOutcome.ALLOW
        assert kernel.check(action, observation, capability).outcome == PolicyOutcome.ESCALATE

    unattended = SafetyKernel(load_profile("simulation"))
    assert unattended.check(action, observation, capability).outcome == PolicyOutcome.ALLOW


def test_default_bridge_is_collectiveos_when_mode_unset(monkeypatch):
    monkeypatch.delenv("PAR_COMPUTER_USE_MODE", raising=False)
    assert isinstance(_default_bridge(), CollectiveOSBridge)


def test_default_bridge_is_simulated_arm_when_mode_set(monkeypatch):
    monkeypatch.setenv("PAR_COMPUTER_USE_MODE", "simulated_arm")
    assert isinstance(_default_bridge(), SimulatedArmBridge)


def test_default_bridge_is_vision_guided_arm_when_mode_set(monkeypatch):
    monkeypatch.setenv("PAR_COMPUTER_USE_MODE", "vision_guided_arm")
    assert isinstance(_default_bridge(), VisionGuidedArmBridge)


def test_computer_augmented_robot_uses_env_selected_default_when_no_bridge_given(monkeypatch):
    monkeypatch.setenv("PAR_COMPUTER_USE_MODE", "simulated_arm")
    robot = ComputerAugmentedRobot(MockRobot())
    assert isinstance(robot._bridge, SimulatedArmBridge)
