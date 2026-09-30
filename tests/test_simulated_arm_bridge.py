from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action
from par.integrations.simulated_arm_computer_use import SimulatedArmBridge


def _action(task: str) -> Action:
    return Action(
        action_id="a1", skill_name="use_computer", parameters={"task": task}, created_at=datetime.now(timezone.utc)
    )


class _FakeArmBridge:
    def __init__(self, press_success: bool = True, press_message: str = "ok", screen_text: str = "READY") -> None:
        self.press_success = press_success
        self.press_message = press_message
        self.screen_text = screen_text
        self.pressed: list[str] = []

    def press_button(self, button: str, timeout: float = 20.0) -> dict:
        self.pressed.append(button)
        return {"success": self.press_success, "message": self.press_message}

    def get_observation_payload(self, timeout: float = 10.0) -> dict:
        return {"screen_state": "idle", "screen_text": self.screen_text, "gantry_position": {"x": 0.0, "z": 0.0}}


def test_resolves_check_alert_status_keywords_to_check_button():
    for task in ["check whether any maintenance alerts are open", "any alerts?", "get status"]:
        arm = _FakeArmBridge()
        bridge = SimulatedArmBridge(arm_bridge=arm)
        result = bridge.run_task(_action(task), task, timeout=180.0)
        assert result.success is True
        assert arm.pressed == ["check"]


def test_resolves_confirm_and_clear_reset_keywords():
    arm = _FakeArmBridge()
    bridge = SimulatedArmBridge(arm_bridge=arm)
    bridge.run_task(_action("please confirm this"), "please confirm this", timeout=180.0)
    bridge.run_task(_action("please reset it"), "please reset it", timeout=180.0)
    assert arm.pressed == ["confirm", "clear"]


def test_unmapped_task_fails_explicitly_not_silently():
    arm = _FakeArmBridge()
    bridge = SimulatedArmBridge(arm_bridge=arm)
    result = bridge.run_task(_action("do something unrelated"), "do something unrelated", timeout=180.0)
    assert result.success is False
    assert "no button mapped" in result.message
    assert arm.pressed == []


def test_press_failure_propagates_as_task_failure():
    arm = _FakeArmBridge(press_success=False, press_message="pressed toward 'check' but no contact was registered")
    bridge = SimulatedArmBridge(arm_bridge=arm)
    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)
    assert result.success is False
    assert "no contact was registered" in result.message


def test_success_message_includes_resulting_screen_text():
    arm = _FakeArmBridge(screen_text="1 ALERT: LOW BATTERY")
    bridge = SimulatedArmBridge(arm_bridge=arm)
    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)
    assert result.success is True
    assert "1 ALERT: LOW BATTERY" in result.message


def test_run_task_never_raises_action_id_is_preserved():
    arm = _FakeArmBridge()
    bridge = SimulatedArmBridge(arm_bridge=arm)
    action = _action("check status")
    result = bridge.run_task(action, "check status", timeout=180.0)
    assert result.action_id == action.action_id
