from __future__ import annotations

import json
from datetime import datetime, timezone

from par.core.action import Action
from par.integrations.vision_guided_arm import VisionGuidedArmBridge


def _action(task: str) -> Action:
    return Action(
        action_id="a1", skill_name="use_computer", parameters={"task": task}, created_at=datetime.now(timezone.utc)
    )


class _FakeArmBridge:
    def __init__(
        self,
        snapshot_path: str = "/tmp/fake_snapshot.jpg",
        press_success: bool = True,
        press_message: str = "ok",
        screen_text: str = "READY",
    ) -> None:
        self.snapshot_path = snapshot_path
        self.press_success = press_success
        self.press_message = press_message
        self.screen_text = screen_text
        self.pressed: list[str] = []

    def capture_camera_snapshot(self, timeout: float = 10.0) -> dict:
        return {"path": self.snapshot_path}

    def press_button(self, button: str, timeout: float = 20.0) -> dict:
        self.pressed.append(button)
        return {"success": self.press_success, "message": self.press_message}

    def get_observation_payload(self, timeout: float = 10.0) -> dict:
        return {"screen_state": "idle", "screen_text": self.screen_text, "gantry_position": {"x": 0.0, "z": 0.0}}


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, response_json: dict, error: Exception | None = None) -> None:
        self._response_json = response_json
        self._error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeResponse(json.dumps(self._response_json))


class _FakeClient:
    def __init__(self, response_json: dict, error: Exception | None = None) -> None:
        self.models = _FakeModels(response_json, error)


def _write_fake_jpeg(tmp_path) -> str:
    path = tmp_path / "snapshot.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return str(path)


def test_correct_button_chosen_presses_it(tmp_path):
    arm = _FakeArmBridge(snapshot_path=_write_fake_jpeg(tmp_path))
    client = _FakeClient({"button": "check", "reason": "panel is idle"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is True
    assert arm.pressed == ["check"]
    assert "panel is idle" in result.message
    assert "READY" in result.message


def test_none_response_is_clean_failure_no_press_attempted(tmp_path):
    arm = _FakeArmBridge(snapshot_path=_write_fake_jpeg(tmp_path))
    client = _FakeClient({"button": "none", "reason": "already confirmed, nothing to do"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is False
    assert "already confirmed" in result.message
    assert arm.pressed == []


def test_gemini_call_raising_is_caught_never_propagates(tmp_path):
    arm = _FakeArmBridge(snapshot_path=_write_fake_jpeg(tmp_path))
    client = _FakeClient({}, error=RuntimeError("quota exceeded"))
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is False
    assert "quota exceeded" in result.message
    assert arm.pressed == []


def test_missing_snapshot_is_clean_failure():
    class _NoSnapshotArm(_FakeArmBridge):
        def capture_camera_snapshot(self, timeout: float = 10.0) -> dict:
            return {"path": "", "raw": {"error": "kiosk_camera device not found"}}

    arm = _NoSnapshotArm()
    client = _FakeClient({"button": "check", "reason": "n/a"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is False
    assert "kiosk_camera device not found" in result.message
    assert arm.pressed == []


def test_unrecognized_button_is_clean_failure(tmp_path):
    arm = _FakeArmBridge(snapshot_path=_write_fake_jpeg(tmp_path))
    client = _FakeClient({"button": "launch_missiles", "reason": "n/a"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is False
    assert "unrecognized button" in result.message
    assert arm.pressed == []


def test_press_failure_propagates_as_task_failure(tmp_path):
    arm = _FakeArmBridge(
        snapshot_path=_write_fake_jpeg(tmp_path),
        press_success=False,
        press_message="pressed toward 'check' but no contact was registered",
    )
    client = _FakeClient({"button": "check", "reason": "panel is idle"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    result = bridge.run_task(_action("check status"), "check status", timeout=180.0)

    assert result.success is False
    assert "no contact was registered" in result.message


def test_run_task_never_raises_action_id_is_preserved(tmp_path):
    arm = _FakeArmBridge(snapshot_path=_write_fake_jpeg(tmp_path))
    client = _FakeClient({"button": "check", "reason": "panel is idle"})
    bridge = VisionGuidedArmBridge(arm_bridge=arm, client=client)

    action = _action("check status")
    result = bridge.run_task(action, "check status", timeout=180.0)

    assert result.action_id == action.action_id
