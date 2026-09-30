"""
SimulatedArmBridge - a drop-in alternative to CollectiveOSBridge for
ComputerAugmentedRobot's `bridge` parameter (par.robots.computer_bridge).
Instead of delegating use_computer to CollectiveOS's real Navigation Agent
(which controls the real host screen via pyautogui), this maps the task's
text to a short, deterministic sequence of physical button presses on
computer_arm's kiosk panel via ComputerArmBridge - the robot's own
simulated body genuinely presses buttons that drive a real (if small,
scripted) simulated computer, entirely inside Webots.

Selected via PAR_COMPUTER_USE_MODE=simulated_arm (see
par.robots.computer_bridge._default_bridge); the real-CollectiveOS path
stays the default.

v1 scope, deliberately: keyword matching against a small fixed vocabulary,
not real vision/AI decision-making - matches the actual use_computer task
shapes this project has used so far (simple check/confirm/clear-style
requests), not general typing. Swapping in real per-screen-image reasoning
here would be a separate, much larger future integration (redirecting a
vision model's input to computer_arm's Display texture and its output to
button presses instead of real OS automation).

Never raises: matches CollectiveOSBridge's `run_task(action, task, timeout)
-> ActionResult` contract exactly (confirmed by
tests/test_computer_bridge.py's _FakeBridge shape), so it's a true drop-in.
"""
from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.integrations.computer_arm import ComputerArmBridge

# Single-press sequences only for v1 - computer_arm_bridge.py's own
# _MAX_PRESS_SECONDS (15s) per press must stay comfortably under
# use_computer's DEFAULT_TIMEOUT_SECONDS (180s, enforced by
# Runtime._execute_with_timeout independent of what this bridge does
# internally) - a single press plus one get_observation call is nowhere
# near that budget; multi-press sequences would need to budget deliberately
# against it too (see the plan's timeout-composition note).
_TASK_KEYWORDS: dict[str, str] = {
    "check": "check",
    "alert": "check",
    "status": "check",
    "confirm": "confirm",
    "clear": "clear",
    "reset": "clear",
}


class SimulatedArmBridge:
    def __init__(self, arm_bridge: ComputerArmBridge | None = None) -> None:
        self._arm = arm_bridge or ComputerArmBridge()

    def run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
        button = self._resolve_button(task)
        if button is None:
            return self._result(action, False, f"no button mapped for task: {task!r}")

        press_timeout = min(timeout, 20.0)
        reply = self._arm.press_button(button, timeout=press_timeout)
        if not reply.get("success"):
            return self._result(action, False, f"press '{button}' failed: {reply.get('message', '')}")

        obs = self._arm.get_observation_payload(timeout=min(timeout, 10.0))
        return self._result(action, True, f"pressed '{button}'; kiosk now shows: {obs.get('screen_text', '')}")

    def _resolve_button(self, task: str) -> str | None:
        lowered = task.lower()
        for keyword, button in _TASK_KEYWORDS.items():
            if keyword in lowered:
                return button
        return None

    def _result(self, action: Action, success: bool, message: str) -> ActionResult:
        return ActionResult(
            action_id=action.action_id, success=success, message=message, completed_at=datetime.now(timezone.utc)
        )
