"""
VisionGuidedArmBridge - a third ComputerAugmentedRobot bridge mode, alongside
CollectiveOSBridge (real host screen) and SimulatedArmBridge (keyword
matching against the task text alone). Instead of mapping the task's text to
a button by keyword, this captures a real image of computer_arm's kiosk
panel (ComputerArmBridge.capture_camera_snapshot) and asks Gemini to look at
it before deciding which button to press - the same kind of vision-grounded
decision CollectiveOS's NavAgent already makes against the real host screen
(CollectiveOS/src/agents/nav_agent.py), scoped down to a 3-way button choice
instead of general desktop control.

Selected via PAR_COMPUTER_USE_MODE=vision_guided_arm (see
par.robots.computer_bridge._default_bridge). Needs GEMINI_API_KEY - unlike
SimulatedArmBridge, every call here spends real Gemini quota.

Uses gemini-3.1-flash-lite (par.core.gemini_planner.DEFAULT_MODEL), not
CollectiveOS's gemini-3.6-flash: that model has a hard 20-requests-per-day
cap (see Reach/experiments/README.md), far too scarce for a per-use_computer
-call image classification. Confirmed by a real call (2026-10-01) that
flash-lite accepts image input and correctly reasons from a real captured
kiosk photo - this was an open question, not an assumption; no fallback
model is needed.

Call shape mirrors CollectiveOS's nav_agent.py exactly (same google-genai
client, same Content/Part.from_bytes/GenerateContentConfig/response_schema
pattern) - not a new invention, just a 3-way classification instead of
general desktop-action decoding.

Never raises: matches the CollectiveOSBridge/SimulatedArmBridge
`run_task(action, task, timeout) -> ActionResult` contract exactly
(confirmed by tests/test_computer_bridge.py's _FakeBridge shape).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from par.core.action import Action, ActionResult
from par.integrations.computer_arm import ComputerArmBridge

_MODEL = "gemini-3.1-flash-lite"

_SYSTEM_INSTRUCTION = (
    "You are looking at a photo of a small kiosk panel with three physical "
    "buttons stacked vertically: CHECK (top, red), CONFIRM (middle, blue), "
    "CLEAR (bottom, green). Pressing CHECK shows a status/alert message. "
    "Pressing CONFIRM only has an effect once a status is already shown. "
    "Pressing CLEAR returns the panel to its idle/ready state. Given the "
    "task and the photo, decide which single button to press next, or "
    "'none' if no press is appropriate right now."
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "button": {"type": "string", "enum": ["check", "confirm", "clear", "none"]},
        "reason": {"type": "string", "description": "One-line reason for this choice."},
    },
    "required": ["button", "reason"],
}

_VALID_BUTTONS = ("check", "confirm", "clear")


class VisionGuidedArmBridge:
    def __init__(self, arm_bridge: ComputerArmBridge | None = None, client: Any = None, model: str = _MODEL) -> None:
        self._arm = arm_bridge or ComputerArmBridge()
        self._client = client
        self._model = model

    def run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
        try:
            return self._run_task(action, task, timeout)
        except Exception as exc:
            return self._result(action, False, f"vision-guided decision failed: {exc}")

    def _run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
        snapshot = self._arm.capture_camera_snapshot(timeout=min(timeout, 10.0))
        path = snapshot.get("path", "")
        if not path:
            error = snapshot.get("raw", {}).get("error", "unknown error")
            return self._result(action, False, f"could not capture kiosk camera image: {error}")

        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
        except OSError as exc:
            return self._result(action, False, f"could not read captured image at {path}: {exc}")

        button, reason = self._decide_button(task, image_bytes)
        if button == "none":
            return self._result(action, False, f"vision model chose no action: {reason}")
        if button not in _VALID_BUTTONS:
            return self._result(action, False, f"vision model returned unrecognized button {button!r}")

        press_timeout = min(timeout, 20.0)
        reply = self._arm.press_button(button, timeout=press_timeout)
        if not reply.get("success"):
            return self._result(action, False, f"press '{button}' failed: {reply.get('message', '')}")

        obs = self._arm.get_observation_payload(timeout=min(timeout, 10.0))
        return self._result(
            action, True, f"pressed '{button}' ({reason}); kiosk now shows: {obs.get('screen_text', '')}"
        )

    def _decide_button(self, task: str, image_bytes: bytes) -> tuple[str, str]:
        from google.genai import types as gtypes

        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        prompt = f"Task: {task}\n\nWhich button should be pressed next?"
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        gtypes.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        return str(data.get("button", "none")), str(data.get("reason", ""))

    def _result(self, action: Action, success: bool, message: str) -> ActionResult:
        return ActionResult(
            action_id=action.action_id, success=success, message=message, completed_at=datetime.now(timezone.utc)
        )
