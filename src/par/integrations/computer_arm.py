"""
computer_arm bridge — lets PAR drive the physically-real gantry that presses
buttons on computer_arm's kiosk panel (Reach/webots/controllers/
computer_arm_bridge/computer_arm_bridge.py), instead of software-automating
a real screen via CollectiveOS. See
par.integrations.simulated_arm_computer_use.SimulatedArmBridge for the
run_task(...)-shaped bridge that actually uses this for use_computer.

Talks to computer_arm_bridge.py's WebSocket server, not to Webots itself
directly - same relationship WebotsBridge has to par_bridge.py. Wire
protocol (plain JSON):

    {"type": "get_observation"}
        -> {"screen_state": str, "gantry_position": {"x": float, "z": float}}

    {"type": "action", "action_id": ..., "skill_name": "press_button",
     "parameters": {"button": "check" | "confirm" | "clear"}}
        -> {"success": bool, "message": str}

Configuration (env var):
    COMPUTER_ARM_BRIDGE_URL   e.g. ws://localhost:6002 (default)
"""
from __future__ import annotations

import json
import os
from uuid import uuid4

_DEFAULT_URL = "ws://localhost:6002"


class ComputerArmBridge:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("COMPUTER_ARM_BRIDGE_URL", _DEFAULT_URL)

    def get_observation_payload(self, timeout: float = 10.0) -> dict:
        """Never raises: a failure comes back as an empty state with the
        reason stashed in `raw.error`, matching WebotsBridge's convention."""
        reply, error = self._request({"type": "get_observation"}, timeout)
        if error:
            return {"screen_state": "", "screen_text": "", "gantry_position": {}, "raw": {"error": error}}
        return {
            "screen_state": reply.get("screen_state", ""),
            "screen_text": reply.get("screen_text", ""),
            "gantry_position": reply.get("gantry_position", {}),
        }

    def press_button(self, button: str, timeout: float = 20.0) -> dict:
        payload = {"action_id": str(uuid4()), "skill_name": "press_button", "parameters": {"button": button}}
        reply, error = self._request({"type": "action", **payload}, timeout)
        if error:
            return {"success": False, "message": error}
        return {"success": bool(reply.get("success", False)), "message": reply.get("message", "")}

    def _request(self, message: dict, timeout: float) -> tuple[dict, str]:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            return {}, f"websockets package not installed (pip install 'par[webots]'): {exc}"

        try:
            with connect(self.url, open_timeout=timeout, close_timeout=5) as websocket:
                websocket.send(json.dumps(message))
                raw = websocket.recv(timeout=timeout)
        except TimeoutError:
            return {}, f"computer_arm bridge did not reply within {timeout}s ({self.url})"
        except Exception as exc:
            return {}, f"computer_arm bridge connection failed ({self.url}): {exc}"

        try:
            return json.loads(raw), ""
        except (TypeError, ValueError) as exc:
            return {}, f"computer_arm bridge sent an invalid reply: {exc}"
