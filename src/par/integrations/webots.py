"""
Webots bridge — lets PAR drive a real physically-simulated robot (an e-puck,
by default) instead of MockRobot, so `move`/Safety-Kernel behavior can be
watched live instead of just trusted from ActionResult.success.

Talks to a small WebSocket server run by the Webots-side extern controller
(Reach/webots/controllers/par_bridge/par_bridge.py), not to Webots itself
directly. The wire protocol is plain JSON:

    {"type": "get_observation"}
        -> {"robot_state": {...}, "detections": [...]}

    {"type": "action", "action_id": ..., "skill_name": ..., "parameters": {...}}
        -> {"success": bool, "message": str}

These payload shapes match par.robots.ros2_mapping's
payload_to_observation/action_to_payload exactly, so this bridge carries the
same schema real ROS 2 would eventually carry - only the transport differs.

Configuration (env var):
    WEBOTS_BRIDGE_URL   e.g. ws://localhost:6001 (default)
"""
from __future__ import annotations

import json
import os

_DEFAULT_URL = "ws://localhost:6001"


class WebotsBridge:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("WEBOTS_BRIDGE_URL", _DEFAULT_URL)

    def get_observation_payload(self, timeout: float = 10.0) -> dict:
        """Never raises: a failure comes back as an empty state with the
        reason stashed in `raw.error`, since Observation has no success flag."""
        reply, error = self._request({"type": "get_observation"}, timeout)
        if error:
            return {"robot_state": {}, "detections": [], "raw": {"error": error}}
        return {"robot_state": reply.get("robot_state", {}), "detections": reply.get("detections", [])}

    def send_action(self, payload: dict, timeout: float = 30.0) -> dict:
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
            return {}, f"Webots bridge did not reply within {timeout}s ({self.url})"
        except Exception as exc:
            return {}, f"Webots bridge connection failed ({self.url}): {exc}"

        try:
            return json.loads(raw), ""
        except (TypeError, ValueError) as exc:
            return {}, f"Webots bridge sent an invalid reply: {exc}"
