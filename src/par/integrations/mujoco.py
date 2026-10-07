"""
MuJoCo bridge — lets PAR drive a physically-simulated Panda arm (Reach/
mujoco/bridge/mujoco_bridge.py) instead of MockRobot, so `move`/Safety-
Kernel behavior can be watched live.

Mirrors par.integrations.webots.WebotsBridge line-for-line: thin WebSocket
client transport, same wire-protocol shape (which matches
par.robots.ros2_mapping's payload_to_observation/action_to_payload exactly,
so this bridge carries the same schema real ROS 2 would eventually carry,
only the transport differs). The ONLY things different from WebotsBridge
are the env-var name and the default port.

    {"type": "get_observation"}
        -> {"robot_state": {...}, "detections": [...]}

    {"type": "action", "action_id": ..., "skill_name": ..., "parameters": {...}}
        -> {"success": bool, "message": str}

Configuration (env var):
    MUJOCO_BRIDGE_URL   e.g. ws://localhost:6003 (default)
"""
from __future__ import annotations

import json
import os

_DEFAULT_URL = "ws://localhost:6003"


class MuJoCoBridge:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("MUJOCO_BRIDGE_URL", _DEFAULT_URL)

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
            return {}, f"websockets package not installed (pip install 'par[mujoco]'): {exc}"

        try:
            with connect(self.url, open_timeout=timeout, close_timeout=5) as websocket:
                websocket.send(json.dumps(message))
                raw = websocket.recv(timeout=timeout)
        except TimeoutError:
            return {}, f"MuJoCo bridge did not reply within {timeout}s ({self.url})"
        except Exception as exc:
            return {}, f"MuJoCo bridge connection failed ({self.url}): {exc}"

        try:
            return json.loads(raw), ""
        except (TypeError, ValueError) as exc:
            return {}, f"MuJoCo bridge sent an invalid reply: {exc}"
