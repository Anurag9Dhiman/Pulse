"""
CollectiveOS bridge — lets PAR delegate a computer-shaped step of a physical
task to CollectiveOS's Navigation Agent and get a text result back.

Talks to CollectiveOS's existing `WEBSOCKET /robot/ws` endpoint
(`CollectiveOS/src/robot_stream.py`), which already runs the nav agent
synchronously with demonstration recording on. One connection per call: PAR's
Action model is stateless per step, so this doesn't rely on that endpoint's
per-connection context accumulation.

Configuration (env vars):
    COLLECTIVEOS_WS_URL   e.g. ws://localhost:8000/robot/ws (required)
    COLLECTIVEOS_API_TOKEN  matches CollectiveOS's API_TOKEN (required)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from par.core.action import Action, ActionResult


class CollectiveOSError(Exception):
    pass


class CollectiveOSBridge:
    def __init__(self, ws_url: str | None = None, api_token: str | None = None) -> None:
        self.ws_url = ws_url or os.environ.get("COLLECTIVEOS_WS_URL", "")
        self.api_token = api_token or os.environ.get("COLLECTIVEOS_API_TOKEN", "")

    def run_task(self, action: Action, task: str, timeout: float) -> ActionResult:
        """Sends `task` to CollectiveOS and blocks for its reply.

        Never raises: connection/protocol/timeout failures come back as a
        failed ActionResult so PAR's loop can record the failure and
        re-plan instead of crashing.
        """
        if not self.ws_url:
            return self._result(action, False, "COLLECTIVEOS_WS_URL is not configured")

        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            return self._result(
                action, False, f"websockets package not installed (pip install 'par[computer]'): {exc}"
            )

        url = self.ws_url
        if self.api_token:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}token={self.api_token}"

        try:
            with connect(url, open_timeout=timeout, close_timeout=5) as websocket:
                websocket.recv(timeout=timeout)  # initial {"type": "ack", ...}
                websocket.send(json.dumps({"type": "task", "text": task}))
                raw = websocket.recv(timeout=timeout)
        except TimeoutError:
            return self._result(action, False, f"CollectiveOS did not reply within {timeout}s")
        except Exception as exc:
            return self._result(action, False, f"CollectiveOS connection failed: {exc}")

        try:
            reply = json.loads(raw)
        except (TypeError, ValueError) as exc:
            return self._result(action, False, f"CollectiveOS sent an invalid reply: {exc}")

        if reply.get("type") == "error":
            return self._result(action, False, reply.get("message", "CollectiveOS returned an error"))
        if reply.get("type") != "reply":
            return self._result(action, False, f"unexpected message type from CollectiveOS: {reply.get('type')!r}")

        status = reply.get("status", "")
        text = reply.get("text", "")
        return self._result(action, status == "done", text or f"[{status}]")

    def _result(self, action: Action, success: bool, message: str) -> ActionResult:
        return ActionResult(
            action_id=action.action_id,
            success=success,
            message=message,
            completed_at=datetime.now(timezone.utc),
        )
