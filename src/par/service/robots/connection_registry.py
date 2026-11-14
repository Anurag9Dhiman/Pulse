from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import uuid
from typing import Any, Protocol

_REPLY_TYPES = {"observation", "action_result"}


class RobotConnectionError(Exception):
    pass


class RobotConnection(Protocol):
    def send_request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class RemoteRobotConnection:
    """One live inbound WebSocket from a robot's own bridge process.

    send_request() is synchronous and is called from a task's background
    thread (where Runtime runs); the socket itself belongs to the service's
    asyncio loop, so the send is scheduled onto that loop and the caller
    blocks until the receive loop hands back the reply with the matching
    request_id.
    """

    def __init__(self, websocket: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._websocket = websocket
        self._loop = loop
        self._lock = threading.Lock()
        self._pending: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
        self._closed = False

    def send_request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        reply: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        with self._lock:
            if self._closed:
                raise RobotConnectionError("robot disconnected")
            self._pending[request_id] = reply
        try:
            message = json.dumps({**payload, "request_id": request_id})
            sent = asyncio.run_coroutine_threadsafe(self._websocket.send_text(message), self._loop)
            sent.result(timeout)
            return reply.result(timeout)
        except concurrent.futures.TimeoutError as exc:
            raise RobotConnectionError(f"robot did not reply within {timeout}s") from exc
        except RobotConnectionError:
            raise
        except Exception as exc:
            raise RobotConnectionError(f"robot connection failed: {exc}") from exc
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def on_message(self, message: dict[str, Any]) -> None:
        if message.get("type") not in _REPLY_TYPES:
            return
        with self._lock:
            reply = self._pending.get(str(message.get("request_id")))
        if reply is not None and not reply.done():
            reply.set_result(message)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._pending.values())
        for reply in pending:
            if not reply.done():
                reply.set_exception(RobotConnectionError("robot disconnected"))


class RobotConnectionRegistry:
    """(tenant_id, robot_id) -> the live connection, if any. In-memory, so
    only the process holding the socket can serve that robot's tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: dict[tuple[str, str], RobotConnection] = {}

    def register(self, tenant_id: str, robot_id: str, connection: RobotConnection) -> None:
        with self._lock:
            self._connections[(tenant_id, robot_id)] = connection

    def unregister(self, tenant_id: str, robot_id: str, connection: RobotConnection) -> None:
        with self._lock:
            # A reconnect may already have replaced this entry; don't evict the newer socket.
            if self._connections.get((tenant_id, robot_id)) is connection:
                del self._connections[(tenant_id, robot_id)]

    def get(self, tenant_id: str, robot_id: str) -> RobotConnection | None:
        with self._lock:
            return self._connections.get((tenant_id, robot_id))
