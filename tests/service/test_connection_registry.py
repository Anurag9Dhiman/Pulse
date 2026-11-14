from __future__ import annotations

import asyncio
import json
import threading

import pytest

from par.service.robots.connection_registry import (
    RemoteRobotConnection,
    RobotConnectionError,
    RobotConnectionRegistry,
)


class _FakeWebSocket:
    """Records what the service sent; optionally answers like a robot bridge would."""

    def __init__(self, connection_ref: list, reply: dict | None) -> None:
        self.sent: list[dict] = []
        self._connection_ref = connection_ref
        self._reply = reply

    async def send_text(self, text: str) -> None:
        message = json.loads(text)
        self.sent.append(message)
        if self._reply is not None:
            self._connection_ref[0].on_message({**self._reply, "request_id": message["request_id"]})


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(timeout=2)
    event_loop.close()


def _connection(loop, reply: dict | None) -> tuple[RemoteRobotConnection, _FakeWebSocket]:
    ref: list = []
    websocket = _FakeWebSocket(ref, reply)
    connection = RemoteRobotConnection(websocket, loop)
    ref.append(connection)
    return connection, websocket


def test_send_request_returns_the_reply_carrying_the_same_request_id(loop):
    connection, websocket = _connection(loop, {"type": "observation", "robot_state": {"holding": None}})

    reply = connection.send_request({"type": "get_observation"}, timeout=2)

    assert reply["robot_state"] == {"holding": None}
    assert websocket.sent[0]["type"] == "get_observation"
    assert reply["request_id"] == websocket.sent[0]["request_id"]


def test_send_request_raises_when_the_robot_never_replies(loop):
    connection, _ = _connection(loop, reply=None)

    with pytest.raises(RobotConnectionError, match="did not reply"):
        connection.send_request({"type": "get_observation"}, timeout=0.2)


def test_a_reply_for_a_different_request_id_does_not_satisfy_the_caller(loop):
    # Without correlation, a late reply to an earlier (timed-out) request
    # would be mistaken for the answer to the current one.
    connection, _ = _connection(loop, reply=None)
    connection.on_message({"type": "observation", "request_id": "someone-else"})

    with pytest.raises(RobotConnectionError):
        connection.send_request({"type": "get_observation"}, timeout=0.2)


def test_close_fails_new_requests_immediately_instead_of_waiting_for_a_timeout(loop):
    connection, websocket = _connection(loop, {"type": "observation"})
    connection.close()

    with pytest.raises(RobotConnectionError, match="disconnected"):
        connection.send_request({"type": "get_observation"}, timeout=30)
    assert websocket.sent == []


def test_registry_is_keyed_by_tenant_and_robot_so_equal_robot_ids_do_not_collide():
    registry = RobotConnectionRegistry()
    a, b = object(), object()
    registry.register("tenant-a", "arm-1", a)
    registry.register("tenant-b", "arm-1", b)

    assert registry.get("tenant-a", "arm-1") is a
    assert registry.get("tenant-b", "arm-1") is b
    assert registry.get("tenant-a", "arm-2") is None


def test_unregistering_a_stale_connection_does_not_evict_its_replacement():
    # A robot that reconnects registers the new socket before the old
    # socket's handler finishes cleaning up.
    registry = RobotConnectionRegistry()
    old, new = object(), object()
    registry.register("t", "r", old)
    registry.register("t", "r", new)

    registry.unregister("t", "r", old)

    assert registry.get("t", "r") is new
