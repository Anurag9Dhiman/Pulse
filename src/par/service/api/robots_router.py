from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from par.service.auth import authenticate_robot_token
from par.service.robots.connection_registry import RemoteRobotConnection

router = APIRouter()


@router.websocket("/v1/robots/ws")
async def robot_ws(websocket: WebSocket, token: str = "") -> None:
    """Inbound connection from a robot's own bridge process.

    Server -> robot: {"type": "get_observation", "request_id"}
                     {"type": "action", "request_id", "action_id", "skill_name", "parameters"}
    Robot -> server: {"type": "observation", "request_id", "robot_state", "detections", "raw"}
                     {"type": "action_result", "request_id", "success", "message"}
    """
    state = websocket.app.state
    identity = await run_in_threadpool(authenticate_robot_token, state.session_factory, token)
    if identity is None:
        await websocket.close(code=4401, reason="invalid robot token")
        return

    await websocket.accept()
    connection = RemoteRobotConnection(websocket, asyncio.get_running_loop())
    tenant_id = str(identity.tenant_id)
    state.connections.register(tenant_id, identity.robot_id, connection)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if isinstance(message, dict):
                connection.on_message(message)
    except WebSocketDisconnect:
        pass
    finally:
        state.connections.unregister(tenant_id, identity.robot_id, connection)
        connection.close()
