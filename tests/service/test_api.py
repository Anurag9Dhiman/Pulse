from __future__ import annotations

import time

import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from par.core.planner import TASK_COMPLETE
from par.service.db.models import ApiKey, LLMCredential

ADMIN = {"X-Admin-Token": "test-admin-token"}
_TERMINAL = {"succeeded", "failed", "incomplete", "error"}


def _tenant(client, name: str = "acme") -> dict[str, str]:
    tenant_id = client.post("/v1/admin/tenants", json={"name": name}, headers=ADMIN).json()["tenant_id"]
    created = client.post(f"/v1/admin/tenants/{tenant_id}/api-keys", json={}, headers=ADMIN).json()
    return {"Authorization": f"Bearer {created['api_key']}"}


def _robot(client, headers: dict[str, str], robot_id: str = "arm-1") -> str:
    return client.post("/v1/robots", json={"robot_id": robot_id}, headers=headers).json()["robot_token"]


def _wait_connected(client, headers: dict[str, str], robot_id: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        robots = {r["robot_id"]: r["connected"] for r in client.get("/v1/robots", headers=headers).json()}
        if robots.get(robot_id):
            return
        time.sleep(0.02)
    raise AssertionError(f"{robot_id} never showed as connected")


def _wait_terminal(client, headers: dict[str, str], task_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = client.get(f"/v1/tasks/{task_id}", headers=headers).json()
        if task["status"] in _TERMINAL:
            return task
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} never finished: {task}")


def test_health_and_readiness_need_no_credentials(client):
    assert client.get("/v1/healthz").json() == {"status": "ok"}
    assert client.get("/v1/readyz").status_code == 200


def test_admin_endpoints_reject_a_missing_or_wrong_admin_token(client):
    assert client.post("/v1/admin/tenants", json={"name": "x"}).status_code == 401
    assert client.post("/v1/admin/tenants", json={"name": "x"}, headers={"X-Admin-Token": "nope"}).status_code == 401


def test_tenant_endpoints_reject_missing_invalid_and_revoked_api_keys(client, app):
    headers = _tenant(client)
    assert client.get("/v1/robots").status_code == 401
    assert client.get("/v1/robots", headers={"Authorization": "Bearer par_not-a-real-key"}).status_code == 401
    assert client.get("/v1/robots", headers=headers).status_code == 200

    from datetime import datetime, timezone

    with app.state.session_factory() as session:
        session.scalar(select(ApiKey)).revoked_at = datetime.now(timezone.utc)
        session.commit()
    assert client.get("/v1/robots", headers=headers).status_code == 401


def test_api_keys_are_stored_hashed_never_in_plaintext(client, app):
    headers = _tenant(client)
    plaintext = headers["Authorization"].removeprefix("Bearer ")

    with app.state.session_factory() as session:
        stored = session.scalar(select(ApiKey))

    assert plaintext not in (stored.key_hash, stored.key_prefix)
    assert plaintext.startswith(stored.key_prefix)


def test_llm_credentials_are_encrypted_at_rest_and_round_trip_through_the_cipher(client, app):
    headers = _tenant(client)

    response = client.put("/v1/llm-credentials/gemini", json={"api_key": "sk-tenant-secret"}, headers=headers)

    assert response.json() == {"provider": "gemini", "configured": True}
    with app.state.session_factory() as session:
        stored = session.scalar(select(LLMCredential))
    assert "sk-tenant-secret" not in stored.encrypted_api_key
    assert app.state.cipher.decrypt(stored.encrypted_api_key) == "sk-tenant-secret"


def test_unknown_llm_provider_is_rejected(client):
    headers = _tenant(client)
    assert client.put("/v1/llm-credentials/openai", json={"api_key": "k"}, headers=headers).status_code == 422


def test_one_tenant_cannot_see_or_delete_another_tenants_robots(client):
    acme, globex = _tenant(client, "acme"), _tenant(client, "globex")
    _robot(client, acme, "arm-1")

    assert client.get("/v1/robots", headers=globex).json() == []
    assert client.delete("/v1/robots/arm-1", headers=globex).status_code == 404
    assert [r["robot_id"] for r in client.get("/v1/robots", headers=acme).json()] == ["arm-1"]


def test_registering_the_same_robot_id_twice_conflicts(client):
    headers = _tenant(client)
    _robot(client, headers, "arm-1")
    assert client.post("/v1/robots", json={"robot_id": "arm-1"}, headers=headers).status_code == 409


def test_robot_websocket_rejects_an_unknown_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/robots/ws?token=parrobot_wrong"):
            pass


def test_task_submission_distinguishes_unknown_robot_disconnected_robot_and_missing_credential(client):
    headers = _tenant(client)
    body = {"robot_id": "arm-1", "goal": "stop the robot"}
    assert client.post("/v1/tasks", json=body, headers=headers).status_code == 404

    token = _robot(client, headers, "arm-1")
    assert client.post("/v1/tasks", json=body, headers=headers).status_code == 409

    with client.websocket_connect(f"/v1/robots/ws?token={token}"):
        _wait_connected(client, headers, "arm-1")
        response = client.post("/v1/tasks", json=body, headers=headers)
        assert response.status_code == 400  # connected, but no LLM credential and no planner named


def test_task_runs_end_to_end_through_a_robot_connected_over_the_websocket(client, planner_factory):
    """The whole boundary in one pass: tenant auth, the tenant's own
    decrypted LLM key reaching the planner, governance by the unmodified
    Runtime/Safety Kernel, and actions executed by a robot that dialed in."""
    headers = _tenant(client)
    client.put("/v1/llm-credentials/gemini", json={"api_key": "sk-tenant-secret"}, headers=headers)
    token = _robot(client, headers, "arm-1")
    planner_factory.steps = [
        ("move", {"x": 0.5, "y": 0.0, "z": 0.0}),
        (TASK_COMPLETE, {"message": "moved", "success": True}),
    ]

    with client.websocket_connect(f"/v1/robots/ws?token={token}") as robot:
        _wait_connected(client, headers, "arm-1")
        accepted = client.post("/v1/tasks", json={"robot_id": "arm-1", "goal": "move a bit"}, headers=headers)
        assert accepted.status_code == 202
        task_id = accepted.json()["task_id"]

        observe = robot.receive_json()
        assert observe["type"] == "get_observation"
        observation = {"type": "observation", "robot_state": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}, "detections": []}
        robot.send_json({**observation, "request_id": observe["request_id"]})

        action = robot.receive_json()
        assert (action["type"], action["skill_name"], action["parameters"]) == (
            "action",
            "move",
            {"x": 0.5, "y": 0.0, "z": 0.0},
        )
        robot.send_json({"type": "action_result", "request_id": action["request_id"], "success": True, "message": "reached"})

        observe = robot.receive_json()
        robot.send_json({**observation, "request_id": observe["request_id"]})

        task = _wait_terminal(client, headers, task_id)

    assert task["status"] == "succeeded"
    assert task["completion_message"] == "moved"
    assert task["step_count"] == 1
    assert task["steps"][0]["result"]["message"] == "reached"
    assert planner_factory.calls == [("gemini", "sk-tenant-secret")]


def test_safety_kernel_still_governs_tasks_submitted_through_the_service(client, planner_factory):
    # x=50 is outside the simulation workspace: the action must be denied
    # inside the service process and never be sent to the robot at all.
    headers = _tenant(client)
    token = _robot(client, headers, "arm-1")
    planner_factory.steps = [
        ("move", {"x": 50.0, "y": 0.0, "z": 0.0}),
        (TASK_COMPLETE, {"message": "blocked", "success": False}),
    ]
    observation = {"type": "observation", "robot_state": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}, "detections": []}

    with client.websocket_connect(f"/v1/robots/ws?token={token}") as robot:
        _wait_connected(client, headers, "arm-1")
        task_id = client.post(
            "/v1/tasks", json={"robot_id": "arm-1", "goal": "go far", "planner": "rule_based"}, headers=headers
        ).json()["task_id"]

        received = []
        for _ in range(2):  # one observation per step; no "action" message in between
            message = robot.receive_json()
            received.append(message["type"])
            robot.send_json({**observation, "request_id": message["request_id"]})
        task = _wait_terminal(client, headers, task_id)

    assert received == ["get_observation", "get_observation"]
    assert task["status"] == "failed"
    assert task["steps"][0]["result"]["rejected_by_safety"] is True


def test_a_task_is_invisible_to_other_tenants(client, planner_factory):
    acme, globex = _tenant(client, "acme"), _tenant(client, "globex")
    token = _robot(client, acme, "arm-1")
    planner_factory.steps = [(TASK_COMPLETE, {"message": "nothing to do", "success": True})]

    with client.websocket_connect(f"/v1/robots/ws?token={token}") as robot:
        _wait_connected(client, acme, "arm-1")
        task_id = client.post(
            "/v1/tasks", json={"robot_id": "arm-1", "goal": "noop", "planner": "rule_based"}, headers=acme
        ).json()["task_id"]
        message = robot.receive_json()
        robot.send_json({"type": "observation", "request_id": message["request_id"]})
        _wait_terminal(client, acme, task_id)

    assert client.get(f"/v1/tasks/{task_id}", headers=globex).status_code == 404
    assert client.post(f"/v1/tasks/{task_id}/cancel", headers=globex).status_code == 404
    assert client.get("/v1/tasks", headers=globex).json() == []
    assert [t["task_id"] for t in client.get("/v1/tasks", headers=acme).json()] == [task_id]


def test_robot_disconnecting_mid_task_fails_the_task_instead_of_hanging_it(client, planner_factory):
    headers = _tenant(client)
    token = _robot(client, headers, "arm-1")
    planner_factory.steps = [("stop", {})]

    with client.websocket_connect(f"/v1/robots/ws?token={token}") as robot:
        _wait_connected(client, headers, "arm-1")
        task_id = client.post(
            "/v1/tasks", json={"robot_id": "arm-1", "goal": "stop", "planner": "rule_based"}, headers=headers
        ).json()["task_id"]
        message = robot.receive_json()
        robot.send_json({"type": "observation", "request_id": message["request_id"]})
        assert robot.receive_json()["type"] == "action"
        # leave the block without answering the action: the bridge drops

    task = _wait_terminal(client, headers, task_id)
    assert task["status"] == "failed"
    assert "disconnected" in task["steps"][0]["result"]["message"]
