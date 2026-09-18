from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from par.core.action import Action
from par.core.observation import Observation


def payload_to_observation(payload: dict[str, Any]) -> Observation:
    return Observation(
        observation_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        source="ros2",
        robot_state=payload.get("robot_state", {}),
        detections=payload.get("detections", []),
    )


def action_to_payload(action: Action) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "skill_name": action.skill_name,
        "parameters": action.parameters,
    }
