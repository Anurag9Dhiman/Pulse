from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from par.core.action import Action
from par.core.observation import ChannelReport, Observation
from par.core.perception import PERCEPTION, PROPRIOCEPTION


def _channels(payload: dict[str, Any]) -> dict[str, ChannelReport]:
    """Channel health for a wire payload.

    A bridge that speaks the contract sends a "channels" block and it is
    used as-is. One that does not gets a conservative reading of its
    error field instead: a transport that failed has told us nothing
    about the world, and the empty `robot_state`/`detections` it returns
    alongside `raw.error` must not be read as "the robot is at the origin
    and nothing is in the way".
    """
    declared = payload.get("channels")
    if isinstance(declared, dict) and declared:
        return {name: ChannelReport(**report) for name, report in declared.items()}

    error = (payload.get("raw") or {}).get("error")
    if error:
        unavailable = ChannelReport(available=False, detail=str(error))
        return {PROPRIOCEPTION: unavailable, PERCEPTION: unavailable}
    return {}


def payload_to_observation(payload: dict[str, Any]) -> Observation:
    return Observation(
        observation_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        source="ros2",
        robot_state=payload.get("robot_state", {}),
        detections=payload.get("detections", []),
        channels=_channels(payload),
        raw=payload.get("raw", {}),
    )


def action_to_payload(action: Action) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "skill_name": action.skill_name,
        "parameters": action.parameters,
    }
