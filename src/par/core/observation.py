from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    observation_id: str
    timestamp: datetime
    source: Literal["mock", "ros2", "camera"]
    robot_state: dict[str, Any] = Field(default_factory=dict)
    detections: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
