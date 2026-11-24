from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChannelReport(BaseModel):
    """What one perception channel says about its own health.

    Without this the Safety Kernel cannot tell "I looked and the world is
    clear" from "I could not look": both arrive as an empty `detections`
    list, and the kernel approves motion either way. A report makes the
    distinction survive the trip from the physical world into the digital
    one, so an adapter whose sensor has failed can say so and be refused
    rather than being read as reporting open space.

    `captured_at` is when the world was actually sensed, which is not
    `Observation.timestamp` - that is only when the packet was assembled,
    and an adapter serving a cached reading keeps it fresh indefinitely.
    """

    available: bool = True
    captured_at: datetime | None = None
    detail: str = ""


class Observation(BaseModel):
    observation_id: str
    timestamp: datetime
    # A plain string, not a closed Literal: a new physical backend should
    # never require an edit to a core model just to name itself.
    source: str
    robot_state: dict[str, Any] = Field(default_factory=dict)
    detections: list[dict[str, Any]] = Field(default_factory=list)
    # Keyed by channel name (par.core.perception.PROPRIOCEPTION /
    # PERCEPTION). An absent entry is no claim either way, which is how
    # adapters written before this contract behave; whether silence is
    # acceptable is a profile decision, not a property of the data.
    channels: dict[str, ChannelReport] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
