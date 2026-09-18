from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from par.core.action import Action, ActionResult
from par.core.observation import Observation


class TelemetryEvent(BaseModel):
    task: str
    observation: Observation
    skill: str
    safety_decision: str
    rejection_reason: str | None = None
    action: Action
    result: ActionResult
    latency_seconds: float
    success: bool
    logged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
