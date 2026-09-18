from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from par.core.observation import Observation


class ActionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Action(BaseModel):
    action_id: str
    skill_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime


class ActionResult(BaseModel):
    action_id: str
    success: bool
    message: str = ""
    completed_at: datetime
    observation: Observation | None = None
