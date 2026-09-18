from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"


class AgentState(BaseModel):
    agent_id: str
    goal: str | None = None
    status: AgentStatus = AgentStatus.IDLE
    step_count: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
