from __future__ import annotations

from pydantic import BaseModel

from par.core.agent_state import AgentStatus
from par.core.observation import Observation


class WorldState(BaseModel):
    """Minimal aggregate view for dashboards/telemetry, not a planning input."""

    latest_observation: Observation | None = None
    agent_status: AgentStatus = AgentStatus.IDLE
    last_safety_decision: str | None = None
    step_count: int = 0
