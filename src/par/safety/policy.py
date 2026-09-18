from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class PolicyOutcome(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    DENY = "deny"
    ESCALATE = "escalate"


class SafetyDecision(BaseModel):
    outcome: PolicyOutcome
    reason: str = ""
    modified_parameters: dict[str, Any] | None = None
