from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Capability(BaseModel):
    name: str
    description: str
    input_schema: dict[str, str] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    supports_rollback: bool = False
    env_profiles: list[str] = Field(default_factory=lambda: ["simulation"])
