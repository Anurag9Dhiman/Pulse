from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

LLMProvider = Literal["anthropic", "gemini"]
PlannerChoice = Literal["anthropic", "gemini", "rule_based"]
SafetyProfileName = Literal["simulation", "real_robot"]


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class TenantCreated(BaseModel):
    tenant_id: str


class ApiKeyCreate(BaseModel):
    label: str | None = Field(default=None, max_length=200)


class ApiKeyCreated(BaseModel):
    api_key: str  # shown once; only its hash is stored
    key_prefix: str


class LLMCredentialSet(BaseModel):
    api_key: str = Field(min_length=1)


class LLMCredentialStatus(BaseModel):
    provider: LLMProvider
    configured: bool


class RobotCreate(BaseModel):
    robot_id: str = Field(min_length=1, max_length=200)


class RobotCreated(BaseModel):
    robot_id: str
    robot_token: str  # shown once; only its hash is stored


class RobotInfo(BaseModel):
    robot_id: str
    connected: bool
    created_at: datetime


class TaskCreate(BaseModel):
    robot_id: str
    goal: str = Field(min_length=1)
    planner: PlannerChoice | None = None
    max_steps: int = Field(default=10, ge=1, le=200)
    safety_profile: SafetyProfileName = "simulation"


class TaskAccepted(BaseModel):
    task_id: str
    status: str
    robot_id: str
    goal: str


class TaskDetail(BaseModel):
    task_id: str
    robot_id: str
    goal: str
    status: str
    agent_status: str
    step_count: int
    steps: list[dict[str, Any]]
    completion_message: str | None
    final_result: list[dict[str, Any]] | None
    error: str | None
    cancel_requested: bool
    created_at: datetime


class CancelAccepted(BaseModel):
    status: str
