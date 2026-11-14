from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from par.core.agent import Agent
from par.core.planner import Planner, RuleBasedPlanner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.service.api.schemas import CancelAccepted, TaskAccepted, TaskCreate, TaskDetail
from par.service.auth import get_current_tenant, get_db
from par.service.db.models import LLMCredential, RobotCredential, Tenant
from par.service.robots.remote_robot import RemoteRobot
from par.service.tasks.registry import TERMINAL_STATUSES, TaskRecord
from par.service.tasks.runner import request_cancel, start_task
from par.skills import builtin_skills

router = APIRouter(prefix="/v1")


def build_planner(provider: str, api_key: str | None) -> Planner:
    """The key goes straight into the vendor SDK constructor; the service
    never falls back to process-global ANTHROPIC_API_KEY / GEMINI_API_KEY."""
    if provider == "rule_based":
        return RuleBasedPlanner()
    if provider == "anthropic":
        from par.core.llm_planner import LLMPlanner

        return LLMPlanner.from_api_key(api_key=api_key)
    if provider == "gemini":
        from par.core.gemini_planner import GeminiPlanner

        return GeminiPlanner.from_api_key(api_key=api_key)
    raise ValueError(f"unknown planner '{provider}'")


def _registry_for(profile_name: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = [profile_name]
        registry.register(skill)
    return registry


def _resolve_planner(request: Request, db: Session, tenant: Tenant, choice: str | None) -> Planner:
    if choice == "rule_based":
        return request.app.state.planner_factory("rule_based", None)
    credentials = {
        c.provider: c for c in db.scalars(select(LLMCredential).where(LLMCredential.tenant_id == tenant.id))
    }
    if choice is None:
        if len(credentials) != 1:
            raise HTTPException(
                status_code=400,
                detail="specify 'planner': this tenant has no LLM credential or more than one",
            )
        choice = next(iter(credentials))
    if choice not in credentials:
        raise HTTPException(status_code=400, detail=f"no '{choice}' credential configured for this tenant")
    api_key = request.app.state.cipher.decrypt(credentials[choice].encrypted_api_key)
    try:
        return request.app.state.planner_factory(choice, api_key)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"planner '{choice}' is not installed on this server") from exc


def _detail(record: TaskRecord) -> TaskDetail:
    state = record.agent.state
    return TaskDetail(
        task_id=record.task_id,
        robot_id=record.robot_id,
        goal=record.goal,
        status=record.status.value,
        agent_status=state.status.value,
        step_count=state.step_count,
        steps=list(state.history),
        completion_message=state.completion_message,
        final_result=None if record.results is None else [r.model_dump(mode="json") for r in record.results],
        error=record.error,
        cancel_requested=record.cancel_requested,
        created_at=record.created_at,
    )


def _owned_task(request: Request, tenant: Tenant, task_id: str) -> TaskRecord:
    record = request.app.state.tasks.get(task_id)
    # Another tenant's task is reported exactly like a missing one.
    if record is None or record.tenant_id != str(tenant.id):
        raise HTTPException(status_code=404, detail="unknown task")
    return record


@router.post("/tasks", status_code=202, response_model=TaskAccepted)
def submit_task(
    body: TaskCreate,
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> TaskAccepted:
    state = request.app.state
    registered = db.scalar(
        select(RobotCredential).where(
            RobotCredential.tenant_id == tenant.id, RobotCredential.robot_id == body.robot_id
        )
    )
    if registered is None:
        raise HTTPException(status_code=404, detail="unknown robot_id")
    connection = state.connections.get(str(tenant.id), body.robot_id)
    if connection is None:
        raise HTTPException(status_code=409, detail="robot not connected")

    planner = _resolve_planner(request, db, tenant, body.planner)
    robot = RemoteRobot(connection, request_timeout=state.settings.robot_request_timeout_seconds)
    agent = Agent(_registry_for(body.safety_profile), planner=planner)
    runtime = Runtime(agent, robot, safety_kernel=SafetyKernel(load_profile(body.safety_profile)))

    record = state.tasks.create(
        tenant_id=str(tenant.id),
        robot_id=body.robot_id,
        goal=body.goal,
        runtime=runtime,
        agent=agent,
        max_steps=body.max_steps,
    )
    start_task(record)
    return TaskAccepted(task_id=record.task_id, status="queued", robot_id=record.robot_id, goal=record.goal)


@router.get("/tasks/{task_id}", response_model=TaskDetail)
def get_task(task_id: str, request: Request, tenant: Tenant = Depends(get_current_tenant)) -> TaskDetail:
    return _detail(_owned_task(request, tenant, task_id))


@router.get("/tasks", response_model=list[TaskDetail])
def list_tasks(
    request: Request,
    robot_id: str | None = None,
    status: str | None = None,
    tenant: Tenant = Depends(get_current_tenant),
) -> list[TaskDetail]:
    records = request.app.state.tasks.list_for_tenant(str(tenant.id), robot_id)
    return [_detail(r) for r in records if status is None or r.status.value == status]


@router.post("/tasks/{task_id}/cancel", status_code=202, response_model=CancelAccepted)
def cancel_task(task_id: str, request: Request, tenant: Tenant = Depends(get_current_tenant)) -> CancelAccepted:
    record = _owned_task(request, tenant, task_id)
    if record.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail=f"task already {record.status.value}")
    request_cancel(record)
    return CancelAccepted(status="cancel_requested")
