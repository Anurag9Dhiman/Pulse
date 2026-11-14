from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from par.core.action import ActionResult
from par.core.agent import Agent
from par.core.runtime import Runtime


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"  # max_steps reached without task_complete or a failure
    ERROR = "error"  # the driver thread itself raised


TERMINAL_STATUSES = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.INCOMPLETE, TaskStatus.ERROR}


@dataclass
class TaskRecord:
    task_id: str
    tenant_id: str
    robot_id: str
    goal: str
    max_steps: int
    runtime: Runtime
    agent: Agent
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    thread: threading.Thread | None = None
    results: list[ActionResult] | None = None
    error: str | None = None
    cancel_requested: bool = False


class TaskRegistry:
    """One Runtime+Agent per task, keyed by task_id, held in this process's
    memory only. A restart loses every in-flight task, and a second
    worker/replica cannot see tasks this one holds - the service must run
    as a single process until task state is persisted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskRecord] = {}

    def create(
        self, tenant_id: str, robot_id: str, goal: str, runtime: Runtime, agent: Agent, max_steps: int
    ) -> TaskRecord:
        record = TaskRecord(
            task_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            robot_id=robot_id,
            goal=goal,
            max_steps=max_steps,
            runtime=runtime,
            agent=agent,
        )
        with self._lock:
            self._tasks[record.task_id] = record
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_for_tenant(self, tenant_id: str, robot_id: str | None = None) -> list[TaskRecord]:
        with self._lock:
            records = list(self._tasks.values())
        return [
            r for r in records if r.tenant_id == tenant_id and (robot_id is None or r.robot_id == robot_id)
        ]
