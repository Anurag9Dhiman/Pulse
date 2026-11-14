from __future__ import annotations

import threading

from par.core.agent_state import AgentStatus
from par.service.tasks.registry import TaskRecord, TaskStatus


def start_task(record: TaskRecord) -> None:
    """Runs Runtime.run_task() to completion on a dedicated thread. Progress
    is read live off record.agent.state / record.runtime.world_state, which
    run_task already mutates after every step."""

    def _drive() -> None:
        try:
            record.status = TaskStatus.RUNNING
            record.results = record.runtime.run_task(record.goal, max_steps=record.max_steps)
            agent_status = record.agent.state.status
            if agent_status == AgentStatus.FAILED:
                record.status = TaskStatus.FAILED
            elif agent_status == AgentStatus.DONE:
                record.status = (
                    TaskStatus.SUCCEEDED if record.agent.state.reported_success else TaskStatus.FAILED
                )
            else:
                record.status = TaskStatus.INCOMPLETE
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.status = TaskStatus.ERROR
        finally:
            record.runtime.close()

    record.thread = threading.Thread(target=_drive, daemon=True, name=f"par-task-{record.task_id}")
    record.thread.start()


def request_cancel(record: TaskRecord) -> None:
    """Best-effort: engages the task's own emergency stop, so every further
    step is denied by the Safety Kernel until the loop runs out. A thread
    cannot be interrupted mid-action."""
    record.cancel_requested = True
    record.runtime.emergency_stop()
