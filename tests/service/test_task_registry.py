from __future__ import annotations

from typing import Any

from par.core.agent import Agent
from par.core.planner import TASK_COMPLETE, Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.service.tasks.registry import TaskRecord, TaskRegistry, TaskStatus
from par.service.tasks.runner import request_cancel, start_task


class _ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


class _RepeatingPlanner(Planner):
    def propose(self, goal, observation, capabilities):
        return "detect", {}


class _BrokenPlanner(Planner):
    def propose(self, goal, observation, capabilities):
        raise RuntimeError("planner backend unreachable")


def _record(skill_registry: SkillRegistry, planner: Planner, max_steps: int = 5) -> TaskRecord:
    agent = Agent(skill_registry, planner=planner)
    runtime = Runtime(agent, MockRobot(), safety_kernel=SafetyKernel(load_profile("simulation")))
    return TaskRegistry().create("tenant-a", "robot-1", "goal", runtime, agent, max_steps)


def _run(record: TaskRecord) -> None:
    start_task(record)
    record.thread.join(timeout=10)
    assert not record.thread.is_alive()


def test_task_succeeds_when_the_planner_completes_and_reports_success(skill_registry):
    record = _record(
        skill_registry,
        _ScriptedPlanner([("pick", {"object": "red_object"}), (TASK_COMPLETE, {"message": "done", "success": True})]),
    )

    _run(record)

    assert record.status == TaskStatus.SUCCEEDED
    assert [r.success for r in record.results] == [True]
    assert record.agent.state.history[0]["action"]["skill_name"] == "pick"


def test_task_fails_when_the_planner_completes_but_reports_it_did_not_succeed(skill_registry):
    # task_complete only means the planner stopped; its own success flag
    # decides whether the task counts as succeeded.
    record = _record(skill_registry, _ScriptedPlanner([(TASK_COMPLETE, {"message": "blocked", "success": False})]))

    _run(record)

    assert record.status == TaskStatus.FAILED


def test_task_fails_when_an_executed_action_fails(skill_registry):
    record = _record(skill_registry, _ScriptedPlanner([("pick", {"object": "no_such_object"})]))

    _run(record)

    assert record.status == TaskStatus.FAILED
    assert record.results[-1].success is False


def test_task_is_incomplete_when_max_steps_runs_out_before_the_planner_finishes(skill_registry):
    record = _record(skill_registry, _RepeatingPlanner(), max_steps=3)

    _run(record)

    assert record.status == TaskStatus.INCOMPLETE
    assert len(record.results) == 3


def test_task_records_an_error_when_the_planner_raises_instead_of_dying_silently(skill_registry):
    # The exception happens on the task's own thread; without this, the
    # task would sit at "running" forever from a poller's point of view.
    record = _record(skill_registry, _BrokenPlanner())

    _run(record)

    assert record.status == TaskStatus.ERROR
    assert "planner backend unreachable" in record.error


def test_cancel_makes_every_further_step_a_safety_denial(skill_registry):
    record = _record(skill_registry, _RepeatingPlanner(), max_steps=3)
    request_cancel(record)

    _run(record)

    assert record.cancel_requested is True
    assert [r.success for r in record.results] == [False, False, False]
    assert all("emergency stop" in r.message for r in record.results)


def test_registry_lists_only_the_requesting_tenants_tasks(skill_registry):
    registry = TaskRegistry()
    agent = Agent(skill_registry)
    runtime = Runtime(agent, MockRobot())
    mine = registry.create("tenant-a", "robot-1", "g", runtime, agent, 1)
    registry.create("tenant-b", "robot-1", "g", runtime, agent, 1)

    assert [r.task_id for r in registry.list_for_tenant("tenant-a")] == [mine.task_id]
    assert registry.list_for_tenant("tenant-a", robot_id="robot-9") == []
    assert registry.get("missing") is None
