from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Callable

from par.core.action import Action, ActionResult, ActionStatus
from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.core.world_state import WorldState
from par.robots.base import RobotInterface
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome, SafetyDecision
from par.telemetry.events import TelemetryEvent
from par.telemetry.logger import TelemetryLogger

HumanApprovalCallback = Callable[[Action, str], bool]


def _auto_deny(action: Action, reason: str) -> bool:
    """Default human-override callback: safe default when no human is present."""
    return False


class Runtime:
    def __init__(
        self,
        agent: Agent,
        robot: RobotInterface,
        safety_kernel: SafetyKernel | None = None,
        telemetry: TelemetryLogger | None = None,
        human_approval: HumanApprovalCallback = _auto_deny,
    ) -> None:
        self.agent = agent
        self.robot = robot
        self.safety_kernel = safety_kernel
        self.telemetry = telemetry or TelemetryLogger()
        self.human_approval = human_approval
        self.world_state = WorldState(agent_status=agent.state.status)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def run_once(self, goal: str) -> ActionResult | None:
        """Executes exactly one step toward goal. Returns None if already complete."""
        self.agent.set_goal(goal)
        return self._step(goal)

    def run_task(self, goal: str, max_steps: int = 10) -> list[ActionResult]:
        """Runs the ReAct-style loop until task_complete, failure, or max_steps.

        A Safety Kernel denial does not stop this loop - it's fed back to the
        planner (see Agent.record_rejection) so the next iteration re-plans.
        Only a genuine execution failure or task_complete ends it early.
        """
        self.agent.set_goal(goal)
        results: list[ActionResult] = []
        for _ in range(max_steps):
            result = self._step(goal)
            if result is None:
                break
            results.append(result)
            if self.agent.state.status == AgentStatus.FAILED:
                break
        return results

    def emergency_stop(self) -> None:
        if self.safety_kernel is not None:
            self.safety_kernel.emergency_stop()

    def clear_emergency_stop(self) -> None:
        if self.safety_kernel is not None:
            self.safety_kernel.clear_emergency_stop()

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _step(self, goal: str) -> ActionResult | None:
        started = time.monotonic()
        observation = self.robot.get_observation()
        action = self.agent.propose_action(observation)
        if action is None:
            self._update_world_state(observation, "n/a")
            return None

        capability = self.agent.skill_registry.get(action.skill_name).capability
        decision = self._evaluate_safety(action, observation, capability)

        if decision.outcome == PolicyOutcome.ESCALATE:
            decision = self._resolve_escalation(action, decision)

        if decision.outcome == PolicyOutcome.DENY:
            action.status = ActionStatus.REJECTED
            result = ActionResult(
                action_id=action.action_id,
                success=False,
                message=decision.reason,
                completed_at=datetime.now(timezone.utc),
            )
            self.agent.record_rejection(action, decision.reason)
        else:
            if decision.outcome == PolicyOutcome.MODIFY and decision.modified_parameters:
                action.parameters = {**action.parameters, **decision.modified_parameters}
            action.status = ActionStatus.APPROVED
            self.agent.state.status = AgentStatus.EXECUTING
            result = self._execute_with_timeout(action)
            self.agent.record_result(action, result)

        self.agent.planner.record_result(action, result)
        self._update_world_state(observation, decision.outcome.value)

        self.telemetry.log(
            TelemetryEvent(
                task=goal,
                observation=observation,
                skill=action.skill_name,
                safety_decision=decision.outcome.value,
                rejection_reason=decision.reason if decision.outcome == PolicyOutcome.DENY else None,
                action=action,
                result=result,
                latency_seconds=time.monotonic() - started,
                success=result.success,
            )
        )
        return result

    def _evaluate_safety(self, action: Action, observation, capability) -> SafetyDecision:
        if self.safety_kernel is None:
            return SafetyDecision(outcome=PolicyOutcome.ALLOW, reason="no Safety Kernel configured")
        admission = self.safety_kernel.admit(capability)
        if admission.outcome != PolicyOutcome.ALLOW:
            return admission
        return self.safety_kernel.check(action, observation, capability)

    def _resolve_escalation(self, action: Action, decision: SafetyDecision) -> SafetyDecision:
        if self.human_approval(action, decision.reason):
            return SafetyDecision(outcome=PolicyOutcome.ALLOW, reason="human approved")
        return SafetyDecision(outcome=PolicyOutcome.DENY, reason=f"human denied: {decision.reason}")

    def _execute_with_timeout(self, action: Action) -> ActionResult:
        timeout = self.safety_kernel.profile.action_timeout_seconds if self.safety_kernel else None
        future = self._executor.submit(self.robot.execute, action)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                message="action timed out",
                completed_at=datetime.now(timezone.utc),
            )

    def _update_world_state(self, observation, safety_decision: str) -> None:
        self.world_state.latest_observation = observation
        self.world_state.agent_status = self.agent.state.status
        self.world_state.last_safety_decision = safety_decision
        self.world_state.step_count = self.agent.state.step_count
