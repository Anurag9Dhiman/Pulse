"""Runtime variants for Experiment 4's comparison of recovery architectures.
Research-only - never imported by par.core, and never how production code
should construct a Runtime.
"""
from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.runtime import Runtime
from par.safety.policy import PolicyOutcome, SafetyDecision


class NoReplanRuntime(Runtime):
    """Config 2: LLM -> SafetyKernel -> Robot, no re-planning. A denial is
    terminal, exactly like a genuine execution failure - the task loop stops
    (agent.state.status becomes FAILED) instead of feeding the reason back to
    the planner."""

    def _handle_denial(self, action: Action, decision: SafetyDecision) -> ActionResult:
        result = ActionResult(
            action_id=action.action_id,
            success=False,
            message=decision.reason,
            completed_at=datetime.now(timezone.utc),
        )
        self.agent.record_result(action, result)  # sets FAILED, unlike record_rejection
        return result


class FixedRecoveryRuntime(Runtime):
    """Config 3: LLM -> SafetyKernel -> RecoveryModule -> Robot. On a denial,
    applies a fixed, hand-coded strategy (scale the move distance down,
    retry up to max_recovery_attempts times) entirely independent of the
    planner's own reasoning - the traditional architecture PAR's
    feedback-based re-planning (plain Runtime) is compared against. Only
    "move" actions with x/y/z parameters can be recovered this way; anything
    else exhausts immediately.
    """

    def __init__(self, *args, max_recovery_attempts: int = 3, backoff: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_recovery_attempts = max_recovery_attempts
        self._backoff = backoff

    def _handle_denial(self, action: Action, decision: SafetyDecision) -> ActionResult:
        if action.skill_name == "move" and self.safety_kernel is not None:
            capability = self.agent.skill_registry.get(action.skill_name).capability
            for attempt in range(1, self._max_recovery_attempts + 1):
                scale = self._backoff**attempt
                scaled_params = {axis: action.parameters.get(axis, 0.0) * scale for axis in ("x", "y", "z")}
                retry_action = action.model_copy(update={"parameters": scaled_params})
                observation = self.robot.get_observation()
                retry_decision = self.safety_kernel.check(retry_action, observation, capability)
                if retry_decision.outcome in (PolicyOutcome.ALLOW, PolicyOutcome.MODIFY):
                    if retry_decision.modified_parameters:
                        retry_action.parameters = {**retry_action.parameters, **retry_decision.modified_parameters}
                    result = self._execute_with_timeout(retry_action, capability)
                    self.agent.record_result(retry_action, result)
                    return result

        result = ActionResult(
            action_id=action.action_id,
            success=False,
            message=f"[RecoveryModule] exhausted {self._max_recovery_attempts} attempts; last denial: {decision.reason}",
            completed_at=datetime.now(timezone.utc),
        )
        self.agent.record_result(action, result)
        return result
