"""Safety Kernel ablations for Experiment 8 - research-only SafetyKernel
subclasses, each disabling exactly one governance mechanism so its individual
contribution to safety/task-completion can be isolated.

Never imported by par.safety or par.core: these intentionally weaken
governance and must not be reachable from production Runtime construction.
"""
from __future__ import annotations

from par.core.action import Action
from par.core.capability import Capability
from par.core.observation import Observation
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome, SafetyDecision


class NoAdmissionKernel(SafetyKernel):
    """Ablation 2: Admission always passes (except emergency stop)."""

    def admit(self, capability: Capability) -> SafetyDecision:
        if self.is_emergency_stopped:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason="emergency stop engaged")
        return SafetyDecision(outcome=PolicyOutcome.ALLOW)


class NoWorkspaceCheckKernel(SafetyKernel):
    """Ablation 3: workspace bounds are never checked."""

    def _check_workspace(self, target):
        return None


class NoCollisionCheckKernel(SafetyKernel):
    """Ablation 4: collision-margin is never checked."""

    def _check_collision(self, target, observation):
        return None


class NoVelocityCheckKernel(SafetyKernel):
    """Ablation 5: velocity is never checked - an over-limit move is ALLOWED
    unmodified, not clamped and not denied."""

    def _check_velocity(self, target, observation):
        return None


class NoModifyKernel(SafetyKernel):
    """Ablation 6: MODIFY is unavailable - whatever would have been clamped
    is denied outright instead. Isolates what MODIFY specifically buys over
    a strictly binary allow/deny policy (compare against Experiment 14)."""

    def check(self, action: Action, observation: Observation, capability: Capability) -> SafetyDecision:
        decision = super().check(action, observation, capability)
        if decision.outcome == PolicyOutcome.MODIFY:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason=f"[MODIFY disabled] {decision.reason}")
        return decision


class NoEscalateKernel(SafetyKernel):
    """Ablation 8: ESCALATE is unavailable - a high-risk action either runs
    unmodified or (if it also fails another check) is denied by that check,
    never routed to human approval. Mirrors SafetyKernel.check() minus the
    risk/approval_required branch, rather than mutating the shared profile
    (which isn't thread-safe and shouldn't be needed for a read-only check)."""

    def check(self, action: Action, observation: Observation, capability: Capability) -> SafetyDecision:
        if self.is_emergency_stopped:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason="emergency stop engaged")
        if action.skill_name == "move":
            return self._check_move(action, observation)
        return SafetyDecision(outcome=PolicyOutcome.ALLOW)


class NoPolicyGuardKernel(SafetyKernel):
    """Ablation 9: Policy Guard is skipped entirely - Admission is the only
    gate; anything admitted is ALLOWED regardless of runtime state."""

    def check(self, action: Action, observation: Observation, capability: Capability) -> SafetyDecision:
        if self.is_emergency_stopped:
            return SafetyDecision(outcome=PolicyOutcome.DENY, reason="emergency stop engaged")
        return SafetyDecision(outcome=PolicyOutcome.ALLOW)


ABLATIONS: dict[str, type[SafetyKernel]] = {
    "full_par": SafetyKernel,
    "no_admission": NoAdmissionKernel,
    "no_workspace_check": NoWorkspaceCheckKernel,
    "no_collision_check": NoCollisionCheckKernel,
    "no_velocity_check": NoVelocityCheckKernel,
    "no_modify": NoModifyKernel,
    "no_escalate": NoEscalateKernel,
    "no_policy_guard": NoPolicyGuardKernel,
}
