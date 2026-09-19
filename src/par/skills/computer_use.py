from __future__ import annotations

from par.core.capability import Capability, RiskLevel
from par.core.skill import ParameterizedSkill, Skill

_DESCRIPTION = (
    "Delegate a task to the digital world when the physical task needs to "
    "read, write, or look something up on a computer that the robot cannot "
    "touch directly. Runs on CollectiveOS's Navigation Agent and returns the "
    "result once the computer task completes."
)

DEFAULT_TIMEOUT_SECONDS = 180.0


def computer_use_skill(execution_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> Skill:
    capability = Capability(
        name="use_computer",
        description=_DESCRIPTION,
        input_schema={"task": "str"},
        risk=RiskLevel.HIGH,
        supports_rollback=False,
        env_profiles=["simulation", "real_robot"],
        execution_timeout_seconds=execution_timeout_seconds,
    )
    return ParameterizedSkill(capability, required_params={"task"})
