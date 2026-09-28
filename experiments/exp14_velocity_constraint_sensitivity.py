"""Experiment 14: Velocity Constraint Sensitivity.

Does MODIFY (clamp) provide a useful alternative to DENY for actions that
violate only a velocity constraint (not workspace, not collision)?

Compares three policies at the same set of requested distances, all within
workspace bounds and away from any obstacle - isolating velocity as the only
variable: (1) hard DENY on any velocity violation, (2) PAR's actual MODIFY
behavior, (3) unconstrained (no velocity check at all).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from par.core.observation import Observation
from par.core.skill import SkillRegistry
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome
from par.skills import builtin_skills

MAX_VELOCITY = 0.5
ACTION_TIMEOUT = 1.0  # max reachable distance = 0.5m
MAX_REACHABLE = MAX_VELOCITY * ACTION_TIMEOUT
# Below, near, and above the permitted limit.
REQUESTED_DISTANCES = [0.1, 0.3, 0.45, 0.5, 0.6, 0.8, 1.2, 1.8]


def _profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        name="velocity_sweep",
        workspace=Workspace(x=(-2.0, 2.0), y=(-2.0, 2.0), z=(0.0, 2.0)),  # generous: isolate velocity from workspace
        max_velocity=MAX_VELOCITY,
        action_timeout_seconds=ACTION_TIMEOUT,
        approval_required=False,
        collision_margin=0.0,  # no obstacles in this experiment; isolate velocity from collision
    )


def _observation() -> Observation:
    return Observation(
        observation_id="o",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )


def run() -> dict:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    capability = registry.get("move").capability
    kernel = SafetyKernel(_profile())

    rows = []
    for distance in REQUESTED_DISTANCES:
        action = registry.get("move").build_action({"x": distance, "y": 0.0, "z": 0.0})
        decision = kernel.check(action, _observation(), capability)
        violates_velocity = distance > MAX_REACHABLE

        # Policy 1: hard DENY on any velocity violation (no MODIFY at all)
        deny_only_result = 0.0 if violates_velocity else distance  # denied -> robot doesn't move
        deny_only_task_progress = not violates_velocity

        # Policy 2: PAR's actual behavior (MODIFY clamps, doesn't reject)
        if decision.outcome == PolicyOutcome.MODIFY and decision.modified_parameters:
            par_result = decision.modified_parameters["x"]
        elif decision.outcome == PolicyOutcome.ALLOW:
            par_result = distance
        else:
            par_result = 0.0
        par_task_progress = decision.outcome in (PolicyOutcome.ALLOW, PolicyOutcome.MODIFY)

        # Policy 3: unconstrained (no velocity check - direct execution)
        unconstrained_result = distance
        unconstrained_task_progress = True

        rows.append(
            {
                "requested_distance": distance,
                "violates_velocity_limit": violates_velocity,
                "deny_only": {"resulting_distance": deny_only_result, "made_progress": deny_only_task_progress},
                "par_modify": {
                    "outcome": decision.outcome.value,
                    "resulting_distance": par_result,
                    "made_progress": par_task_progress,
                },
                "unconstrained": {"resulting_distance": unconstrained_result, "made_progress": unconstrained_task_progress},
            }
        )

    n = len(rows)
    return {
        "experiment": "14_velocity_constraint_sensitivity",
        "max_velocity": MAX_VELOCITY,
        "action_timeout_seconds": ACTION_TIMEOUT,
        "max_reachable_distance": MAX_REACHABLE,
        "rows": rows,
        "summary": {
            "deny_only_task_progress_rate": sum(r["deny_only"]["made_progress"] for r in rows) / n,
            "par_modify_task_progress_rate": sum(r["par_modify"]["made_progress"] for r in rows) / n,
            "unconstrained_task_progress_rate": sum(r["unconstrained"]["made_progress"] for r in rows) / n,
            "par_modify_safety_violations": sum(
                1 for r in rows if r["par_modify"]["resulting_distance"] > MAX_REACHABLE + 1e-9
            ),
            "unconstrained_safety_violations": sum(1 for r in rows if r["violates_velocity_limit"]),
        },
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp14_velocity_constraint_sensitivity.json"
    out_path.write_text(json.dumps(result, indent=2))
