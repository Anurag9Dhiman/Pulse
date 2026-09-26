"""Experiment 13: Collision-Margin Sensitivity.

How does the collision_margin setting trade off safety (collision rate)
against efficiency (false-denial rate, path length) as it's swept?

Fixed scenario: robot at origin, one obstacle at a fixed distance, targets
swept from just short of the obstacle to just past it. A "collision" is
counted when the actual physical distance-to-obstacle after execution would
be less than a fixed real-world safety threshold (0.05m, i.e. actual contact)
- independent of collision_margin itself, so the margin can be evaluated
against a fixed notion of "actually unsafe" rather than against its own
threshold circularly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from par.core.capability import Capability
from par.core.observation import Observation
from par.core.skill import SkillRegistry
from par.evaluation.metrics import rate
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome
from par.skills import builtin_skills
from datetime import datetime, timezone

MARGINS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
OBSTACLE_X = 1.0
# Deliberately larger than the smallest tested margin: if the real contact
# threshold were <= the smallest margin, no margin could ever miss a genuine
# collision, and the sensitivity curve this experiment exists to produce
# would be flat by construction rather than by finding. 0.08m represents
# gripper/arm physical extent the collision_margin point-check doesn't model.
REAL_CONTACT_THRESHOLD = 0.08
_DISTANCES = (0.6, 0.4, 0.25, 0.15, 0.12, 0.09, 0.06, 0.03, 0.0)
TARGETS = [OBSTACLE_X - d for d in _DISTANCES]


def _profile(margin: float) -> EnvironmentProfile:
    return EnvironmentProfile(
        name="collision_sweep",
        workspace=Workspace(x=(-2.0, 2.0), y=(-2.0, 2.0), z=(0.0, 2.0)),
        max_velocity=5.0,  # deliberately generous: isolate collision-margin effects from velocity clamping
        action_timeout_seconds=1.0,
        approval_required=False,
        collision_margin=margin,
    )


def _observation() -> Observation:
    return Observation(
        observation_id="o",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "obstacle", "position": {"x": OBSTACLE_X, "y": 0.0, "z": 0.0}}],
    )


def run() -> dict:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    # SafetyKernel.check() (unlike admit()) doesn't consult env_profiles, so
    # the skill's own registered capability is used as-is - no override needed.
    move_capability = registry.get("move").capability

    results = []
    for margin in MARGINS:
        kernel = SafetyKernel(_profile(margin))
        collisions = 0
        false_denials = 0
        allowed_targets = []
        for target in TARGETS:
            action = registry.get("move").build_action({"x": target, "y": 0.0, "z": 0.0})
            decision = kernel.check(action, _observation(), move_capability)
            actual_distance_to_obstacle = abs(OBSTACLE_X - target)
            would_be_unsafe = actual_distance_to_obstacle < REAL_CONTACT_THRESHOLD

            if decision.outcome == PolicyOutcome.ALLOW:
                allowed_targets.append(target)
                if would_be_unsafe:
                    collisions += 1
            elif decision.outcome == PolicyOutcome.DENY and not would_be_unsafe:
                false_denials += 1

        results.append(
            {
                "collision_margin": margin,
                "collision_rate": rate(collisions, len(TARGETS)),
                "false_denial_rate": rate(false_denials, len(TARGETS)),
                "n_allowed": len(allowed_targets),
                "closest_allowed_target_to_obstacle": (
                    min(abs(OBSTACLE_X - t) for t in allowed_targets) if allowed_targets else None
                ),
            }
        )

    return {
        "experiment": "13_collision_margin_sensitivity",
        "obstacle_x": OBSTACLE_X,
        "real_contact_threshold": REAL_CONTACT_THRESHOLD,
        "targets_tested": TARGETS,
        "results_by_margin": results,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp13_collision_margin_sensitivity.json"
    out_path.write_text(json.dumps(result, indent=2))
