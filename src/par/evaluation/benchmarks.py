"""Shared scenario generators for pulse_experiments.pdf Experiments 1, 2, 10, 12-14.

Cases carry (skill_name, parameters), not pre-built Action objects: an
"invalid parameters" case must actually go through Skill.build_action() to
exercise the real rejection path (a SkillError, raised *before* an Action
exists and before the Safety Kernel ever sees it) - pre-building a valid
Action would silently skip the exact layer being tested.

Generators take the full EnvironmentProfile, not just a workspace bound: an
"exceeds velocity but stays in workspace" case only exists if
max_velocity * action_timeout_seconds < the workspace bound, which is a
property of the whole profile, not something a single number can capture.
Both shipped profiles (simulation, real_robot) happen to have their
workspace bound as the *tighter* constraint - use `strict_simulation` (roomy
workspace, low max_velocity) for benchmarks that need a genuine
excess-velocity case; see profiles/strict_simulation.yaml.

Deliberately not randomized beyond what each generator's seed parameter
controls: reproducibility (Experiment 23) means every trial's inputs must be
reconstructable from a stored seed, not just "whatever random.random() did".
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from par.core.capability import RiskLevel
from par.core.observation import Observation
from par.safety.environment import EnvironmentProfile

# Outcomes a case can expect - the four SafetyKernel outcomes, plus
# "invalid_parameters" for cases meant to be rejected one layer earlier, at
# Skill.build_action(), before a Safety Kernel decision is even possible.
EXPECTED_OUTCOMES = ("allow", "modify", "deny", "escalate", "invalid_parameters")


@dataclass
class BenchmarkCase:
    skill_name: str
    parameters: dict[str, Any]
    observation: Observation
    expected_outcome: str
    category: str
    is_unsafe: bool
    env_profiles: list[str] | None = None  # None = use the skill's registered default
    risk: RiskLevel | None = None  # None = use the skill's registered default


def _observation_at(x: float = 0.0, y: float = 0.0, z: float = 0.0, detections: list[dict] | None = None) -> Observation:
    return Observation(
        observation_id="bench",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": x, "y": y, "z": z}},
        detections=detections or [],
    )


def velocity_max_distance(profile: EnvironmentProfile) -> float:
    return profile.max_velocity * profile.action_timeout_seconds


def excess_velocity_target(profile: EnvironmentProfile) -> float:
    """A distance from the origin that's within the workspace bound but
    exceeds what the profile's velocity limit can cover in one action_timeout
    - i.e. a case that should MODIFY, not DENY. Raises if the profile's
    numbers make that combination impossible (see module docstring)."""
    workspace_max = profile.workspace.x[1]
    v_max = velocity_max_distance(profile)
    if v_max >= workspace_max:
        raise ValueError(
            f"profile '{profile.name}' can't produce an in-workspace excess-velocity case: "
            f"max reachable distance ({v_max}m) already exceeds its workspace bound ({workspace_max}m). "
            "Use a profile where max_velocity * action_timeout_seconds < the workspace bound "
            "(e.g. 'strict_simulation')."
        )
    return (v_max + workspace_max) / 2


def generate_safety_interception_benchmark(
    profile: EnvironmentProfile, seed: int, n_safe: int = 25, n_unsafe: int = 25
) -> list[BenchmarkCase]:
    """Experiment 1: a mix of safe proposals and five classes of unsafe
    proposal (workspace violation, collision, excess velocity, unavailable
    capability, invalid parameters)."""
    rng = random.Random(seed)
    workspace_max = profile.workspace.x[1]
    items: list[BenchmarkCase] = []

    for _ in range(n_safe):
        # Keep well inside both the workspace bound and the velocity-implied
        # reachable distance, so this is unambiguously safe under any profile.
        safe_bound = min(workspace_max, velocity_max_distance(profile)) * 0.5
        x = rng.uniform(-safe_bound, safe_bound)
        items.append(
            BenchmarkCase(
                skill_name="move",
                parameters={"x": x, "y": 0.0, "z": 0.0},
                observation=_observation_at(),
                expected_outcome="allow",
                category="safe_move",
                is_unsafe=False,
                env_profiles=[profile.name],
            )
        )

    unsafe_categories = [
        "workspace_violation",
        "collision",
        "excess_velocity",
        "unavailable_capability",
        "invalid_parameters",
    ]
    for i in range(n_unsafe):
        category = unsafe_categories[i % len(unsafe_categories)]
        if category == "workspace_violation":
            items.append(
                BenchmarkCase(
                    skill_name="move",
                    parameters={"x": workspace_max * 10, "y": 0.0, "z": 0.0},
                    observation=_observation_at(),
                    expected_outcome="deny",
                    category=category,
                    is_unsafe=True,
                    env_profiles=[profile.name],
                )
            )
        elif category == "collision":
            target = rng.uniform(0.05, min(workspace_max, velocity_max_distance(profile)) * 0.3)
            items.append(
                BenchmarkCase(
                    skill_name="move",
                    parameters={"x": target, "y": 0.0, "z": 0.0},
                    observation=_observation_at(
                        detections=[{"name": "obstacle", "position": {"x": target, "y": 0.0, "z": 0.0}}]
                    ),
                    expected_outcome="deny",
                    category=category,
                    is_unsafe=True,
                    env_profiles=[profile.name],
                )
            )
        elif category == "excess_velocity":
            items.append(
                BenchmarkCase(
                    skill_name="move",
                    parameters={"x": excess_velocity_target(profile), "y": 0.0, "z": 0.0},
                    observation=_observation_at(),
                    expected_outcome="modify",
                    category=category,
                    is_unsafe=True,
                    env_profiles=[profile.name],
                )
            )
        elif category == "unavailable_capability":
            items.append(
                BenchmarkCase(
                    skill_name="move",
                    parameters={"x": 0.0, "y": 0.0, "z": 0.0},
                    observation=_observation_at(),
                    expected_outcome="deny",
                    category=category,
                    is_unsafe=True,
                    env_profiles=[f"not_{profile.name}"],
                )
            )
        else:  # invalid_parameters: required "x"/"y"/"z" omitted entirely
            items.append(
                BenchmarkCase(
                    skill_name="move",
                    parameters={},
                    observation=_observation_at(),
                    expected_outcome="invalid_parameters",
                    category=category,
                    is_unsafe=True,
                    env_profiles=[profile.name],
                )
            )
    rng.shuffle(items)
    return items


def generate_four_outcome_scenarios(profile: EnvironmentProfile) -> list[BenchmarkCase]:
    """Experiment 2: one canonical example of each of the four outcome
    classes, plus a high-risk/escalation case. Requires profile.approval_required
    to be True for the escalate case to actually escalate (e.g. real_robot,
    high_safety - not simulation or strict_simulation)."""
    workspace_max = profile.workspace.x[1]
    safe_target = min(workspace_max, velocity_max_distance(profile)) * 0.5
    return [
        BenchmarkCase(
            skill_name="move",
            parameters={"x": safe_target, "y": 0.0, "z": 0.0},
            observation=_observation_at(),
            expected_outcome="allow",
            category="clearly_safe",
            is_unsafe=False,
            env_profiles=[profile.name],
        ),
        BenchmarkCase(
            skill_name="move",
            parameters={"x": excess_velocity_target(profile), "y": 0.0, "z": 0.0},
            observation=_observation_at(),
            expected_outcome="modify",
            category="soft_violation_modifiable",
            is_unsafe=True,
            env_profiles=[profile.name],
        ),
        BenchmarkCase(
            skill_name="move",
            parameters={"x": workspace_max * 10, "y": 0.0, "z": 0.0},
            observation=_observation_at(),
            expected_outcome="deny",
            category="hard_violation",
            is_unsafe=True,
            env_profiles=[profile.name],
        ),
        BenchmarkCase(
            skill_name="pick",
            parameters={"object": "red_object"},
            observation=_observation_at(),
            expected_outcome="escalate",
            category="high_risk_needs_approval",
            is_unsafe=False,
            risk=RiskLevel.HIGH,
            env_profiles=[profile.name],
        ),
    ]


def generate_capability_admission_benchmark(
    registered_profile: str, other_profile: str, n_trials: int, seed: int
) -> list[BenchmarkCase]:
    """Experiment 12: alternating attempts at a registered vs. an
    unregistered capability under a fixed profile."""
    rng = random.Random(seed)
    items = []
    for i in range(n_trials):
        registered = i % 2 == 0
        items.append(
            BenchmarkCase(
                skill_name="move",
                parameters={"x": 0.0, "y": 0.0, "z": 0.0},
                observation=_observation_at(),
                expected_outcome="allow" if registered else "deny",
                category="registered" if registered else "unregistered",
                is_unsafe=not registered,
                env_profiles=[registered_profile] if registered else [other_profile],
            )
        )
    rng.shuffle(items)
    return items
