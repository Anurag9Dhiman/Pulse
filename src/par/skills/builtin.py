from __future__ import annotations

from par.core.capability import Capability, RiskLevel
from par.core.skill import ParameterizedSkill, Skill

_SPECS: list[tuple[str, str, RiskLevel, bool, set[str], dict[str, str]]] = [
    ("detect", "Detect objects in the current observation", RiskLevel.LOW, False, set(), {}),
    (
        "move",
        "Move the end effector or base to a target position",
        RiskLevel.MEDIUM,
        False,
        {"x", "y", "z"},
        {"x": "float", "y": "float", "z": "float"},
    ),
    ("pick", "Pick up a target object", RiskLevel.MEDIUM, True, {"object"}, {"object": "str"}),
    (
        "place",
        "Place the held object at a target location",
        RiskLevel.MEDIUM,
        True,
        {"target"},
        {"target": "str"},
    ),
    ("stop", "Immediately halt all motion", RiskLevel.LOW, False, set(), {}),
    (
        "inspect",
        "Closely examine a target to verify its state",
        RiskLevel.LOW,
        False,
        {"target"},
        {"target": "str"},
    ),
]


def builtin_skills() -> list[Skill]:
    skills = []
    for name, description, risk, supports_rollback, required, input_schema in _SPECS:
        capability = Capability(
            name=name,
            description=description,
            input_schema=input_schema,
            risk=risk,
            supports_rollback=supports_rollback,
        )
        skills.append(ParameterizedSkill(capability, required_params=required))
    return skills
