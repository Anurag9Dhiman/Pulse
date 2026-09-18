from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

import yaml
from pydantic import BaseModel


class Workspace(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def contains(self, x: float, y: float, z: float) -> bool:
        return self.x[0] <= x <= self.x[1] and self.y[0] <= y <= self.y[1] and self.z[0] <= z <= self.z[1]


class EnvironmentProfile(BaseModel):
    name: str
    workspace: Workspace
    max_velocity: float
    action_timeout_seconds: float
    approval_required: bool = False


def load_profile(name: str) -> EnvironmentProfile:
    """Loads a built-in profile (currently "simulation" or "real_robot")."""
    resource = resources.files("par.safety.profiles").joinpath(f"{name}.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return EnvironmentProfile(**data)


def load_profile_from_path(path: str | Path) -> EnvironmentProfile:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return EnvironmentProfile(**data)
