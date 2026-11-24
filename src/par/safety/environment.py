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
    collision_margin: float = 0.2
    # --- perception trust policy ---------------------------------------
    # Both default off, and deliberately so: they express how much a
    # deployment trusts its sensors, which is not a property PAR can
    # choose on its operator's behalf. The failures that are unambiguous
    # bugs - a malformed position, a detection the kernel cannot place, an
    # adapter declaring its own sensor dead - are refused unconditionally
    # by par.core.perception and need no flag.
    #
    # Oldest reading the kernel will reason about, in seconds. None
    # disables the check: there is no defensible universal value, since a
    # 1 Hz lidar and a 1 kHz joint encoder differ by three orders of
    # magnitude.
    observation_max_age_seconds: float | None = None
    # When true, an adapter that declares nothing about a channel's health
    # is treated as unknown rather than healthy, and motion depending on
    # that channel is denied. Off by default so adapters written before
    # the contract keep working; on, it closes the last fail-open path -
    # an adapter that simply forgot to report.
    require_declared_perception: bool = False


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
