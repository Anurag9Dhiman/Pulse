from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from par.core.action import Action
from par.core.capability import Capability


class SkillError(Exception):
    pass


class Skill(ABC):
    def __init__(self, capability: Capability) -> None:
        self.capability = capability

    @property
    def name(self) -> str:
        return self.capability.name

    @abstractmethod
    def validate_parameters(self, parameters: dict[str, Any]) -> None: ...

    @abstractmethod
    def build_action(self, parameters: dict[str, Any]) -> Action: ...


class ParameterizedSkill(Skill):
    """Covers the six MVP skills, which differ only in name and required parameters."""

    def __init__(self, capability: Capability, required_params: set[str]) -> None:
        super().__init__(capability)
        self._required_params = required_params

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        missing = self._required_params - parameters.keys()
        if missing:
            raise SkillError(f"{self.name} missing parameters: {sorted(missing)}")

    def build_action(self, parameters: dict[str, Any]) -> Action:
        self.validate_parameters(parameters)
        return Action(
            action_id=str(uuid4()),
            skill_name=self.name,
            parameters=parameters,
            created_at=datetime.now(timezone.utc),
        )


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise SkillError(f"skill '{skill.name}' already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillError(f"unknown skill '{name}'") from exc

    def list_names(self) -> list[str]:
        return sorted(self._skills)

    def capabilities(self) -> list[Capability]:
        return [skill.capability for skill in self._skills.values()]
