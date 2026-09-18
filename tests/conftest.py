from __future__ import annotations

import pytest

from par.core.agent import Agent
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.skills import builtin_skills


@pytest.fixture
def skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


@pytest.fixture
def robot() -> MockRobot:
    return MockRobot()


@pytest.fixture
def agent(skill_registry: SkillRegistry) -> Agent:
    return Agent(skill_registry)
