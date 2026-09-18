import pytest

from par.core.skill import SkillError, SkillRegistry
from par.skills import builtin_skills


def test_registry_lists_all_builtin_skills(skill_registry: SkillRegistry):
    assert skill_registry.list_names() == sorted(
        {"detect", "move", "pick", "place", "stop", "inspect"}
    )


def test_duplicate_registration_rejected():
    registry = SkillRegistry()
    skills = builtin_skills()
    registry.register(skills[0])
    with pytest.raises(SkillError):
        registry.register(skills[0])


def test_unknown_skill_lookup_raises(skill_registry: SkillRegistry):
    with pytest.raises(SkillError):
        skill_registry.get("fly")


def test_missing_parameters_rejected(skill_registry: SkillRegistry):
    pick = skill_registry.get("pick")
    with pytest.raises(SkillError):
        pick.build_action({})


def test_valid_parameters_build_action(skill_registry: SkillRegistry):
    pick = skill_registry.get("pick")
    action = pick.build_action({"object": "red_object"})
    assert action.skill_name == "pick"
    assert action.parameters == {"object": "red_object"}
