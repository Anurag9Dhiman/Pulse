"""Week 1 milestone: Observation -> Agent -> Skill -> Action -> Mock Robot -> Result."""

from par.core.agent import Agent
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.skills import builtin_skills


def main() -> None:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)

    robot = MockRobot()
    agent = Agent(registry)
    runtime = Runtime(agent, robot)

    for task in ("detect objects", "pick the red_object", "place at blue_container"):
        result = runtime.run_once(task)
        print(f"{task!r:35} -> success={result.success}  {result.message}")


if __name__ == "__main__":
    main()
