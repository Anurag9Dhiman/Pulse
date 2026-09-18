"""Week 2: the same loop as basic_loop.py, planned by an LLM instead of RuleBasedPlanner.

Requires `pip install -e ".[llm]"` and ANTHROPIC_API_KEY set in the environment.
"""

import os

from par.core.agent import Agent
from par.core.llm_planner import LLMPlanner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.skills import builtin_skills


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example against the real Claude API.")
        return

    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)

    agent = Agent(registry, planner=LLMPlanner.from_api_key())
    runtime = Runtime(agent, MockRobot())

    results = runtime.run_task("Pick up the red_object and place it in the blue_container.")
    for result in results:
        print(f"success={result.success}  {result.message}")


if __name__ == "__main__":
    main()
