"""Same loop as basic_loop.py, planned by Gemini instead of RuleBasedPlanner.

Requires `pip install -e ".[gemini]"` and GEMINI_API_KEY, set in the
environment or in a project-root .env file (see .env.example).
"""

import os

from par.core.agent import Agent
from par.core.gemini_planner import GeminiPlanner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.env import load_env
from par.robots.mock import MockRobot
from par.skills import builtin_skills


def main() -> None:
    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this example against the real Gemini API.")
        return

    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)

    agent = Agent(registry, planner=GeminiPlanner.from_api_key())
    runtime = Runtime(agent, MockRobot())

    results = runtime.run_task("Pick up the red_object and place it in the blue_container.")
    for result in results:
        print(f"success={result.success}  {result.message}")


if __name__ == "__main__":
    main()
