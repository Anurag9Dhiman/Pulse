"""PAR delegating a computer-shaped step of a physical task to CollectiveOS.

Wires the `use_computer` capability (par.skills.computer_use) through a
ComputerAugmentedRobot into the normal Agent/Runtime loop, using the
real_robot safety profile so the HIGH-risk `use_computer` capability
escalates to a human-approval prompt before PAR ever dispatches to
CollectiveOS - CollectiveOS's own `/robot/ws` path has no HITL of its own,
so this is the one approval gate in the whole round trip.

Requires:
    pip install -e ".[llm,computer]"
    ANTHROPIC_API_KEY        - for the LLM planner
    COLLECTIVEOS_WS_URL      - e.g. ws://localhost:8000/robot/ws
    COLLECTIVEOS_API_TOKEN   - matches CollectiveOS's API_TOKEN

And CollectiveOS running and reachable:
    cd CollectiveOS && uvicorn src.api:app --port 8000
"""
import os

from par.core.action import Action
from par.core.agent import Agent
from par.core.llm_planner import LLMPlanner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.computer_bridge import ComputerAugmentedRobot
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills, computer_use_skill


def _prompt_for_approval(action: Action, reason: str) -> bool:
    print(f"\n[ESCALATION] {reason}")
    print(f"  skill={action.skill_name} parameters={action.parameters}")
    return input("  Approve? [y/N] ").strip().lower() == "y"


def main() -> None:
    missing = [
        var
        for var in ("ANTHROPIC_API_KEY", "COLLECTIVEOS_WS_URL", "COLLECTIVEOS_API_TOKEN")
        if not os.environ.get(var)
    ]
    if missing:
        print(f"Set the following env vars to run this example: {', '.join(missing)}")
        return

    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    registry.register(computer_use_skill())

    robot = ComputerAugmentedRobot(MockRobot())
    agent = Agent(registry, planner=LLMPlanner.from_api_key())
    safety = SafetyKernel(load_profile("real_robot"))
    runtime = Runtime(agent, robot, safety_kernel=safety, human_approval=_prompt_for_approval)

    goal = "Look up today's date, then move to the blue_container."
    results = runtime.run_task(goal, max_steps=10)
    for result in results:
        print(f"success={result.success}  {result.message}")


if __name__ == "__main__":
    main()
