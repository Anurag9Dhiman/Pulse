"""Week 3: Use Case 3 - Safety Rejection (PAR spec, Section 4).

Demonstrates: Agent Action -> Safety Check -> REJECTED -> Reason Returned.
The planner below deliberately proposes an out-of-bounds move to trigger the
rejection; no ROS 2 or hardware required.
"""

from par.core.agent import Agent
from par.core.planner import Planner
from par.core.runtime import Runtime
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.safety.environment import load_profile
from par.safety.kernel import SafetyKernel
from par.skills import builtin_skills


class _UnsafeMovePlanner(Planner):
    """Deliberately proposes a move outside the workspace, to demonstrate rejection."""

    def propose(self, goal, observation, capabilities):
        return "move", {"x": 5.0, "y": 0.0, "z": 0.0}


def main() -> None:
    # Builtin skills default to env_profiles=["simulation"]; register them for
    # real_robot too so this demo exercises the workspace check, not admission.
    registry = SkillRegistry()
    for skill in builtin_skills():
        skill.capability.env_profiles = ["real_robot"]
        registry.register(skill)

    agent = Agent(registry, planner=_UnsafeMovePlanner())
    robot = MockRobot()
    safety = SafetyKernel(load_profile("real_robot"))  # workspace: x,y in [-1,1]
    runtime = Runtime(agent, robot, safety_kernel=safety)

    result = runtime.run_once("move far away")
    print(f"outcome: success={result.success}")
    print(f"reason:  {result.message}")
    runtime.close()


if __name__ == "__main__":
    main()
