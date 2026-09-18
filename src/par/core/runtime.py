from __future__ import annotations

import time

from par.core.action import ActionResult, ActionStatus
from par.core.agent import Agent
from par.core.agent_state import AgentStatus
from par.robots.base import RobotInterface
from par.telemetry.events import TelemetryEvent
from par.telemetry.logger import TelemetryLogger


class Runtime:
    def __init__(
        self,
        agent: Agent,
        robot: RobotInterface,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.agent = agent
        self.robot = robot
        self.telemetry = telemetry or TelemetryLogger()

    def run_once(self, goal: str) -> ActionResult:
        self.agent.set_goal(goal)
        started = time.monotonic()

        observation = self.robot.get_observation()
        action = self.agent.propose_action(observation)

        # No Safety Kernel yet (Week 3); every proposed action auto-approves for now.
        action.status = ActionStatus.APPROVED
        self.agent.state.status = AgentStatus.EXECUTING

        result = self.robot.execute(action)
        self.agent.record_result(action, result)

        self.telemetry.log(
            TelemetryEvent(
                task=goal,
                observation=observation,
                skill=action.skill_name,
                safety_decision="approved",
                action=action,
                result=result,
                latency_seconds=time.monotonic() - started,
                success=result.success,
            )
        )
        return result
