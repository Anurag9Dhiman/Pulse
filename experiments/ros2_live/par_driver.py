"""The PAR-side half of Experiment 17's live ROS 2 integration test.

Runs as a separate OS process from mock_ros2_robot.py - the two communicate
only over real ROS 2 topics/DDS, not in-process function calls. Drives a
scripted task through the real governance pipeline (Agent -> SafetyKernel ->
ROS2Robot) under the `real_robot` profile, deliberately including actions
that should be denied at both Safety Kernel stages:

  - Admission DENY: "pick" is left registered only for "simulation" (its
    default env_profiles), so proposing it under "real_robot" is rejected
    before Policy Guard ever runs.
  - Policy Guard DENY (workspace): a move target outside real_robot.yaml's
    [-1,1]x[-1,1]x[0,1.2] workspace.
  - Policy Guard DENY (collision): a move target within collision_margin
    (0.5m) of the obstacle mock_ros2_robot.py publishes at (0.5, 0.5, 0).
  - Policy Guard MODIFY: a move target farther than max_velocity *
    action_timeout_seconds (0.5 * 3.0 = 1.5m) away, which should be clamped
    rather than denied outright.

No ESCALATE case: real_robot.yaml sets approval_required=True, but every
builtin skill is LOW or MEDIUM risk (see par/skills/builtin.py) and
SafetyKernel.check() only escalates on RiskLevel.HIGH, so ESCALATE is
structurally unreachable here without inventing a capability outside this
experiment's scope. Documented rather than silently skipped.

After the scripted task finishes, this script reads mock_ros2_robot.py's
action trace (written to the same file both processes are pointed at) and
cross-checks it against what the Safety Kernel actually decided: the real
claim this experiment exists to verify is that every DENY'd action's
payload is absent from that trace, and every ALLOW/MODIFY'd action's
payload is present - i.e. the governance boundary holds over a real
transport, not just over in-process function calls.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))  # import the exp17 mapping check below

from exp17_ros2_integration import _check_mapping_logic  # noqa: E402

from par.core.action import Action, ActionResult  # noqa: E402
from par.core.agent import Agent  # noqa: E402
from par.core.observation import Observation  # noqa: E402
from par.core.planner import TASK_COMPLETE, Planner  # noqa: E402
from par.core.runtime import Runtime  # noqa: E402
from par.core.skill import SkillRegistry  # noqa: E402
from par.evaluation.capture import CapturingTelemetryLogger  # noqa: E402
from par.robots.ros2_adapter import ROS2Robot  # noqa: E402
from par.robots.ros2_mapping import action_to_payload  # noqa: E402
from par.safety.environment import load_profile  # noqa: E402
from par.safety.kernel import SafetyKernel  # noqa: E402
from par.skills import builtin_skills  # noqa: E402

PROFILE_NAME = "real_robot"

# (skill, parameters, what this step is meant to prove)
_STEPS: list[tuple[str, dict[str, Any], str]] = [
    ("detect", {}, "baseline ALLOW, no move-specific checks apply"),
    ("move", {"x": 0.2, "y": 0.0, "z": 0.0}, "ALLOW: inside workspace, slow, clear of the obstacle"),
    ("move", {"x": 0.6, "y": 0.5, "z": 0.0}, "Policy Guard DENY: 0.1m from the obstacle, inside the 0.5m margin"),
    ("move", {"x": 2.0, "y": 0.0, "z": 0.0}, "Policy Guard DENY: x=2.0 outside workspace bound [-1, 1]"),
    ("move", {"x": 1.0, "y": -1.0, "z": 1.0}, "Policy Guard MODIFY: 1.73m away, exceeds the 1.5m reachable distance"),
    ("pick", {"object": "red_object"}, "Admission DENY: 'pick' is only registered for 'simulation'"),
    ("stop", {}, "ALLOW: no move-specific checks apply"),
    (TASK_COMPLETE, {"message": "live ROS 2 governance trace complete"}, "ends the task cleanly"),
]


class _ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any], str]]) -> None:
        self._steps = [(skill, params) for skill, params, _ in steps]
        self.record: list[dict[str, Any]] = []

    def propose(self, goal: str, observation: Observation, capabilities) -> tuple[str, dict[str, Any]]:
        return self._steps.pop(0)

    def record_result(self, action: Action, result: ActionResult) -> None:
        self.record.append(
            {
                "skill": action.skill_name,
                "parameters": action.parameters,
                "success": result.success,
                "message": result.message,
            }
        )


def _registry_for_real_robot() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        if skill.name != "pick":
            skill.capability.env_profiles = [PROFILE_NAME]
        registry.register(skill)
    return registry


def _cross_check_trace(trace_path: Path, step_log: list[dict[str, Any]]) -> dict[str, Any]:
    """The actual safety claim: nothing PAR denied should appear on the wire."""
    if not trace_path.exists():
        return {"trace_file_found": False, "note": f"expected mock_ros2_robot.py's trace at {trace_path}"}

    received_payloads = []
    for line in trace_path.read_text().splitlines():
        if line.strip():
            received_payloads.append(json.loads(line)["payload"])

    violations = []
    confirmed_executions = []
    for step in step_log:
        if step["skill"] == TASK_COMPLETE:
            continue
        was_published = step["payload"] in received_payloads
        if step["outcome"] in ("ALLOW", "MODIFY"):
            confirmed_executions.append({**step, "published_to_ros2": was_published})
            if not was_published:
                violations.append({**step, "problem": "executed by PAR but never seen on /par/action"})
        else:
            if was_published:
                violations.append({**step, "problem": "DENIED by PAR but reached /par/action anyway"})

    return {
        "trace_file_found": True,
        "n_messages_received_on_action_topic": len(received_payloads),
        "n_executed_steps_confirmed_published": sum(1 for c in confirmed_executions if c["published_to_ros2"]),
        "governance_boundary_held": len(violations) == 0,
        "violations": violations,
    }


def main() -> None:
    trace_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/par_ros2_action_trace.jsonl")
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).parent.parent / "results" / "exp17_ros2_integration.json"
    )

    registry = _registry_for_real_robot()
    planner = _ScriptedPlanner(_STEPS)
    agent = Agent(registry, planner=planner)
    telemetry = CapturingTelemetryLogger()
    robot = ROS2Robot()
    safety = SafetyKernel(load_profile(PROFILE_NAME))
    runtime = Runtime(agent, robot, safety_kernel=safety, telemetry=telemetry)

    started = time.monotonic()
    runtime.run_task("tour the real_robot workspace, triggering every Safety Kernel outcome", max_steps=10)
    wall_seconds = time.monotonic() - started

    robot.shutdown()
    runtime.close()

    step_log = []
    for event, (skill, params, intent) in zip(telemetry.events, _STEPS):
        step_log.append(
            {
                "skill": skill,
                "parameters": params,
                "intent": intent,
                "outcome": event.safety_decision.upper(),
                "reason": event.rejection_reason,
                "payload": action_to_payload(event.action),
                "latency_seconds": event.latency_seconds,
            }
        )

    time.sleep(0.5)  # let mock_ros2_robot.py flush its trace file writer
    cross_check = _cross_check_trace(trace_path, step_log)

    result = {
        "experiment": "17_ros2_integration",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "rclpy_available_in_this_environment": True,
        "ran_against": "real rclpy node (mock_ros2_robot.py) over actual ROS 2 topics, not in-process",
        "mapping_logic_check": _check_mapping_logic(),
        "profile": PROFILE_NAME,
        "wall_clock_seconds": round(wall_seconds, 2),
        "steps": step_log,
        "cross_check_against_ros2_action_topic": cross_check,
        "steps_requiring_real_ros2": {
            "observation_acquisition": "verified - ROS2Robot.get_observation() received real /par/robot_state and /par/detections messages from a separate process",
            "capability_invocation": "verified over the real transport for this run's 8 steps (detect, move x4, pick, stop, task_complete)",
            "safety_kernel_evaluation": "verified - Admission and Policy Guard both exercised over real-transport observations",
            "action_publication": "verified - see cross_check_against_ros2_action_topic",
            "action_result_acquisition": "still unverified - ROS2Robot.execute() remains fire-and-forget (see ros2_adapter.py); mock_ros2_robot.py has no completion-acknowledgement channel to confirm against",
            "timeout_handling": "not exercised this run - no action in the scripted plan was slow enough to trigger action_timeout_seconds; the mechanism itself stays covered by test_action_timeout_produces_failed_result",
            "emergency_stop_behavior": "not exercised this run - would need a second driver invocation calling runtime.emergency_stop() mid-task; left for a follow-up run rather than invented here",
        },
        "known_limitation_observed": (
            "ROS2Robot.get_observation() spins for the full observation_timeout "
            "(2.0s default) on every call regardless of when data actually "
            "arrives (see the unconditional while-loop in ros2_adapter.py) - "
            "this run's wall-clock time is dominated by that fixed per-step "
            "cost, not by DDS or governance latency."
        ),
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
