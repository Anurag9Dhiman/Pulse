"""Experiment 17: ROS 2 Integration.

Does PAR correctly preserve its governance boundary when connected to a real
ROS 2 execution interface?

No ROS 2 installation exists in this environment (confirmed: `import rclpy`
fails, `which ros2` finds nothing - see par doctor). This script is written
to be genuinely runnable once ROS 2 Jazzy/Humble is installed and sourced,
and it runs the one part that IS testable without ROS 2 today: the pure
message-mapping functions (ros2_mapping.py) that ROS2Robot depends on. The
pub/sub node lifecycle, Admission/Policy Guard behavior over a real ROS 2
topic, and the "no prohibited action reaches the robot topic" check are
left as documented, unrun steps - not faked with invented numbers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from par.core.action import Action
from par.robots.ros2_mapping import action_to_payload, payload_to_observation

try:
    import rclpy

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def _check_mapping_logic() -> dict:
    """The one part of this experiment that's actually runnable here."""
    obs = payload_to_observation({"robot_state": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}, "detections": [{"name": "obstacle"}]})
    action = Action(action_id="a1", skill_name="move", parameters={"x": 1.0, "y": 0.0, "z": 0.0}, created_at=datetime.now(timezone.utc))
    payload = action_to_payload(action)
    return {
        "payload_to_observation_roundtrip_ok": obs.robot_state == {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}
        and obs.detections == [{"name": "obstacle"}],
        "action_to_payload_roundtrip_ok": payload
        == {"action_id": "a1", "skill_name": "move", "parameters": {"x": 1.0, "y": 0.0, "z": 0.0}},
    }


def run() -> dict:
    mapping_check = _check_mapping_logic()
    return {
        "experiment": "17_ros2_integration",
        "rclpy_available_in_this_environment": ROS2_AVAILABLE,
        "mapping_logic_check": mapping_check,
        "mapping_logic_verified": all(mapping_check.values()),
        "steps_requiring_real_ros2": {
            "observation_acquisition": "unverified - needs a running ROS 2 node publishing to /par/robot_state, /par/detections",
            "capability_invocation": "unverified - covered structurally by par.core (Experiments 1-14), untested over the real ROS 2 transport",
            "safety_kernel_evaluation": "unverified over ROS 2 specifically - SafetyKernel itself is fully verified (Experiments 1-14) independent of transport",
            "action_publication": "unverified - needs a running ROS 2 node subscribing to /par/action",
            "action_result_acquisition": "unverified - ROS2Robot.execute() currently fire-and-forget (see module docstring in ros2_adapter.py); no completion-acknowledgement channel exists yet",
            "timeout_handling": "unverified over real ROS 2 latency - timeout mechanism itself is verified (test_action_timeout_produces_failed_result) independent of transport",
            "emergency_stop_behavior": "unverified over ROS 2 specifically - mechanism itself is verified (Experiment 8, test_emergency_stop_blocks_all_actions) independent of transport",
        },
        "how_to_actually_run_this_experiment": (
            "1) Install and source ROS 2 Jazzy or Humble. 2) Run the par_bridge "
            "extern-controller pattern (see par.integrations.webots for the "
            "equivalent Webots pattern) or a minimal ROS 2 node publishing "
            "mock state/detections and echoing published actions as if executed. "
            "3) Construct Runtime(agent, ROS2Robot(), safety_kernel=SafetyKernel(...)) "
            "and run the same benchmark generators from Experiment 1 through it. "
            "4) Record full ROS 2 topic traces (ros2 bag record) for each trial "
            "to verify no prohibited action was ever published to the command "
            "topic. This script provides the mapping-logic half; the rest needs "
            "a machine with ROS 2 installed, which this environment does not have."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp17_ros2_integration.json"
    out_path.write_text(json.dumps(result, indent=2))
