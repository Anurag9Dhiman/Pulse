#!/bin/bash
# Entrypoint for Experiment 17's live ROS 2 container. The repo is bind-mounted
# at /workspace at `docker run` time (not COPYed into the image) so iterating
# on par_driver.py never requires a rebuild - only pydantic/pyyaml/python-dotenv
# need installing here, and that's fast enough to redo on every container start.
set -eo pipefail
source /opt/ros/jazzy/setup.bash  # not nounset-safe; -u stays off for this line and beyond

pip install --break-system-packages --quiet -e /workspace

TRACE_PATH=/tmp/par_ros2_action_trace.jsonl
RESULT_PATH=/workspace/experiments/results/exp17_ros2_integration.json

python3 /workspace/experiments/ros2_live/mock_ros2_robot.py "$TRACE_PATH" &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT

sleep 2  # let the mock node's publishers/subscriptions come up before PAR connects

python3 /workspace/experiments/ros2_live/par_driver.py "$TRACE_PATH" "$RESULT_PATH"
