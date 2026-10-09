#!/bin/bash
# Launches just enough of the Reach Webots simulation for Experiment 18:
# Webots itself (headless CLI launch, no manual "press Run" needed) plus
# both extern-controller bridges - Webots holds the whole simulation paused
# at t=0 until BOTH connect, not just the e-puck's par_bridge.py, even
# though this experiment never exercises computer_arm_bridge's own actions.
# Deliberately skips Reach/scripts/run_simulation.sh: that script also
# requires CollectiveOS/.env and starts CollectiveOS itself, neither of
# which Experiment 18's 6 builtin-skill tasks need.
set -eo pipefail

REACH="/Users/anuragdhiman/Documents/Reach/Reach"
STATE_DIR="$(dirname "${BASH_SOURCE[0]}")/.sim_state"
mkdir -p "$STATE_DIR"

if [[ "${1:-}" == "--stop" ]]; then
    for name in webots par_bridge computer_arm_bridge; do
        pidfile="$STATE_DIR/$name.pid"
        [[ -f "$pidfile" ]] && kill -9 "$(cat "$pidfile")" 2>/dev/null
        rm -f "$pidfile"
    done
    pkill -9 -f "MacOS/webots.*par_arena.wbt" 2>/dev/null || true
    pkill -9 -f "par_bridge/par_bridge.py" 2>/dev/null || true
    pkill -9 -f "computer_arm_bridge/computer_arm_bridge.py" 2>/dev/null || true
    exit 0
fi

export WEBOTS_HOME=/Applications/Webots.app
export PYTHONPATH="$WEBOTS_HOME/Contents/lib/controller/python:${PYTHONPATH:-}"

PORT=$((20000 + RANDOM % 20000))
echo "starting Webots on port $PORT..."
nohup "$WEBOTS_HOME/Contents/MacOS/webots" --port="$PORT" --mode=realtime --stdout --stderr \
    "$REACH/webots/worlds/par_arena.wbt" > "$STATE_DIR/webots.log" 2>&1 &
echo $! > "$STATE_DIR/webots.pid"
sleep 6

echo "starting par_bridge.py..."
WEBOTS_CONTROLLER_URL="ipc://$PORT/epuck" nohup python3 \
    "$REACH/webots/controllers/par_bridge/par_bridge.py" > "$STATE_DIR/par_bridge.log" 2>&1 &
echo $! > "$STATE_DIR/par_bridge.pid"

echo "starting computer_arm_bridge.py..."
WEBOTS_CONTROLLER_URL="ipc://$PORT/computer_arm" nohup python3 \
    "$REACH/webots/controllers/computer_arm_bridge/computer_arm_bridge.py" > "$STATE_DIR/computer_arm_bridge.log" 2>&1 &
echo $! > "$STATE_DIR/computer_arm_bridge.pid"

echo "waiting for both bridges..."
for i in $(seq 1 30); do
    bridge_up=0; arm_up=0
    grep -q "listening on ws" "$STATE_DIR/par_bridge.log" 2>/dev/null && bridge_up=1
    grep -q "listening on ws" "$STATE_DIR/computer_arm_bridge.log" 2>/dev/null && arm_up=1
    [[ "$bridge_up" == 1 && "$arm_up" == 1 ]] && break
    sleep 2
done

if [[ "$bridge_up" != 1 || "$arm_up" != 1 ]]; then
    echo "FAILED to bring both bridges up - check $STATE_DIR/{webots,par_bridge,computer_arm_bridge}.log"
    exit 1
fi
echo "ready: ws://localhost:6001 (epuck), ws://localhost:6002 (computer_arm)"
