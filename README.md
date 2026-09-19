# Pulse — Physical Agent Runtime FRAMEWORK

A hardware-independent runtime that sits between AI reasoning and physical robot
execution: `Observation → Agent → Skill → Safety → Action`.

Includes the core runtime classes, a `RuleBasedPlanner` stand-in, an
`LLMPlanner` (Anthropic tool-use, ReAct-style loop), a mock robot, and a ROS 2
adapter skeleton — so the full loop can be tested without hardware, and swapped
to real ROS 2 / a real LLM once those are available.

## Quickstart

```bash
pip install -e ".[dev]"
par demo                           # live end-to-end demo: golden path + obstacle + re-plan
python examples/basic_loop.py      # RuleBasedPlanner + mock robot, no API key needed
python examples/llm_loop.py        # LLMPlanner + mock robot, requires ANTHROPIC_API_KEY
python examples/safety_demo.py     # Safety Kernel rejecting an out-of-bounds move
python examples/computer_use_loop.py  # delegate a step to CollectiveOS, see below
par doctor
pytest
```

`par demo` is the one to run first: it exercises the full
`Observation → Agent → Skill → Safety → Action` loop live, using a scripted
planner (stands in for the LLM, same way `RuleBasedPlanner` does elsewhere) so
it needs no API key, no ROS 2, and no hardware. It walks through the golden
path (pick, place), then introduces an obstacle mid-task and shows the Safety
Kernel rejecting the planned move and the loop re-planning around it — Use
Case 4 from the spec, running for real, not just described.

## Status

Tracking the 4-week MVP plan. Currently: **Week 4 — Demo Hardening**.

- [x] Core classes: `Observation`, `Action`, `Capability`, `Skill`, `Agent`,
      `Runtime`, `AgentState`
- [x] Pydantic validation
- [x] Capability / Skill registry
- [x] Mock robot
- [x] Structured telemetry logging
- [x] LLM planner (`par[llm]` extra; ReAct-style tool-use loop, `run_task()`
      loops until the model signals `task_complete`)
- [x] ROS 2 adapter — JSON-over-topic mapping is unit-tested; the pub/sub node
      lifecycle itself needs verification on a machine with ROS 2 Jazzy/Humble
      installed (not available in this dev environment)
- [x] Safety Kernel — two-stage Admission + Policy Guard, 4-outcome check
      (allow/modify/deny/escalate), workspace + velocity + collision-margin
      constraints, environment profiles (`simulation` / `real_robot` YAML),
      emergency stop, human-override escalation (safe-default deny), action
      timeouts. A denial feeds back to the planner and the task loop keeps
      going - re-planning, not aborting.
- [x] Basic world-state aggregate (`Runtime.world_state`) for future dashboards
- [x] Camera interface (`par.sensors`) + `MockCamera` — real `USBCamera`
      (`par[camera]` extra) is unverified: no camera hardware in this
      environment
- [x] `par demo` — one repeatable, live, hardware-free demonstration of the
      full loop plus environmental-change re-planning (the Week 4 milestone)
- [ ] Physical robot connection — covered by Week 2's `ROS2Robot`, pending
      actual hardware to point it at (see "Moving to Real Hardware" below)
- [x] Computer-use bridge — `use_computer` capability
      (`par.skills.computer_use`) + `ComputerAugmentedRobot`
      (`par.robots.computer_bridge`, `par[computer]` extra) delegate a
      computer-shaped step of a physical task to CollectiveOS's Navigation
      Agent over its `/robot/ws` endpoint, and return the result to PAR's
      loop. `risk=HIGH`, so it only escalates for human approval under the
      `real_robot` profile (`approval_required: true`) — CollectiveOS's own
      `/robot/ws` path has no HITL of its own, so this is the compensating
      gate. Requires `COLLECTIVEOS_WS_URL` and `COLLECTIVEOS_API_TOKEN`; see
      `examples/computer_use_loop.py`.

## Moving to Real Hardware

Everything above is verified in this sandbox with `MockRobot` and, where
applicable, a fake LLM client. Three things are built but genuinely
**unverified** here because the hardware/services don't exist in this
environment — this is what's left before Week 4 is fully done in reality,
not just in code:

| Component | What's verified here | What you need to do |
|---|---|---|
| **LLM planner** | Tool-use protocol logic, against a fake client | `pip install -e ".[llm]"`, set `ANTHROPIC_API_KEY`, run `python examples/llm_loop.py`. `par doctor` will show `LLM Planner ✓` once the key is set. |
| **ROS 2 adapter** | JSON message-mapping functions only | Install ROS 2 Jazzy or Humble, source its `setup.bash`. `par doctor` will show `ROS 2 ✓` once `rclpy` is importable. Then point `ROS2Robot`'s topic names at your robot's actual state/detection/command topics (real or simulated via Gazebo/Isaac Sim - same adapter serves both, per the spec's sim-to-real portability goal). |
| **Camera** | `CameraSource` interface + `MockCamera` | `pip install -e ".[camera]"`, attach a USB camera, use `USBCamera(device_index=...)`. `par doctor` will show `Camera ✓` (package-installed) once opencv is present; actual hardware capture still needs to be checked manually - a static check can't confirm a camera is physically attached and pointed at the workspace. |

Once all three are in place, the same `Agent`/`Skill`/`SafetyKernel` code runs
unchanged - swap the pieces `par demo` uses:

```python
from par.core.llm_planner import LLMPlanner
from par.robots.ros2_adapter import ROS2Robot
from par.safety.environment import load_profile

agent = Agent(registry, planner=LLMPlanner.from_api_key())
robot = ROS2Robot(state_topic="/your_robot/state", command_topic="/your_robot/cmd")
safety = SafetyKernel(load_profile("real_robot"))  # tighter limits than simulation
runtime = Runtime(agent, robot, safety_kernel=safety)
```

That is the whole migration: no other code changes, matching the spec's core
architectural principle that everything above the hardware adapter stays
hardware-independent.
