# Pulse — Physical Agent Runtime (PAR)

A hardware-independent runtime that sits between AI reasoning and physical robot
execution: `Observation → Agent → Skill → Safety → Action`.

Includes the core runtime classes, a `RuleBasedPlanner` stand-in, an
`LLMPlanner` (Anthropic tool-use, ReAct-style loop), a mock robot, and a ROS 2
adapter skeleton — so the full loop can be tested without hardware, and swapped
to real ROS 2 / a real LLM once those are available.

## Quickstart

```bash
pip install -e ".[dev]"
python examples/basic_loop.py      # RuleBasedPlanner + mock robot, no API key needed
python examples/llm_loop.py        # LLMPlanner + mock robot, requires ANTHROPIC_API_KEY
python examples/safety_demo.py     # Safety Kernel rejecting an out-of-bounds move
par doctor
pytest
```

## Status

Tracking the 4-week MVP plan. Currently: **Week 3 — Safety & Closed-Loop
Physical Execution**.

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
      (allow/modify/deny/escalate), workspace + velocity constraints,
      environment profiles (`simulation` / `real_robot` YAML), emergency stop,
      human-override escalation (safe-default deny), action timeouts. A
      denial feeds back to the planner and the task loop keeps going -
      re-planning, not aborting.
- [x] Basic world-state aggregate (`Runtime.world_state`) for future dashboards
- [x] Camera interface (`par.sensors`) + `MockCamera` — real `USBCamera`
      (`par[camera]` extra) is unverified: no camera hardware in this
      environment
- [ ] Physical robot connection — covered by Week 2's `ROS2Robot`, pending
      actual hardware to point it at
- [ ] Investor demo hardening (Week 4)
