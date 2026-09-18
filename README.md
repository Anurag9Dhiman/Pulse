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
par doctor
pytest
```

## Status

Tracking the 4-week MVP plan. Currently: **Week 2 — Agent, Skills & ROS 2**.

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
- [ ] Safety Kernel (Week 3)
- [ ] Physical robot + camera (Week 3)
