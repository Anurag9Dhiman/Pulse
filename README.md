# Pulse — Physical Agent Runtime (PAR)

A hardware-independent runtime that sits between AI reasoning and physical robot
execution: `Observation → Agent → Skill → Safety → Action`.

This is the Week 1 milestone build: the core runtime classes, a rule-based
planner stand-in for the Week 2 LLM integration, and a mock robot so the full
loop can be tested without any hardware or ROS 2 dependency.

## Quickstart

```bash
pip install -e ".[dev]"
python examples/basic_loop.py
par doctor
pytest
```

## Status

Tracking the 4-week MVP plan. Currently: **Week 1 — Runtime Foundation**.

- [x] Core classes: `Observation`, `Action`, `Capability`, `Skill`, `Agent`,
      `Runtime`, `AgentState`
- [x] Pydantic validation
- [x] Capability / Skill registry
- [x] Mock robot
- [x] Structured telemetry logging
- [ ] ROS 2 adapter (Week 2)
- [ ] LLM/VLM planner (Week 2)
- [ ] Safety Kernel (Week 3)
- [ ] Physical robot (Week 3)
