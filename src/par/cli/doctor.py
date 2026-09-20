from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from par.core.agent import Agent
from par.core.skill import SkillRegistry
from par.robots.mock import MockRobot
from par.skills import builtin_skills


@dataclass
class CheckResult:
    ok: bool
    detail: str


def _check_runtime_core() -> CheckResult:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return CheckResult(True, f"{len(registry.list_names())} skills registered")


def _check_mock_robot() -> CheckResult:
    robot = MockRobot()
    observation = robot.get_observation()
    return CheckResult(True, f"{len(observation.detections)} objects visible")


def _check_agent() -> CheckResult:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    Agent(registry)
    return CheckResult(True, "planner: rule-based (Week 1 stub)")


def _check_llm_planner() -> CheckResult:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return CheckResult(False, "anthropic package not installed (pip install par[llm])")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult(False, "ANTHROPIC_API_KEY not set")
    return CheckResult(True, "anthropic client ready")


def _check_gemini_planner() -> CheckResult:
    from par.core.gemini_planner import DEFAULT_MODEL

    try:
        import google.genai  # noqa: F401
    except ImportError:
        return CheckResult(False, "google-genai package not installed (pip install par[gemini])")
    if not os.environ.get("GEMINI_API_KEY"):
        return CheckResult(False, "GEMINI_API_KEY not set (env var or .env file)")
    return CheckResult(True, f"google-genai ready, default model: {DEFAULT_MODEL}")


def _check_ros2() -> CheckResult:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return CheckResult(False, "rclpy not installed (source ROS 2 Jazzy/Humble setup)")
    return CheckResult(True, "rclpy importable")


def _check_camera() -> CheckResult:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return CheckResult(False, "opencv not installed (pip install par[camera]); MockCamera available for dev")
    return CheckResult(True, "opencv importable (camera hardware not verified)")


def _check_safety_kernel() -> CheckResult:
    from par.safety.environment import load_profile
    from par.safety.kernel import SafetyKernel

    profile = load_profile("simulation")
    SafetyKernel(profile)
    return CheckResult(True, f"profile '{profile.name}' loaded")


CHECKS: list[tuple[str, Callable[[], CheckResult]]] = [
    ("Runtime Core", _check_runtime_core),
    ("Agent", _check_agent),
    ("Mock Robot", _check_mock_robot),
    ("LLM Planner (Anthropic)", _check_llm_planner),
    ("LLM Planner (Gemini)", _check_gemini_planner),
    ("ROS 2", _check_ros2),
    ("Camera", _check_camera),
    ("Safety Kernel", _check_safety_kernel),
]


def run() -> int:
    print("Physical Agent Runtime\n")
    all_ok = True
    for label, check in CHECKS:
        result = check()
        mark = "✓" if result.ok else "✗"
        print(f"{label:<24} {mark}  {result.detail}")
        all_ok = all_ok and result.ok
    print()
    print("READY" if all_ok else "NOT READY (see README 'Moving to Real Hardware' for what's missing)")
    return 0 if all_ok else 1
