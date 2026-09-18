from __future__ import annotations

import json
from typing import Any, Protocol

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner, PlannerError

_SYSTEM_PROMPT = (
    "You are the planning component of a physical robot agent. At each step "
    "you receive the current goal and the robot's latest observation, and you "
    "must call exactly one tool: either one of the robot's available skills, "
    "or `task_complete` once the goal has been fully achieved. Never describe "
    "an action in text instead of calling its tool."
)

_TASK_COMPLETE_TOOL: dict[str, Any] = {
    "name": TASK_COMPLETE,
    "description": "Call this once the goal has been fully achieved and no further actions are needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Brief summary of what was accomplished."}
        },
        "required": ["message"],
    },
}

_TYPE_MAP = {"str": "string", "float": "number", "int": "integer", "bool": "boolean"}


class AnthropicClient(Protocol):
    messages: Any


def _capability_to_tool(capability: Capability) -> dict[str, Any]:
    properties = {
        name: {"type": _TYPE_MAP.get(param_type, "string")}
        for name, param_type in capability.input_schema.items()
    }
    return {
        "name": capability.name,
        "description": capability.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        },
    }


def _describe_observation(goal: str, observation: Observation) -> str:
    return (
        f"Goal: {goal}\n"
        f"Robot state: {json.dumps(observation.robot_state)}\n"
        f"Detected objects: {json.dumps(observation.detections)}"
    )


class LLMPlanner(Planner):
    """ReAct-style planner backed by an Anthropic tool-use loop.

    Accepts any client exposing `.messages.create(...)` in Anthropic's shape,
    so tests inject a fake client instead of calling the real API. Each
    propose() call sends the prior action's tool_result (if any) plus a fresh
    text description of the current observation, matching the
    observe-think-act-observe pattern validated for embodied tool use.
    """

    def __init__(
        self,
        client: AnthropicClient,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._history: list[dict[str, Any]] = []
        self._pending_tool_result: dict[str, Any] | None = None
        self._pending_tool_use_id: str | None = None

    @classmethod
    def from_api_key(cls, api_key: str | None = None, **kwargs: Any) -> "LLMPlanner":
        import anthropic

        return cls(anthropic.Anthropic(api_key=api_key), **kwargs)

    def reset(self) -> None:
        self._history = []
        self._pending_tool_result = None
        self._pending_tool_use_id = None

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        tools = [_capability_to_tool(c) for c in capabilities] + [_TASK_COMPLETE_TOOL]

        content_blocks: list[dict[str, Any]] = []
        if self._pending_tool_result is not None:
            content_blocks.append(self._pending_tool_result)
            self._pending_tool_result = None
        content_blocks.append({"type": "text", "text": _describe_observation(goal, observation)})
        self._history.append({"role": "user", "content": content_blocks})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            tools=tools,
            messages=self._history,
        )
        self._history.append({"role": "assistant", "content": response.content})

        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            raise PlannerError("model response did not include a tool call")

        self._pending_tool_use_id = tool_use.id
        return tool_use.name, dict(tool_use.input)

    def record_result(self, action: Action, result: ActionResult) -> None:
        if self._pending_tool_use_id is None:
            return
        self._pending_tool_result = {
            "type": "tool_result",
            "tool_use_id": self._pending_tool_use_id,
            "content": result.message or ("success" if result.success else "failed"),
            "is_error": not result.success,
        }
        self._pending_tool_use_id = None
