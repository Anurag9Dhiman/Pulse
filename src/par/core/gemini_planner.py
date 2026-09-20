from __future__ import annotations

import json
from typing import Any, Protocol

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, Planner, PlannerError

_SYSTEM_INSTRUCTION = (
    "You are the planning component of a physical robot agent. At each step "
    "you receive the current goal and the robot's latest observation, and you "
    "must call exactly one function: either one of the robot's available "
    "skills, or `task_complete` once the goal has been fully achieved. Never "
    "describe an action in text instead of calling its function."
)

_TASK_COMPLETE_DECLARATION: dict[str, Any] = {
    "name": TASK_COMPLETE,
    "description": "Call this once the goal has been fully achieved and no further actions are needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Brief summary of what was accomplished."}
        },
        "required": ["message"],
    },
}

_TYPE_MAP = {"str": "string", "float": "number", "int": "integer", "bool": "boolean"}

# Flash-Lite: ~0.8s per call in live testing vs ~2.4s for gemini-3.6-flash, which
# matters in a per-step robot loop. Model names retire - gemini-2.5-flash started
# returning 404 "no longer available to new users" - so pass model= to override.
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiChat(Protocol):
    def send_message(self, message: Any) -> Any: ...


def _capability_to_declaration(capability: Capability) -> dict[str, Any]:
    properties = {
        name: {"type": _TYPE_MAP.get(param_type, "string")}
        for name, param_type in capability.input_schema.items()
    }
    return {
        "name": capability.name,
        "description": capability.description,
        "parameters": {
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


class GeminiPlanner(Planner):
    """ReAct-style planner backed by Gemini function calling.

    Accepts any client exposing `.chats.create(...)` -> object with
    `.send_message(...)`, matching google-genai's shape, so tests inject a
    fake client instead of calling the real API. The chat config is passed
    as a plain dict (the SDK coerces it), but message parts MUST be real
    `google.genai.types.Part` objects: the SDK's send_message() does a
    strict type check and rejects plain dicts at runtime despite its type
    hints advertising PartDict. Importing this module doesn't require
    google-genai; calling propose() does (a real client needs it anyway).

    Automatic function calling is disabled: PAR's own Runtime/SafetyKernel
    decide whether a proposed call actually executes, not the SDK.
    """

    def __init__(self, client: Any, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model
        self._chat: GeminiChat | None = None
        self._capability_signature: tuple[str, ...] | None = None
        self._pending_function_response: tuple[str, dict[str, Any]] | None = None

    @classmethod
    def from_api_key(cls, api_key: str | None = None, **kwargs: Any) -> "GeminiPlanner":
        from google import genai

        return cls(genai.Client(api_key=api_key), **kwargs)

    def reset(self) -> None:
        self._chat = None
        self._capability_signature = None
        self._pending_function_response = None

    def propose(
        self, goal: str, observation: Observation, capabilities: list[Capability]
    ) -> tuple[str, dict[str, Any]]:
        from google.genai import types

        signature = tuple(c.name for c in capabilities)
        if self._chat is None or signature != self._capability_signature:
            declarations = [_capability_to_declaration(c) for c in capabilities] + [_TASK_COMPLETE_DECLARATION]
            config = {
                "tools": [{"function_declarations": declarations}],
                "system_instruction": _SYSTEM_INSTRUCTION,
                "automatic_function_calling": {"disable": True},
            }
            self._chat = self._client.chats.create(model=self._model, config=config)
            self._capability_signature = signature

        parts: list[Any] = []
        if self._pending_function_response is not None:
            name, response_data = self._pending_function_response
            parts.append(types.Part.from_function_response(name=name, response=response_data))
            self._pending_function_response = None
        parts.append(types.Part.from_text(text=_describe_observation(goal, observation)))

        response = self._chat.send_message(parts)
        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls:
            raise PlannerError("model response did not include a function call")

        call = function_calls[0]
        return call.name, dict(call.args)

    def record_result(self, action: Action, result: ActionResult) -> None:
        message = result.message or ("success" if result.success else "failed")
        self._pending_function_response = (action.skill_name, {"result": message, "success": result.success})
