from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.gemini_planner import GeminiPlanner
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, PlannerError


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class FakeGeminiResponse:
    function_calls: list[FakeFunctionCall] = field(default_factory=list)


class FakeChat:
    def __init__(self, responses: list[FakeGeminiResponse]) -> None:
        self._responses = list(responses)
        self.sent_messages: list[Any] = []

    def send_message(self, message: Any) -> FakeGeminiResponse:
        self.sent_messages.append(message)
        return self._responses.pop(0)


class FakeChats:
    def __init__(self, chat: FakeChat) -> None:
        self._chat = chat
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeChat:
        self.create_calls.append(kwargs)
        return self._chat


class FakeClient:
    def __init__(self, chat: FakeChat) -> None:
        self.chats = FakeChats(chat)


def _observation() -> Observation:
    return Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        detections=[{"name": "red_object"}],
    )


def _capabilities() -> list[Capability]:
    return [Capability(name="pick", description="pick up an object", input_schema={"object": "str"})]


def test_propose_returns_function_call_from_response():
    response = FakeGeminiResponse(function_calls=[FakeFunctionCall(name="pick", args={"object": "red_object"})])
    chat = FakeChat([response])
    planner = GeminiPlanner(FakeClient(chat))

    name, params = planner.propose("pick the red object", _observation(), _capabilities())

    assert name == "pick"
    assert params == {"object": "red_object"}


def test_chat_created_with_tools_including_task_complete():
    response = FakeGeminiResponse(function_calls=[FakeFunctionCall(name="pick", args={"object": "red_object"})])
    chat = FakeChat([response])
    client = FakeClient(chat)
    planner = GeminiPlanner(client)

    planner.propose("pick the red object", _observation(), _capabilities())

    config = client.chats.create_calls[0]["config"]
    declarations = config["tools"][0]["function_declarations"]
    names = {d["name"] for d in declarations}
    assert names == {"pick", TASK_COMPLETE}
    assert config["automatic_function_calling"]["disable"] is True


def test_missing_function_call_raises_planner_error():
    response = FakeGeminiResponse(function_calls=[])
    chat = FakeChat([response])
    planner = GeminiPlanner(FakeClient(chat))

    with pytest.raises(PlannerError):
        planner.propose("pick the red object", _observation(), _capabilities())


def test_record_result_feeds_back_into_next_message():
    first = FakeGeminiResponse(function_calls=[FakeFunctionCall(name="pick", args={"object": "red_object"})])
    second = FakeGeminiResponse(function_calls=[FakeFunctionCall(name=TASK_COMPLETE, args={"message": "done"})])
    chat = FakeChat([first, second])
    planner = GeminiPlanner(FakeClient(chat))

    planner.propose("pick the red object", _observation(), _capabilities())
    action = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "red_object"},
        created_at=datetime.now(timezone.utc),
    )
    result = ActionResult(action_id="a1", success=True, message="picked", completed_at=datetime.now(timezone.utc))
    planner.record_result(action, result)

    name, _params = planner.propose("pick the red object", _observation(), _capabilities())

    assert name == TASK_COMPLETE
    second_message = chat.sent_messages[1]
    function_response_parts = [p for p in second_message if p.function_response is not None]
    assert function_response_parts[0].function_response.name == "pick"
    assert function_response_parts[0].function_response.response["result"] == "picked"


def test_sent_message_parts_are_real_sdk_part_objects():
    # Regression: the real SDK's send_message() strictly type-checks and rejects
    # plain dicts at runtime. Fake-client tests can't catch that on their own,
    # so assert the type directly.
    types = pytest.importorskip("google.genai.types")
    response = FakeGeminiResponse(function_calls=[FakeFunctionCall(name="pick", args={"object": "red_object"})])
    chat = FakeChat([response])
    planner = GeminiPlanner(FakeClient(chat))

    planner.propose("pick the red object", _observation(), _capabilities())

    assert all(type(part) is types.Part for part in chat.sent_messages[0])


def test_reset_starts_a_new_chat():
    response = FakeGeminiResponse(function_calls=[FakeFunctionCall(name="pick", args={"object": "red_object"})])
    chat_a = FakeChat([response])
    chat_b = FakeChat([response])

    client = FakeClient(chat_a)
    planner = GeminiPlanner(client)
    planner.propose("pick the red object", _observation(), _capabilities())

    client.chats._chat = chat_b  # simulate a fresh chat being returned on the next create()
    planner.reset()
    planner.propose("pick the red object", _observation(), _capabilities())

    assert len(chat_a.sent_messages) == 1
    assert len(chat_b.sent_messages) == 1


def test_from_api_key_constructs_without_network_call():
    pytest.importorskip("google.genai")
    planner = GeminiPlanner.from_api_key(api_key="test-key-not-real")
    assert planner is not None
