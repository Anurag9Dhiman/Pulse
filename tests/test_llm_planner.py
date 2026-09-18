from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from par.core.action import Action, ActionResult
from par.core.capability import Capability
from par.core.llm_planner import LLMPlanner
from par.core.observation import Observation
from par.core.planner import TASK_COMPLETE, PlannerError


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "tool_1"
    type: str = "tool_use"


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeResponse:
    content: list[Any]


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        # Snapshot messages at call time - self._history keeps mutating (more
        # turns get appended) after this call returns, and a live reference
        # would make every earlier recorded call look like the final state.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.messages = FakeMessages(responses)


def _observation() -> Observation:
    return Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        detections=[{"name": "red_object"}],
    )


def _capabilities() -> list[Capability]:
    return [Capability(name="pick", description="pick up an object", input_schema={"object": "str"})]


def test_propose_returns_tool_call_from_response():
    response = FakeResponse(content=[FakeToolUseBlock(name="pick", input={"object": "red_object"})])
    client = FakeClient([response])
    planner = LLMPlanner(client)

    name, params = planner.propose("pick the red object", _observation(), _capabilities())

    assert name == "pick"
    assert params == {"object": "red_object"}


def test_tools_sent_to_client_include_task_complete():
    response = FakeResponse(content=[FakeToolUseBlock(name="pick", input={"object": "red_object"})])
    client = FakeClient([response])
    planner = LLMPlanner(client)

    planner.propose("pick the red object", _observation(), _capabilities())

    tool_names = {t["name"] for t in client.messages.calls[0]["tools"]}
    assert tool_names == {"pick", TASK_COMPLETE}


def test_missing_tool_call_raises_planner_error():
    response = FakeResponse(content=[FakeTextBlock(text="I'm thinking about it")])
    client = FakeClient([response])
    planner = LLMPlanner(client)

    with pytest.raises(PlannerError):
        planner.propose("pick the red object", _observation(), _capabilities())


def test_record_result_feeds_back_into_next_prompt():
    first = FakeResponse(
        content=[FakeToolUseBlock(name="pick", input={"object": "red_object"}, id="tu_1")]
    )
    second = FakeResponse(content=[FakeToolUseBlock(name=TASK_COMPLETE, input={"message": "done"})])
    client = FakeClient([first, second])
    planner = LLMPlanner(client)

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
    second_call_messages = client.messages.calls[1]["messages"]
    last_user_message = second_call_messages[-1]
    tool_result_blocks = [b for b in last_user_message["content"] if b["type"] == "tool_result"]
    assert tool_result_blocks[0]["tool_use_id"] == "tu_1"
    assert tool_result_blocks[0]["content"] == "picked"


def test_reset_starts_fresh_conversation():
    response = FakeResponse(content=[FakeToolUseBlock(name="pick", input={"object": "red_object"})])
    client = FakeClient([response, response])
    planner = LLMPlanner(client)

    planner.propose("pick the red object", _observation(), _capabilities())
    planner.reset()
    planner.propose("pick the red object", _observation(), _capabilities())

    assert len(client.messages.calls[1]["messages"]) == 1


def test_from_api_key_constructs_without_network_call():
    pytest.importorskip("anthropic")
    planner = LLMPlanner.from_api_key(api_key="test-key-not-real")
    assert planner is not None
