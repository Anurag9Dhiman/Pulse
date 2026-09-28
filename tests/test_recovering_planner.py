from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action, ActionResult
from par.core.planner import TASK_COMPLETE
from par.evaluation.recovering_planner import RecoveringMovePlanner


def _result(success: bool, message: str = "") -> ActionResult:
    return ActionResult(action_id="a", success=success, message=message, completed_at=datetime.now(timezone.utc))


def _action(x: float, y: float, z: float) -> Action:
    return Action(action_id="a", skill_name="move", parameters={"x": x, "y": y, "z": z}, created_at=datetime.now(timezone.utc))


def test_first_proposal_is_the_original_target():
    planner = RecoveringMovePlanner(target=(1.0, 1.0, 0.0))
    name, params = planner.propose("goal", None, [])
    assert name == "move"
    assert params == {"x": 1.0, "y": 1.0, "z": 0.0}


def test_success_on_first_attempt_reports_task_complete_success():
    planner = RecoveringMovePlanner(target=(1.0, 1.0, 0.0))
    _, params = planner.propose("goal", None, [])
    planner.record_result(_action(**params), _result(True, "moved"))

    name, params = planner.propose("goal", None, [])
    assert name == TASK_COMPLETE
    assert params["success"] is True


def test_denial_produces_a_different_target_next_attempt():
    planner = RecoveringMovePlanner(target=(1.0, 1.0, 0.0), step=0.15)
    _, first = planner.propose("goal", None, [])
    planner.record_result(_action(**first), _result(False, "collision risk"))

    name, second = planner.propose("goal", None, [])
    assert name == "move"
    assert second != first
    assert planner.denial_count == 1


def test_gives_up_honestly_after_max_attempts():
    planner = RecoveringMovePlanner(target=(1.0, 1.0, 0.0), max_attempts=3)
    for _ in range(3):
        _, params = planner.propose("goal", None, [])
        planner.record_result(_action(**params), _result(False, "always denied"))

    name, params = planner.propose("goal", None, [])
    assert name == TASK_COMPLETE
    assert params["success"] is False
    assert "gave up" in params["message"]


def test_reset_clears_attempt_state():
    planner = RecoveringMovePlanner(target=(1.0, 1.0, 0.0))
    _, params = planner.propose("goal", None, [])
    planner.record_result(_action(**params), _result(False, "denied"))
    assert planner.denial_count == 1

    planner.reset()
    assert planner.denial_count == 0
    name, params = planner.propose("goal", None, [])
    assert params == {"x": 1.0, "y": 1.0, "z": 0.0}
