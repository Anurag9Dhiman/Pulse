from datetime import datetime, timezone

from par.core.action import Action, ActionStatus


def test_action_defaults_to_pending():
    action = Action(
        action_id="act-1",
        skill_name="stop",
        created_at=datetime.now(timezone.utc),
    )
    assert action.status == ActionStatus.PENDING
    assert action.parameters == {}
