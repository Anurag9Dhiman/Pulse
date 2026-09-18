from datetime import datetime, timezone

from par.core.observation import Observation


def test_observation_defaults():
    obs = Observation(
        observation_id="obs-1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
    )
    assert obs.robot_state == {}
    assert obs.detections == []
