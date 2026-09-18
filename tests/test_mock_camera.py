from datetime import datetime, timezone

from par.core.observation import Observation
from par.sensors.base import merge_detections
from par.sensors.mock import MockCamera


def test_mock_camera_returns_detections():
    camera = MockCamera(detections=[{"name": "green_object"}])
    frame = camera.capture()
    assert frame["detections"] == [{"name": "green_object"}]


def test_merge_detections_appends_to_observation():
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        detections=[{"name": "red_object"}],
    )
    camera = MockCamera(detections=[{"name": "blue_object"}])
    merged = merge_detections(obs, camera)

    names = {d["name"] for d in merged.detections}
    assert names == {"red_object", "blue_object"}
    assert obs.detections == [{"name": "red_object"}]  # original untouched
