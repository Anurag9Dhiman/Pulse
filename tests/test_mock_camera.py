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


def test_a_camera_that_cannot_produce_a_frame_declares_itself_unavailable():
    frame = MockCamera(available=False, detail="device busy").capture()

    assert frame["detections"] == []
    assert frame["available"] is False
    assert frame["detail"] == "device busy"


def test_merging_a_failed_camera_marks_perception_unavailable_on_the_observation():
    # Without the channel report this merge is invisible: the observation
    # comes out looking exactly like one from a working camera that saw an
    # empty room, and the Safety Kernel clears every path through the
    # space the camera was meant to be watching.
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        detections=[{"name": "crate", "position": {"x": 0.5, "y": 0.0, "z": 0.0}}],
    )

    merged = merge_detections(obs, MockCamera(available=False, detail="device busy"))

    assert merged.channels["detections"].available is False
    assert "device busy" in merged.channels["detections"].detail
    # The robot's own detections are kept: the camera failing does not
    # erase what was already known, it only stops the result being treated
    # as a complete picture.
    assert [d["name"] for d in merged.detections] == ["crate"]


def test_merging_a_working_camera_records_perception_as_healthy_and_timestamped():
    obs = Observation(
        observation_id="o1", timestamp=datetime.now(timezone.utc), source="mock"
    )

    merged = merge_detections(obs, MockCamera(detections=[{"name": "blue_object"}]))

    assert merged.channels["detections"].available is True
    assert merged.channels["detections"].captured_at is not None


def test_the_default_mock_detection_carries_a_position_as_the_contract_requires():
    # A detection the kernel cannot place is a contract violation, so the
    # development camera must not ship one by default.
    [detection] = MockCamera().capture()["detections"]
    assert detection["position"] == {"x": 0.5, "y": 0.2, "z": 0.0}
