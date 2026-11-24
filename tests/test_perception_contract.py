"""The physical-to-digital contract: every silent default replaced by an
explicit refusal."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from par.core.observation import ChannelReport, Observation
from par.core.perception import (
    PERCEPTION,
    PROPRIOCEPTION,
    PerceptionContractError,
    parse_point,
    require_channel,
    require_obstacles,
    require_position,
    to_float,
)


def _observation(**kwargs) -> Observation:
    defaults = dict(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="test",
    )
    defaults.update(kwargs)
    return Observation(**defaults)


# ----------------------------------------------------------------- to_float


@pytest.mark.parametrize("value,expected", [(1, 1.0), (1.5, 1.5), ("1.0", 1.0), ("-2", -2.0)])
def test_numbers_and_numeric_strings_are_accepted(value, expected):
    # A JSON bridge or a language model will emit "1.0" for 1.0; refusing
    # that would deny a well-formed intent over formatting alone.
    assert to_float(value) == expected


@pytest.mark.parametrize("value", [True, False, None, "abc", "", [], {}, object()])
def test_non_numeric_values_are_rejected(value):
    # True is rejected specifically: bool is an int subclass, so it would
    # otherwise become 1.0 and read as a real coordinate.
    assert to_float(value) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_non_finite_values_are_rejected_because_nan_defeats_every_bound(value):
    # NaN compares False against everything, so an unchecked NaN would slip
    # through a `clearance < margin` test as though the path were clear.
    assert to_float(value) is None


# --------------------------------------------------------------- parse_point


def test_a_point_needs_all_three_axes_present():
    assert parse_point({"x": 1.0, "y": 2.0, "z": 3.0}) == (1.0, 2.0, 3.0)
    # A missing axis is not defaulted to zero: zero is a real position, so
    # defaulting invents a measurement instead of reporting its absence.
    assert parse_point({"x": 1.0, "y": 2.0}) is None


@pytest.mark.parametrize("value", [None, "0,0,0", [0, 0, 0], {"x": "left", "y": 0, "z": 0}])
def test_a_malformed_point_is_not_a_point(value):
    assert parse_point(value) is None


# ----------------------------------------------------------- require_position


def test_position_is_returned_when_the_robot_reports_one_properly():
    assert require_position({"position": {"x": 0.5, "y": -0.5, "z": 0.0}}) == (0.5, -0.5, 0.0)


def test_a_robot_that_reports_its_pose_under_another_key_is_refused_not_read_as_the_origin():
    # The bug this exists to kill: an adapter using "pose" instead of
    # "position" used to read as (0, 0, 0), so every swept-volume check
    # started from the wrong place and certified a path the robot was
    # never going to take.
    with pytest.raises(PerceptionContractError) as exc:
        require_position({"pose": {"x": 0.9, "y": 0.9, "z": 0.0}})
    assert exc.value.channel == PROPRIOCEPTION
    assert "pose" in str(exc.value)  # the error names what *was* reported


def test_a_malformed_position_is_refused():
    with pytest.raises(PerceptionContractError):
        require_position({"position": {"x": None, "y": 0.0, "z": 0.0}})


def test_a_robot_state_that_is_not_a_mapping_is_refused():
    with pytest.raises(PerceptionContractError):
        require_position(["position", 1, 2, 3])


# ---------------------------------------------------------- require_obstacles


def test_well_formed_detections_become_obstacles():
    obstacles = require_obstacles(
        [
            {"name": "crate", "position": {"x": 0.5, "y": 0.0, "z": 0.0}},
            {"name": "person", "position": {"x": -0.5, "y": 0.2, "z": 0.0}},
        ]
    )
    assert [o.name for o in obstacles] == ["crate", "person"]
    assert obstacles[0].position == (0.5, 0.0, 0.0)


def test_a_detection_without_a_position_is_refused_rather_than_skipped():
    # Previously skipped silently, which meant an obstacle the perception
    # stack had genuinely seen bounded nothing at all.
    with pytest.raises(PerceptionContractError) as exc:
        require_obstacles([{"name": "unplaced_person", "confidence": 0.9}])
    assert exc.value.channel == PERCEPTION
    assert "unplaced_person" in str(exc.value)


def test_a_detection_may_opt_out_of_collision_checking_but_must_say_so_explicitly():
    # A floor plane or a 2D-only classification genuinely does not bound
    # motion. The adapter is allowed to say that - it just has to say it,
    # so the claim is on the record instead of being inferred from a
    # missing key.
    obstacles = require_obstacles(
        [
            {"name": "floor", "ignore_for_collision": True},
            {"name": "crate", "position": {"x": 0.5, "y": 0.0, "z": 0.0}},
        ]
    )
    assert [o.name for o in obstacles] == ["crate"]


def test_a_detection_with_a_malformed_position_is_refused():
    with pytest.raises(PerceptionContractError):
        require_obstacles([{"name": "crate", "position": {"x": "left", "y": 0.0, "z": 0.0}}])


def test_a_detection_list_that_is_not_a_list_is_refused():
    with pytest.raises(PerceptionContractError):
        require_obstacles({"name": "crate"})


def test_a_detection_that_is_not_a_mapping_is_refused():
    with pytest.raises(PerceptionContractError):
        require_obstacles(["crate"])


# ------------------------------------------------------------ require_channel


def test_a_channel_declaring_itself_unavailable_is_always_refused():
    # No profile flag gates this: an adapter that reports its own
    # blindness must be believed.
    obs = _observation(channels={PERCEPTION: ChannelReport(available=False, detail="camera offline")})
    with pytest.raises(PerceptionContractError) as exc:
        require_channel(obs, PERCEPTION)
    assert "camera offline" in str(exc.value)


def test_an_undeclared_channel_passes_by_default_so_older_adapters_keep_working():
    require_channel(_observation(), PERCEPTION)  # no raise


def test_an_undeclared_channel_is_refused_when_the_profile_demands_a_declaration():
    with pytest.raises(PerceptionContractError) as exc:
        require_channel(_observation(), PERCEPTION, require_declaration=True)
    assert "no availability declared" in str(exc.value)


def test_a_reading_inside_the_age_window_is_accepted():
    require_channel(_observation(), PERCEPTION, max_age_seconds=1.0)  # no raise


def test_a_reading_older_than_the_window_is_refused():
    stale = datetime.now(timezone.utc) - timedelta(seconds=5)
    with pytest.raises(PerceptionContractError) as exc:
        require_channel(_observation(timestamp=stale), PERCEPTION, max_age_seconds=1.0)
    assert "old" in str(exc.value)


def test_freshness_uses_when_the_world_was_sensed_not_when_the_packet_was_built():
    # The ROS 2 cached-payload case. Observation.timestamp is set at
    # packaging, so an adapter re-serving its last message looks brand new
    # forever; captured_at is the only field that can expose it.
    obs = _observation(
        timestamp=datetime.now(timezone.utc),  # packaged right now
        channels={
            PERCEPTION: ChannelReport(
                available=True, captured_at=datetime.now(timezone.utc) - timedelta(seconds=30)
            )
        },
    )
    with pytest.raises(PerceptionContractError):
        require_channel(obs, PERCEPTION, max_age_seconds=1.0)


def test_a_naive_captured_at_is_read_as_utc_rather_than_crashing_the_check():
    # Adapters crossing a JSON boundary routinely drop the offset. Denying
    # motion over a serialisation detail would be its own failure mode.
    naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
    obs = _observation(channels={PERCEPTION: ChannelReport(captured_at=naive_now)})
    require_channel(obs, PERCEPTION, max_age_seconds=5.0)  # no raise
