from datetime import datetime, timedelta, timezone

import pytest

from par.core.action import Action
from par.core.capability import Capability, RiskLevel
from par.core.observation import ChannelReport, Observation
from par.safety.environment import EnvironmentProfile, Workspace
from par.safety.kernel import SafetyKernel
from par.safety.policy import PolicyOutcome

_PROFILE = EnvironmentProfile(
    name="test",
    workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
    max_velocity=0.5,
    action_timeout_seconds=1.0,
    approval_required=False,
)

_MOVE_CAPABILITY = Capability(name="move", description="move", env_profiles=["test"])


def _observation(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Observation:
    return Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": position[0], "y": position[1], "z": position[2]}},
    )


def _move_action(x: float, y: float, z: float) -> Action:
    return Action(
        action_id="a1",
        skill_name="move",
        parameters={"x": x, "y": y, "z": z},
        created_at=datetime.now(timezone.utc),
    )


def test_admit_denies_capability_not_registered_for_profile():
    kernel = SafetyKernel(_PROFILE)
    other = Capability(name="fly", description="flies", env_profiles=["other_profile"])
    assert kernel.admit(other).outcome == PolicyOutcome.DENY


def test_admit_allows_registered_capability():
    kernel = SafetyKernel(_PROFILE)
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


def test_check_denies_move_outside_workspace():
    kernel = SafetyKernel(_PROFILE)
    action = _move_action(5.0, 0.0, 0.0)  # workspace x bound is [-1, 1]
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.DENY


def test_check_allows_slow_move_within_bounds():
    kernel = SafetyKernel(_PROFILE)  # max_velocity=0.5, timeout=1.0s -> max distance 0.5
    action = _move_action(0.3, 0.0, 0.0)
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.ALLOW


def test_check_modifies_move_exceeding_velocity_but_within_workspace():
    kernel = SafetyKernel(_PROFILE)
    action = _move_action(0.9, 0.0, 0.0)  # inside workspace, but distance 0.9 > max 0.5
    decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.MODIFY
    assert decision.modified_parameters is not None
    assert decision.modified_parameters["x"] == pytest.approx(0.5, abs=0.01)


def test_emergency_stop_denies_everything():
    kernel = SafetyKernel(_PROFILE)
    kernel.emergency_stop()
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.DENY
    action = _move_action(0.1, 0.0, 0.0)
    assert kernel.check(action, _observation(), _MOVE_CAPABILITY).outcome == PolicyOutcome.DENY


def test_clear_emergency_stop_restores_normal_operation():
    kernel = SafetyKernel(_PROFILE)
    kernel.emergency_stop()
    kernel.clear_emergency_stop()
    assert kernel.admit(_MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


def test_check_denies_move_that_collides_with_detected_object():
    kernel = SafetyKernel(_PROFILE)
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "obstacle_1", "position": {"x": 0.5, "y": 0.0, "z": 0.0}}],
    )
    action = _move_action(0.5, 0.0, 0.0)  # lands exactly on the obstacle
    decision = kernel.check(action, obs, _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.DENY
    assert "collision" in decision.reason


def test_check_allows_move_far_from_detected_objects():
    kernel = SafetyKernel(_PROFILE)
    obs = Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "obstacle_1", "position": {"x": -0.9, "y": -0.9, "z": 0.0}}],
    )
    action = _move_action(0.3, 0.0, 0.0)  # far from the obstacle, within velocity limit
    decision = kernel.check(action, obs, _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.ALLOW


def test_check_escalates_high_risk_action_when_approval_required():
    profile = EnvironmentProfile(
        name="test-approval",
        workspace=Workspace(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 2.0)),
        max_velocity=0.5,
        action_timeout_seconds=1.0,
        approval_required=True,
    )
    kernel = SafetyKernel(profile)
    capability = Capability(name="pick", description="pick", risk=RiskLevel.HIGH, env_profiles=["test-approval"])
    action = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "red_object"},
        created_at=datetime.now(timezone.utc),
    )
    decision = kernel.check(action, _observation(), capability)
    assert decision.outcome == PolicyOutcome.ESCALATE


# --------------------------------------------------------------------------
# Trajectory-level safety. Both cases below passed the endpoint-only collision
# test that preceded them, which is why the benchmark (whose collision cases
# place the obstacle exactly at the target) never exposed either one.
# --------------------------------------------------------------------------


def _observation_with_obstacle(
    position: tuple[float, float, float], obstacle: tuple[float, float, float]
) -> Observation:
    return Observation(
        observation_id="o1",
        timestamp=datetime.now(timezone.utc),
        source="mock",
        robot_state={"position": {"x": position[0], "y": position[1], "z": position[2]}},
        detections=[
            {"name": "obstacle", "position": {"x": obstacle[0], "y": obstacle[1], "z": obstacle[2]}}
        ],
    )


def test_move_whose_path_crosses_an_obstacle_is_denied_though_its_endpoint_is_clear():
    # The endpoint (0.9,0,0) sits 0.9m from the obstacle, far outside the
    # margin; only sweeping the segment reveals that the robot would have to
    # drive straight over it to get there.
    kernel = SafetyKernel(_PROFILE)
    observation = _observation_with_obstacle((-0.9, 0.0, 0.0), (0.0, 0.0, 0.0))

    decision = kernel.check(_move_action(0.9, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "collision" in decision.reason


def test_obstacle_behind_the_robot_does_not_block_a_move_away_from_it():
    # The guard against over-correcting: measuring to the infinite line
    # rather than the segment would deny this, since the obstacle is
    # collinear with the direction of travel but in the opposite direction.
    kernel = SafetyKernel(_PROFILE)
    observation = _observation_with_obstacle((0.0, 0.0, 0.0), (-0.5, 0.0, 0.0))

    decision = kernel.check(_move_action(0.4, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.ALLOW


def test_modify_never_returns_a_clamped_target_the_kernel_would_itself_deny():
    """The soundness property MODIFY must satisfy: whatever it hands back has
    to pass the same checks a planner-proposed action would. Before the swept
    collision test, clamping a long over-speed move could land the robot on an
    obstacle that the original distant target cleared."""
    kernel = SafetyKernel(_PROFILE)
    reach = _PROFILE.max_velocity * _PROFILE.action_timeout_seconds
    # The obstacle sits exactly where the clamp lands.
    observation = _observation_with_obstacle((0.0, 0.0, 0.0), (reach, 0.0, 0.0))

    decision = kernel.check(_move_action(1.0, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    if decision.modified_parameters is not None:
        clamped = _move_action(**decision.modified_parameters)
        assert kernel.check(clamped, observation, _MOVE_CAPABILITY).outcome != PolicyOutcome.DENY


def test_a_clamped_move_along_a_clear_path_is_still_modified_not_denied():
    # Re-validation must not turn every clamp into a denial: with no obstacle
    # on the segment, an over-speed move still clamps and proceeds.
    kernel = SafetyKernel(_PROFILE)

    decision = kernel.check(_move_action(1.0, 0.0, 0.0), _observation(), _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.MODIFY
    reach = _PROFILE.max_velocity * _PROFILE.action_timeout_seconds
    assert decision.modified_parameters["x"] == pytest.approx(reach)


def test_clamping_is_revalidated_when_the_robot_starts_outside_the_workspace():
    """The one case the subset argument does not cover. Clamping is safe
    because the clamped target lies on the validated segment - but if the
    robot is already outside the workspace, that segment starts outside it,
    so the clamped point can be outside too and must be re-checked."""
    kernel = SafetyKernel(_PROFILE)
    # Robot parked beyond the x bound; the requested target is inside.
    observation = _observation((-3.0, 0.0, 0.0))

    decision = kernel.check(_move_action(0.5, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "still unsafe" in decision.reason


def test_non_numeric_move_parameters_are_denied_rather_than_raising():
    """A planner is untrusted input. Small language models emit "1.0" or null
    for a coordinate; if the kernel raises on that, the exception escapes the
    governance layer and no verdict is recorded for the action at all."""
    kernel = SafetyKernel(_PROFILE)

    for bad in ({"x": "left", "y": 0.0, "z": 0.0}, {"x": None, "y": 0.0, "z": 0.0}):
        action = Action(
            action_id="a1", skill_name="move", parameters=bad, created_at=datetime.now(timezone.utc)
        )
        decision = kernel.check(action, _observation(), _MOVE_CAPABILITY)
        assert decision.outcome == PolicyOutcome.DENY
        assert "invalid move parameters" in decision.reason


def test_numeric_strings_are_coerced_rather_than_denied_on_formatting():
    kernel = SafetyKernel(_PROFILE)
    action = Action(
        action_id="a1",
        skill_name="move",
        parameters={"x": "0.4", "y": "0", "z": "0"},
        created_at=datetime.now(timezone.utc),
    )

    assert kernel.check(action, _observation(), _MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


# ---------------------------------------------------------------------------
# Unknown is not safe.
#
# Every test below used to produce ALLOW. The kernel reasoned about a world
# the observation did not actually describe - a missing position read as the
# origin, an unplaceable detection was skipped, a dead sensor's empty
# detection list read as open space - and then certified motion through it.
# ---------------------------------------------------------------------------


def _bare_observation(**kwargs) -> Observation:
    defaults = dict(observation_id="o1", timestamp=datetime.now(timezone.utc), source="test")
    defaults.update(kwargs)
    return Observation(**defaults)


def test_a_move_is_denied_when_the_robot_does_not_report_where_it_is():
    kernel = SafetyKernel(_PROFILE)
    decision = kernel.check(_move_action(0.3, 0.0, 0.0), _bare_observation(), _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "cannot verify this move against the world" in decision.reason


def test_a_move_is_denied_when_the_robot_reports_its_pose_under_an_unrecognised_key():
    # The third-party-adapter case: a robot reporting "pose" instead of
    # "position" silently placed itself at the origin, so the swept check
    # validated a segment starting somewhere the robot was not.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(robot_state={"pose": {"x": 0.9, "y": 0.9, "z": 0.0}})

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "robot_state" in decision.reason


def test_a_move_is_denied_when_a_detection_cannot_be_placed():
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "person", "confidence": 0.99}],  # seen, but not located
    )

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "person" in decision.reason


def test_a_detection_explicitly_marked_irrelevant_to_geometry_does_not_block_motion():
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[{"name": "floor", "ignore_for_collision": True}],
    )

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.ALLOW


def test_a_dead_sensor_is_denied_even_though_it_reports_an_empty_world():
    # The headline case. An empty detection list is ambiguous: it means
    # either "I looked and the space is clear" or "I could not look".
    # Without the channel report the kernel could not tell them apart and
    # chose the permissive reading.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[],
        channels={"detections": ChannelReport(available=False, detail="lidar timed out")},
    )

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "lidar timed out" in decision.reason


def test_the_same_empty_world_is_allowed_when_the_sensor_says_it_is_working():
    # The other half of the previous test: the fix must not simply deny
    # more, it must deny exactly the unknown cases.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        detections=[],
        channels={"detections": ChannelReport(available=True, captured_at=datetime.now(timezone.utc))},
    )

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.ALLOW


def test_a_stale_observation_is_denied_when_the_profile_sets_an_age_limit():
    profile = _PROFILE.model_copy(update={"observation_max_age_seconds": 0.5})
    kernel = SafetyKernel(profile)
    observation = _bare_observation(
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=10),
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )

    decision = kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "old" in decision.reason


def test_the_same_stale_observation_is_accepted_when_no_age_limit_is_configured():
    # Freshness is deployment policy, not a universal truth: a 1 Hz lidar
    # and a 1 kHz encoder disagree by three orders of magnitude, so the
    # shipped profiles set no window and nothing silently changes for them.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=10),
        robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )

    assert kernel.check(_move_action(0.3, 0.0, 0.0), observation, _MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW


def test_an_adapter_that_declares_nothing_is_denied_only_under_a_profile_that_requires_it():
    silent = _bare_observation(robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    action = _move_action(0.3, 0.0, 0.0)

    lenient = SafetyKernel(_PROFILE)
    assert lenient.check(action, silent, _MOVE_CAPABILITY).outcome == PolicyOutcome.ALLOW

    strict = SafetyKernel(_PROFILE.model_copy(update={"require_declared_perception": True}))
    decision = strict.check(action, silent, _MOVE_CAPABILITY)
    assert decision.outcome == PolicyOutcome.DENY
    assert "no availability declared" in decision.reason


def test_a_nan_move_target_is_denied_rather_than_slipping_past_the_collision_test():
    # NaN compares False against every bound, so `clearance < margin` would
    # read as clear for a target that is not a position at all.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(robot_state={"position": {"x": 0.0, "y": 0.0, "z": 0.0}})

    decision = kernel.check(_move_action(float("nan"), 0.0, 0.0), observation, _MOVE_CAPABILITY)

    assert decision.outcome == PolicyOutcome.DENY
    assert "invalid move parameters" in decision.reason


def test_a_non_move_action_is_unaffected_by_perception_faults():
    # The contract is enforced where the kernel actually depends on the
    # data. `stop` consults no world model, and must stay available
    # precisely when sensors have failed - denying it would be the
    # dangerous choice.
    kernel = SafetyKernel(_PROFILE)
    observation = _bare_observation(
        channels={"detections": ChannelReport(available=False, detail="lidar timed out")}
    )
    stop = Action(action_id="a2", skill_name="stop", parameters={}, created_at=datetime.now(timezone.utc))

    decision = kernel.check(stop, observation, Capability(name="stop", description="stop", env_profiles=["test"]))

    assert decision.outcome == PolicyOutcome.ALLOW
