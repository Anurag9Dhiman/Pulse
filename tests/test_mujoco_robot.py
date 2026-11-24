from __future__ import annotations

from datetime import datetime, timezone

import pytest

from par.core.action import Action

mujoco = pytest.importorskip("mujoco")  # the sim extra is optional

from par.robots.mujoco_robot import MuJoCoRobot  # noqa: E402

_OBSTACLE = ("obstacle", 0.5, 0.0, 0.12)


def _move(x: float, y: float) -> Action:
    return Action(
        action_id="a1",
        skill_name="move",
        parameters={"x": x, "y": y, "z": 0.0},
        created_at=datetime.now(timezone.utc),
    )


def test_observation_reports_position_and_obstacles_in_the_shared_schema():
    robot = MuJoCoRobot(obstacles=[_OBSTACLE])

    observation = robot.get_observation()

    assert observation.source == "mujoco"
    assert observation.robot_state["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert [d["name"] for d in observation.detections] == ["obstacle"]


def test_a_move_along_a_clear_path_succeeds_and_arrives():
    robot = MuJoCoRobot(obstacles=[_OBSTACLE])

    result = robot.execute(_move(0.0, 0.8))

    assert result.success is True
    x, y, _ = robot.position
    assert (round(x, 1), round(y, 1)) == (0.0, 0.8)


def test_a_move_through_an_obstacle_fails_in_physics_not_by_a_safety_check():
    """No Safety Kernel is involved here. The simulator's contact solver is
    what stops the robot, which is what makes it usable as ground truth for
    whether a governed decision was right."""
    robot = MuJoCoRobot(obstacles=[_OBSTACLE])

    result = robot.execute(_move(1.0, 0.0))

    assert result.success is False
    assert "collided" in result.message
    assert robot.last_contacts == ["obstacle"]
    # Blocked at the combined radii (0.08 + 0.12), well short of the target.
    assert robot.position[0] < 0.4


def test_contact_is_reported_only_for_the_obstacle_not_the_floor():
    robot = MuJoCoRobot(obstacles=[_OBSTACLE])

    robot.execute(_move(1.0, 0.0))

    assert "floor" not in robot.last_contacts


def test_an_unknown_skill_is_refused_rather_than_silently_ignored():
    robot = MuJoCoRobot()

    result = robot.execute(
        Action(action_id="a", skill_name="weld", parameters={}, created_at=datetime.now(timezone.utc))
    )

    assert result.success is False
    assert "cannot execute" in result.message
