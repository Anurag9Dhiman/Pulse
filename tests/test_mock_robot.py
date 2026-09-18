from datetime import datetime, timezone

from par.core.action import Action
from par.robots.mock import MockRobot


def test_observation_lists_known_objects(robot: MockRobot):
    obs = robot.get_observation()
    names = {d["name"] for d in obs.detections}
    assert {"red_object", "blue_container"} <= names


def test_pick_unknown_object_fails(robot: MockRobot):
    action = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "green_object"},
        created_at=datetime.now(timezone.utc),
    )
    result = robot.execute(action)
    assert result.success is False


def test_pick_then_place_round_trip(robot: MockRobot):
    pick = Action(
        action_id="a1",
        skill_name="pick",
        parameters={"object": "red_object"},
        created_at=datetime.now(timezone.utc),
    )
    assert robot.execute(pick).success is True

    place = Action(
        action_id="a2",
        skill_name="place",
        parameters={"target": "blue_container"},
        created_at=datetime.now(timezone.utc),
    )
    result = robot.execute(place)
    assert result.success is True
    assert "red_object" in result.message


def test_add_obstacle_appears_in_next_observation(robot: MockRobot):
    robot.add_obstacle("obstacle_1", 1.0, 1.0, 0.0)
    obs = robot.get_observation()
    obstacle = next(d for d in obs.detections if d["name"] == "obstacle_1")
    assert obstacle["position"] == {"x": 1.0, "y": 1.0, "z": 0.0}
