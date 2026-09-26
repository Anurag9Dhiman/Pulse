from __future__ import annotations

from datetime import datetime, timezone

from par.core.action import Action
from par.evaluation.verifiers import verify_pick_and_place, verify_position_within
from par.robots.mock import MockRobot


def _act(skill_name: str, parameters: dict) -> Action:
    return Action(
        action_id=f"a-{skill_name}", skill_name=skill_name, parameters=parameters, created_at=datetime.now(timezone.utc)
    )


def test_verify_pick_and_place_true_after_real_pick_and_place():
    robot = MockRobot()
    robot.execute(_act("pick", {"object": "red_object"}))
    robot.execute(_act("place", {"target": "blue_container"}))

    verifier = verify_pick_and_place("red_object", "blue_container")
    assert verifier(robot.get_observation()) is True


def test_verify_pick_and_place_false_when_never_placed():
    robot = MockRobot()
    robot.execute(_act("pick", {"object": "red_object"}))

    verifier = verify_pick_and_place("red_object", "blue_container")
    assert verifier(robot.get_observation()) is False


def test_verify_pick_and_place_false_for_unknown_object():
    robot = MockRobot()
    verifier = verify_pick_and_place("nonexistent_object", "blue_container")
    assert verifier(robot.get_observation()) is False


def test_verify_position_within_tolerance():
    robot = MockRobot()
    robot.execute(_act("move", {"x": 1.0, "y": 0.0, "z": 0.0}))

    assert verify_position_within((1.0, 0.0, 0.0), tolerance=0.01)(robot.get_observation()) is True
    assert verify_position_within((5.0, 0.0, 0.0), tolerance=0.01)(robot.get_observation()) is False
