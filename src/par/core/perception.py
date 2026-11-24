"""The contract a robot adapter must satisfy for the Safety Kernel's
guarantees to mean anything.

`Observation.robot_state` and `Observation.detections` are open
dictionaries on purpose: an adapter needs room for vendor-specific fields,
and PAR should not have to learn a new schema per robot. That openness was
unbounded, and three silent defaults grew in the gap:

  * a robot reporting its pose under a key the kernel did not recognise
    read as the origin, so the kernel swept a straight line from the wrong
    place and pronounced it clear;
  * a detection carrying no position was skipped, so an obstacle the
    perception stack had actually seen bounded nothing;
  * a sensor that failed returned an empty detection list, which is
    indistinguishable from a sensor reporting an empty world.

Every one of those turned "I do not know" into "it is safe" - the one
substitution a governance layer may never make. The parsers here replace
them with explicit, named failures. This module is the only place the
physical world is allowed into the digital one: anything reading
`robot_state` or `detections` to reach a safety decision must come through
here, so that a new adapter either satisfies the contract or is refused,
and never silently voids the guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

PROPRIOCEPTION = "robot_state"
PERCEPTION = "detections"

Point = tuple[float, float, float]
_AXES = ("x", "y", "z")


class PerceptionContractError(Exception):
    """An observation cannot support the decision being asked of it.

    Raised rather than returned so a parser reads linearly; the Safety
    Kernel catches it once and converts it into a DENY, because the kernel
    must always produce a verdict and never propagate an exception out of
    the governance layer.
    """

    def __init__(self, channel: str, detail: str) -> None:
        super().__init__(f"{channel}: {detail}")
        self.channel = channel
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Obstacle:
    """A detection that has been shown to bound motion geometrically."""

    name: str
    position: Point


def to_float(value: Any) -> float | None:
    """A finite float, or None if `value` cannot be one.

    Numeric strings are accepted: a JSON bridge or a language model will
    sometimes emit "1.0" for 1.0, and refusing that would reject a
    well-formed intent over formatting. Booleans are not, because `True`
    would otherwise silently become 1.0. Non-finite values are not either:
    NaN compares False against every bound, so an unchecked NaN would slip
    past a `distance < margin` test as though it were clear.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_point(value: Any) -> Point | None:
    """An (x, y, z) point from a mapping, or None if it is not one.

    A missing axis is not defaulted to zero. Zero is a real position, so
    defaulting to it invents a measurement rather than reporting its
    absence.
    """
    if not isinstance(value, dict):
        return None
    coerced = []
    for axis in _AXES:
        if axis not in value:
            return None
        number = to_float(value[axis])
        if number is None:
            return None
        coerced.append(number)
    return (coerced[0], coerced[1], coerced[2])


def require_position(robot_state: dict[str, Any]) -> Point:
    """Where the robot is, or a contract error naming what was wrong.

    This replaces a silent `or {"x": 0.0, ...}` default. That default was
    the most dangerous line in the kernel: an adapter reporting its pose
    under any other key made every swept-volume check start from the
    origin, so the kernel validated a path the robot was never going to
    take and the soundness argument quietly stopped applying.
    """
    if not isinstance(robot_state, dict):
        raise PerceptionContractError(
            PROPRIOCEPTION, f"expected a mapping, got {type(robot_state).__name__}"
        )
    if "position" not in robot_state:
        raise PerceptionContractError(
            PROPRIOCEPTION,
            "no 'position' reported, so the robot's location is unknown "
            f"(keys present: {sorted(robot_state)})",
        )
    point = parse_point(robot_state["position"])
    if point is None:
        raise PerceptionContractError(
            PROPRIOCEPTION,
            f"position {robot_state['position']!r} is not a mapping of finite "
            "numeric x, y and z",
        )
    return point


def require_obstacles(detections: list[Any]) -> list[Obstacle]:
    """Every detection, as something that bounds motion.

    A detection whose position is missing or malformed raises instead of
    being skipped: the perception stack has asserted that something is
    there, and a kernel that cannot place it cannot bound a path around
    it. An adapter that knows a detection is irrelevant to geometry - a
    floor plane, a 2D-only classification - says so explicitly with
    `ignore_for_collision: true`, which keeps the claim on the record
    instead of leaving the kernel to guess.
    """
    if not isinstance(detections, list):
        raise PerceptionContractError(
            PERCEPTION, f"expected a list, got {type(detections).__name__}"
        )
    obstacles = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise PerceptionContractError(
                PERCEPTION, f"detection {index} is {type(detection).__name__}, not a mapping"
            )
        name = detection.get("name", f"<unnamed #{index}>")
        if detection.get("ignore_for_collision") is True:
            continue
        if "position" not in detection:
            raise PerceptionContractError(
                PERCEPTION,
                f"detection '{name}' has no position, so no path can be bounded "
                "around it (set ignore_for_collision=true if it genuinely does "
                "not constrain motion)",
            )
        point = parse_point(detection["position"])
        if point is None:
            raise PerceptionContractError(
                PERCEPTION,
                f"detection '{name}' has position {detection['position']!r}, which "
                "is not a mapping of finite numeric x, y and z",
            )
        obstacles.append(Obstacle(name=str(name), position=point))
    return obstacles


def _as_utc(moment: datetime) -> datetime:
    """A naive datetime is read as UTC rather than rejected: adapters that
    cross a JSON boundary routinely drop the offset, and treating that as a
    contract violation would deny motion over a serialisation detail."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def require_channel(
    observation: Any,
    channel: str,
    *,
    require_declaration: bool = False,
    max_age_seconds: float | None = None,
    now: datetime | None = None,
) -> None:
    """Check that `channel` is fit to be relied on, or raise.

    Three separable questions, because they have different answers in
    different deployments:

    * Did the adapter say the channel failed? Always enforced. An adapter
      that reports its own blindness must be believed.
    * Did the adapter say anything at all? Enforced only when the profile
      asks (`require_declaration`). An adapter predating this contract
      makes no claim, and whether silence is acceptable is a deployment
      policy, not a property of the data.
    * Is the reading current? Enforced only when the profile sets a
      window. There is no safe universal value - a 1 Hz lidar and a
      1 kHz encoder disagree by three orders of magnitude - so PAR
      refuses to invent one.
    """
    report = observation.channels.get(channel)

    if report is not None and not report.available:
        detail = report.detail or "no reason given"
        raise PerceptionContractError(channel, f"channel reported unavailable: {detail}")

    if report is None and require_declaration:
        raise PerceptionContractError(
            channel,
            "no availability declared, so a failed sensor cannot be told apart "
            "from an empty world (this profile requires an explicit declaration)",
        )

    if max_age_seconds is None:
        return

    # captured_at is when the world was sensed; observation.timestamp is only
    # when the packet was assembled, which a cached reading can keep fresh
    # forever. Prefer the former and fall back to the latter.
    sensed_at = (report.captured_at if report is not None else None) or observation.timestamp
    if sensed_at is None:
        raise PerceptionContractError(
            channel, "no timestamp, so its age cannot be established"
        )
    age = ((now or datetime.now(timezone.utc)) - _as_utc(sensed_at)).total_seconds()
    if age > max_age_seconds:
        raise PerceptionContractError(
            channel, f"reading is {age:.3f}s old, over the {max_age_seconds}s limit"
        )
