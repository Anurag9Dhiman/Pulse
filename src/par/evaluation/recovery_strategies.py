"""Recovery strategies: what to try next when the Safety Kernel denies a move.

These exist so that "where recovery logic lives" can be varied independently
of "what the recovery logic does". Experiment 4 compares recovery inside the
planner's own loop against recovery in a dedicated module; unless both run
the *same* strategy, that comparison measures the strategies rather than the
architectures. Earlier versions of that experiment did exactly this - the
planner extended its target while the recovery module shortened it - and the
difference went unnoticed because an endpoint-only collision check scored
both as succeeding.

A strategy yields candidate targets in the order they should be attempted.
It never consults the Safety Kernel: a candidate is a *proposal*, and whether
it is permitted is the kernel's decision, arriving as feedback on denial.
"""
from __future__ import annotations

import math
from typing import Iterator, Protocol

from par.core.observation import Observation

Point = tuple[float, float, float]


class RecoveryStrategy(Protocol):
    name: str

    def candidates(self, goal: Point, origin: Point, observation: Observation) -> Iterator[Point]:
        """Targets to try, best first, after the direct move to `goal` is denied."""
        ...


class ExtendStrategy:
    """Push the target further along the same bearing.

    Retained because it is what the original Experiment 4 planner did, and it
    is a useful negative control: under an endpoint-only collision test it
    appears to recover, but once the whole swept path is checked, extending a
    path *through* the obstacle that blocks it cannot help. Measured recovery
    falls from 100% to 15.3% when the collision test is made sound.
    """

    name = "extend"

    def __init__(self, step: float = 0.15, max_candidates: int = 5) -> None:
        self._step = step
        self._max_candidates = max_candidates

    def candidates(self, goal: Point, origin: Point, observation: Observation) -> Iterator[Point]:
        for attempt in range(1, self._max_candidates + 1):
            offset = self._step * attempt
            yield (goal[0] + offset, goal[1] + offset, goal[2])


class BackoffStrategy:
    """Shorten the move toward the origin, the classic fail-safe retreat.

    Always yields a safe action when the obstacle lies near the goal, but by
    construction it never reaches the goal - which is invisible to a success
    metric that only asks whether the robot moved without colliding.
    """

    name = "backoff"

    def __init__(self, factor: float = 0.5, max_candidates: int = 5) -> None:
        self._factor = factor
        self._max_candidates = max_candidates

    def candidates(self, goal: Point, origin: Point, observation: Observation) -> Iterator[Point]:
        for attempt in range(1, self._max_candidates + 1):
            scale = self._factor**attempt
            yield tuple(origin[i] + (goal[i] - origin[i]) * scale for i in range(3))  # type: ignore[misc]


class DetourStrategy:
    """Step aside, perpendicular to the blocked approach, then continue.

    Offsets alternate sides and grow, so the first candidates are the small
    deviations a short detour needs. This is the cheapest strategy that can
    still *reach* a goal whose direct path is blocked: retreating cannot, and
    extending cannot.
    """

    name = "detour"

    def __init__(self, step: float = 0.2, max_candidates: int = 6) -> None:
        self._step = step
        self._max_candidates = max_candidates

    def candidates(self, goal: Point, origin: Point, observation: Observation) -> Iterator[Point]:
        direction = (goal[0] - origin[0], goal[1] - origin[1], goal[2] - origin[2])
        length = math.dist(origin, goal)
        if length == 0.0:
            return
        # Perpendicular to the approach, in the plane the robot drives in.
        perpendicular = (-direction[1] / length, direction[0] / length, 0.0)
        midpoint = tuple((origin[i] + goal[i]) / 2.0 for i in range(3))
        for attempt in range(1, self._max_candidates + 1):
            magnitude = self._step * ((attempt + 1) // 2)
            side = 1.0 if attempt % 2 else -1.0
            yield (
                midpoint[0] + perpendicular[0] * magnitude * side,
                midpoint[1] + perpendicular[1] * magnitude * side,
                midpoint[2],
            )


STRATEGIES = {s.name: s for s in (ExtendStrategy(), BackoffStrategy(), DetourStrategy())}
