from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from par.sensors.base import CameraSource


class MockCamera(CameraSource):
    """In-memory camera for development and tests.

    `available=False` simulates a camera that cannot produce a frame,
    which is the case worth testing: the Safety Kernel must deny motion
    then, not treat the empty detection list as an empty room.

    The default detection carries a position because the perception
    contract requires one - a detection the kernel cannot place is a
    contract violation, not an obstacle it may quietly ignore.
    """

    def __init__(
        self,
        detections: list[dict[str, Any]] | None = None,
        *,
        available: bool = True,
        detail: str = "",
    ) -> None:
        self._detections = detections or [
            {"name": "red_object", "confidence": 0.95, "position": {"x": 0.5, "y": 0.2, "z": 0.0}}
        ]
        self._available = available
        self._detail = detail

    def capture(self) -> dict[str, Any]:
        if not self._available:
            return {
                "detections": [],
                "available": False,
                "detail": self._detail or "mock camera unavailable",
            }
        return {
            "detections": list(self._detections),
            "available": True,
            "captured_at": datetime.now(timezone.utc),
        }
