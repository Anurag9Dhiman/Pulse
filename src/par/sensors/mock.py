from __future__ import annotations

from typing import Any

from par.sensors.base import CameraSource


class MockCamera(CameraSource):
    """In-memory camera for development: no OpenCV or hardware needed."""

    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self._detections = detections or [{"name": "red_object", "confidence": 0.95}]

    def capture(self) -> dict[str, Any]:
        return {"detections": list(self._detections)}
